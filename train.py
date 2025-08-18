import os
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from Utils import TrafficSignDataset, create_model, get_normalization


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    pbar = tqdm(dataloader, desc='Training', leave=False)

    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
            inputs = inputs.to(model.visual.conv1.weight.dtype)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.4f}'})

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc='Validation', leave=False)

    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        total_samples += batch_size

        if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
            inputs = inputs.to(model.visual.conv1.weight.dtype)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data).item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1


def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc='Testing')

    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        if hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
            inputs = inputs.to(model.visual.conv1.weight.dtype)

        with torch.no_grad():
            outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1


def train_model(model_name, args, selected_sources, device):
    all_sources = ['gtsrb', 'lisa', 'roboflow', 'mapillary']
    using_all_sources = set(selected_sources) == set(all_sources)

    if using_all_sources:
        source_dir = 'unified'
    else:
        source_dir = '_'.join(sorted(selected_sources))

    model_output_dir = os.path.join(args.output_dir, model_name, source_dir)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"\nTraining {model_name.upper()} model")
    print(f"Saving outputs to: {model_output_dir}")

    mean, std = get_normalization(model_name)

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if model_name.startswith('dinov3_'):
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(518),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def load_and_filter_split(split):
        meta_path = os.path.join(args.data_root, split, 'metadata.csv')
        if not os.path.exists(meta_path):
            return pd.DataFrame()
        df = pd.read_csv(meta_path)
        return df[df['source'].isin(selected_sources)]

    train_meta = load_and_filter_split('train')
    val_meta = load_and_filter_split('val')
    test_meta = load_and_filter_split('test')

    all_classes = set(train_meta['unified_class'].unique())
    all_classes.update(val_meta['unified_class'].unique())
    all_classes.update(test_meta['unified_class'].unique())

    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"Actual classes in {selected_sources}: {len(class_to_idx)}")

    def create_filtered_dataset(split, meta_df, transform):
        temp_meta_path = os.path.join(model_output_dir, f'temp_{split}_metadata.csv')
        meta_df.to_csv(temp_meta_path, index=False)

        dataset = TrafficSignDataset(
            root_dir=args.data_root,
            metadata_file=temp_meta_path,
            transform=transform,
            class_to_idx=class_to_idx
        )

        os.remove(temp_meta_path)
        return dataset

    train_dataset = create_filtered_dataset('train', train_meta, train_transform)
    val_dataset = create_filtered_dataset('val', val_meta, test_transform)
    test_dataset = create_filtered_dataset('test', test_meta, test_transform)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Using normalization: mean={mean}, std={std}")

    model = create_model(model_name, len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    class_mapping_path = os.path.join(model_output_dir, 'class_mappings.txt')
    with open(class_mapping_path, 'w') as f:
        for cls_name, idx in class_to_idx.items():
            f.write(f"{idx}: {cls_name}\n")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_output_dir, f"{model_name}_best_model_finetuned.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")

    final_model_path = os.path.join(model_output_dir, f"{model_name}_final_model_finetuned.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(model_output_dir, f"{model_name}_best_model_finetuned.pth")
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_f1 = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    results_path = os.path.join(model_output_dir, 'results.csv')
    results = {
        'model': model_name,
        'sources': ','.join(selected_sources),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'normalization_mean': mean,
        'normalization_std': std,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'num_classes': len(class_to_idx),
        'training_time': training_time,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_val_acc': best_val_acc
    }

    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            writer.writerow([key, value])

    print(f"Training results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Recognition Training')
    parser.add_argument('--data_root', default='./Datasets/Adjust_Global_Dataset', help='Root directory of unified dataset')
    parser.add_argument('--model', required=True, nargs='+',
                        choices=['dinov1', 'dinov3', 'swin', 'google_vit', 'convnext', 'yolov11'],
                        help='Model architecture(s) to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output_dir', default='./results', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sources', default='all',
                        help='Comma-separated list of sources to include (gtsrb,lisa,roboflow,mapillary) or "all"')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.sources.lower() == 'all':
        selected_sources = ['gtsrb', 'lisa', 'roboflow', 'mapillary']
    else:
        selected_sources = [s.strip() for s in args.sources.split(',')]
    print(f"Using data from sources: {', '.join(selected_sources)}")

    for model_name in args.model:
        train_model(model_name, args, selected_sources, device)


if __name__ == '__main__':
    main()

