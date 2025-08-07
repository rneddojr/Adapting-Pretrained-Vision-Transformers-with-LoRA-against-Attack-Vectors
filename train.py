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

        pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.4f}'
        })

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

        pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.4f}'
        })

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

        # Handle CLIP dtype
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


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Recognition Training')
    parser.add_argument('--data_root', default='./processed',
                        help='Root directory of processed dataset')
    parser.add_argument('--model', required=True,
                        choices=['dinov2', 'swin', 'vit', 'clip', 'googlenet', 'resnet', 'convnext'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output_dir', default='./results',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Saving all outputs to: {model_output_dir}")

    mean, std = get_normalization(args.model)

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

    train_dataset = TrafficSignDataset(
        os.path.join(args.data_root, 'train', 'images'),
        os.path.join(args.data_root, 'train', 'metadata.csv'),
        transform=train_transform
    )

    class_to_idx = train_dataset.class_to_idx
    print(f"Total classes: {len(class_to_idx)}")

    val_dataset = TrafficSignDataset(
        os.path.join(args.data_root, 'val', 'images'),
        os.path.join(args.data_root, 'val', 'metadata.csv'),
        transform=test_transform,
        class_to_idx=class_to_idx
    )

    test_dataset = TrafficSignDataset(
        os.path.join(args.data_root, 'test', 'images'),
        os.path.join(args.data_root, 'test', 'metadata.csv'),
        transform=test_transform,
        class_to_idx=class_to_idx
    )

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Using normalization: mean={mean}, std={std}")

    model = create_model(args.model, len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    class_mapping_path = os.path.join(model_output_dir, 'class_mappings.txt')
    with open(class_mapping_path, 'w') as f:
        for idx, cls_name in train_dataset.idx_to_class.items():
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
            best_model_path = os.path.join(model_output_dir, f"{args.model}_best_model_finetuned.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to: {best_model_path}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")

    final_model_path = os.path.join(model_output_dir, f"{args.model}_final_model_finetuned.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(model_output_dir, f"{args.model}_best_model_finetuned.pth")
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_f1 = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

    results_path = os.path.join(model_output_dir, 'results.csv')
    results = {
        'model': args.model,
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


if __name__ == '__main__':
    main()