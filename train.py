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
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from Utils import TrafficSignDataset, create_vit_model, get_normalization


def get_model_output(outputs):
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    else:
        return outputs


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

        optimizer.zero_grad()
        outputs = model(inputs)
        logits = get_model_output(outputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(logits, 1)
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

        with torch.no_grad():
            outputs = model(inputs)
            logits = get_model_output(outputs)
            loss = criterion(logits, labels)

        _, preds = torch.max(logits, 1)
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

        with torch.no_grad():
            outputs = model(inputs)
            logits = get_model_output(outputs)

        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1


def train_model(args, device):
    model_name = 'google_vit'
    source = args.source

    args.data_root = os.path.abspath(args.data_root)
    args.output_dir = os.path.abspath(args.output_dir)

    model_output_dir = os.path.join(args.output_dir, model_name, source)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"Training {model_name.upper()} on {source}")
    print(f"Output directory: {model_output_dir}")

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

    def load_and_filter_split(split):
        meta_path = os.path.join(args.data_root, split, 'metadata.csv')
        if not os.path.exists(meta_path):
            return pd.DataFrame()
        df = pd.read_csv(meta_path)
        return df[df['source'] == source]

    train_meta = load_and_filter_split('train')
    val_meta = load_and_filter_split('val')
    test_meta = load_and_filter_split('test')

    if train_meta.empty:
        raise ValueError(f"No training data found for source: {source}")

    all_classes = set(train_meta['unified_class'].unique())
    all_classes.update(val_meta['unified_class'].unique()) if not val_meta.empty else None
    all_classes.update(test_meta['unified_class'].unique()) if not test_meta.empty else None

    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")

    def create_dataset(split, meta_df, transform):
        if meta_df.empty:
            return None

        temp_meta_path = os.path.join(model_output_dir, f'temp_{split}_metadata.csv')
        meta_df.to_csv(temp_meta_path, index=False)

        dataset = TrafficSignDataset(
            root_dir=args.data_root,
            metadata_file=temp_meta_path,
            transform=transform,
            class_to_idx=class_to_idx
        )

        try:
            os.remove(temp_meta_path)
        except:
            pass

        return dataset

    train_dataset = create_dataset('train', train_meta, train_transform)
    val_dataset = create_dataset('val', val_meta, test_transform) if not val_meta.empty else None
    test_dataset = create_dataset('test', test_meta, test_transform) if not test_meta.empty else None

    if train_dataset is None:
        raise ValueError("No training dataset created!")

    print(
        f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}, Test={len(test_dataset) if test_dataset else 0}")

    print("Creating ViT model...")
    model = create_vit_model(num_classes).to(device)

    print("Model type:", type(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True) if test_dataset else None

    # Save class mapping
    class_mapping_path = os.path.join(model_output_dir, 'class_mappings.txt')
    with open(class_mapping_path, 'w') as f:
        for cls_name, idx in class_to_idx.items():
            f.write(f"{idx}: {cls_name}\n")

    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    start_time = time.time()
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader:
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if val_loader:
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        if val_loader:
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(model_output_dir, f"{model_name}_best_model_finetuned.pth")

            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved to: {best_model_path}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time // 60:.0f}m {training_time % 60:.0f}s")

    final_model_path = os.path.join(model_output_dir, f"{model_name}_final_model_finetuned.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")

    if val_loader and test_loader:
        print("\nEvaluating best model on test set...")
        best_model_path = os.path.join(model_output_dir, f"{model_name}_best_model_finetuned.pth")

        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_acc, test_f1 = test(model, test_loader, device)
            print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        else:
            test_acc, test_f1 = 0.0, 0.0
            print("Warning: Best model not found for testing")
    else:
        test_acc, test_f1 = 0.0, 0.0

    results_path = os.path.join(model_output_dir, 'training_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        results = {
            'model': model_name,
            'source': source,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'num_classes': num_classes,
            'training_time': training_time,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'best_val_acc': best_val_acc if val_loader else 0.0
        }
        for key, value in results.items():
            writer.writerow([key, value])

    print(f"✓ Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ViT Model')
    parser.add_argument('--data_root', required=True, help='Root directory of dataset')
    parser.add_argument('--output_dir', default='./base_models', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--source', default='mapillary', help='Source dataset')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_model(args, device)


if __name__ == '__main__':
    main()