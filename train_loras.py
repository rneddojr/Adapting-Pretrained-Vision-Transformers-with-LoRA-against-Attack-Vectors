import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import ViTForImageClassification
from Utils import TrafficSignDataset, get_normalization
import argparse
from tqdm import tqdm
import pandas as pd
import json
import traceback
from sklearn.metrics import accuracy_score, f1_score


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Validation', leave=False)
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        total_samples += batch_size

        with torch.no_grad():
            outputs = model.base_model(pixel_values=images)
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
            elif hasattr(outputs, 'logits'):
                outputs = outputs.logits
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data).item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1


def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model.base_model(pixel_values=images)
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
            elif hasattr(outputs, 'logits'):
                outputs = outputs.logits
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return acc, f1


def setup_peft_lora(model, rank=16, alpha=16, dropout=0.1, target_modules=None):
    if target_modules is None:
        target_modules = ["query", "key", "value", "output.dense"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def get_patch_attack_dirs(adv_root, model_name, source, attack):
    attack_dirs = {}

    if attack.startswith('patch_'):
        base_pattern = os.path.join(adv_root, model_name, source)

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(base_pattern, split)
            if os.path.exists(split_dir):
                for item in os.listdir(split_dir):
                    if item == attack:
                        attack_dir = os.path.join(split_dir, item)
                        meta_path = os.path.join(attack_dir, 'metadata.csv')
                        if os.path.exists(meta_path):
                            attack_dirs[split] = (attack_dir, meta_path)
                            break

    return attack_dirs


def train_lora_for_model_and_attack(model_name, source, attack, model_path, args, device):
    try:
        if model_name != 'google_vit' or source != 'mapillary':
            print(f"Skipping {model_name} on {source} - focusing on google_vit and mapillary")
            return {}

        output_dir = os.path.join(args.output_dir, model_name, source, attack)
        os.makedirs(output_dir, exist_ok=True)

        all_results = {}

        model_dir = os.path.dirname(model_path)
        class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

        if not os.path.exists(class_mapping_path):
            print(f"Class mapping file not found: {class_mapping_path}")
            return {}

        class_to_idx = {}
        with open(class_mapping_path, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    idx, cls_name = parts
                    class_to_idx[cls_name] = int(idx)

        num_classes = len(class_to_idx)

        print(f"Loading base model: {model_name} for {num_classes} classes")

        base_model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(device)

        print(f"Loading fine-tuned weights from: {model_path}")
        base_model.load_state_dict(torch.load(model_path, map_location=device))

        attack_dirs = {}

        if attack.startswith('patch_'):
            attack_dirs = get_patch_attack_dirs(args.adv_root, model_name, source, attack)
        else:
            for split in ['train', 'val', 'test']:
                attack_dir = os.path.join(args.adv_root, model_name, source, split, attack)
                meta_path = os.path.join(attack_dir, 'metadata.csv')
                if os.path.exists(meta_path):
                    attack_dirs[split] = (attack_dir, meta_path)

        if not attack_dirs:
            print(f"No data found for attack: {attack}")
            return {}

        clean_test_dir = os.path.join(args.data_root, 'test')
        clean_test_meta_path = os.path.join(clean_test_dir, 'metadata.csv')

        if not os.path.exists(clean_test_meta_path):
            print(f"Clean test metadata not found: {clean_test_meta_path}")
            return {}

        clean_test_meta = pd.read_csv(clean_test_meta_path)
        clean_test_meta = clean_test_meta[clean_test_meta['source'] == source]

        temp_clean_meta_path = os.path.join(output_dir, 'temp_clean_test_metadata.csv')
        clean_test_meta.to_csv(temp_clean_meta_path, index=False)

        mean, std = get_normalization(model_name)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset = None
        val_dataset = None
        test_adv_dataset = None

        if 'train' in attack_dirs:
            root_dir, meta_path = attack_dirs['train']
            train_dataset = TrafficSignDataset(
                root_dir=root_dir,
                metadata_file=meta_path,
                transform=transform,
                class_to_idx=class_to_idx
            )
            print(f"Training samples: {len(train_dataset)}")

        if 'val' in attack_dirs:
            root_dir, meta_path = attack_dirs['val']
            val_dataset = TrafficSignDataset(
                root_dir=root_dir,
                metadata_file=meta_path,
                transform=transform,
                class_to_idx=class_to_idx
            )

        if 'test' in attack_dirs:
            root_dir, meta_path = attack_dirs['test']
            test_adv_dataset = TrafficSignDataset(
                root_dir=root_dir,
                metadata_file=meta_path,
                transform=transform,
                class_to_idx=class_to_idx
            )

        test_clean_dataset = TrafficSignDataset(
            root_dir=args.data_root,
            metadata_file=temp_clean_meta_path,
            transform=transform,
            class_to_idx=class_to_idx
        )

        if train_dataset is None:
            print("No training data available")
            return {}

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        ) if val_dataset else None

        test_adv_loader = DataLoader(
            test_adv_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        ) if test_adv_dataset else None

        test_clean_loader = DataLoader(
            test_clean_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        for rank in args.ranks:
            print(f"\n{'=' * 50}")
            print(f"Training {model_name} on {source} with {attack} attack, rank {rank}")
            print(f"{'=' * 50}")

            rank_base_model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            ).to(device)
            rank_base_model.load_state_dict(torch.load(model_path, map_location=device))

            peft_model = setup_peft_lora(rank_base_model, rank=rank)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(peft_model.parameters(), lr=args.lr)

            best_val_acc = 0.0
            rank_results = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'val_f1': []
            }

            for epoch in range(args.epochs):
                print(f'\nEpoch {epoch + 1}/{args.epochs}')
                peft_model.train()
                running_loss = 0.0
                running_corrects = 0
                total_samples = 0

                pbar = tqdm(train_loader, desc='Training', leave=False)
                for images, labels, _ in pbar:
                    images, labels = images.to(device), labels.to(device)
                    batch_size = images.size(0)
                    total_samples += batch_size

                    optimizer.zero_grad()

                    outputs = peft_model.base_model(pixel_values=images)
                    logits = outputs.logits

                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(logits, 1)
                    running_loss += loss.item() * batch_size
                    running_corrects += torch.sum(preds == labels.data).item()

                    epoch_loss = running_loss / total_samples
                    epoch_acc = running_corrects / total_samples

                    pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.4f}'})

                train_loss = running_loss / total_samples
                train_acc = running_corrects / total_samples
                rank_results['train_loss'].append(train_loss)
                rank_results['train_acc'].append(train_acc)

                if val_loader:
                    val_loss, val_acc, val_f1 = validate(peft_model, val_loader, criterion, device)
                    rank_results['val_loss'].append(val_loss)
                    rank_results['val_acc'].append(val_acc)
                    rank_results['val_f1'].append(val_f1)

                    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_lora_dir = os.path.join(output_dir, f'rank{rank}_best_adapter')
                        peft_model.save_pretrained(best_lora_dir)
                        print(f"New best LoRA adapter saved to: {best_lora_dir}")
                else:
                    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
                    if train_acc > best_val_acc:
                        best_val_acc = train_acc
                        best_lora_dir = os.path.join(output_dir, f'rank{rank}_best_adapter')
                        peft_model.save_pretrained(best_lora_dir)
                        print(f"New best LoRA adapter saved to: {best_lora_dir}")

            final_lora_dir = os.path.join(output_dir, f'rank{rank}_final_adapter')
            peft_model.save_pretrained(final_lora_dir)
            print(f'Final LoRA adapter saved to: {final_lora_dir}')

            print("\nTesting on clean test data...")
            clean_acc, clean_f1 = test_model(peft_model, test_clean_loader, device)
            print(f"Clean Test Accuracy: {clean_acc:.4f}, F1: {clean_f1:.4f}")

            if test_adv_loader:
                print("\nTesting on adversarial test data...")
                adv_acc, adv_f1 = test_model(peft_model, test_adv_loader, device)
                print(f"Adversarial Test Accuracy: {adv_acc:.4f}, F1: {adv_f1:.4f}")
            else:
                adv_acc, adv_f1 = 0.0, 0.0
                print("No adversarial test data available")

            all_results[rank] = {
                'train_loss': rank_results['train_loss'],
                'train_acc': rank_results['train_acc'],
                'val_loss': rank_results['val_loss'],
                'val_acc': rank_results['val_acc'],
                'val_f1': rank_results['val_f1'],
                'clean_test_acc': clean_acc,
                'clean_test_f1': clean_f1,
                'adv_test_acc': adv_acc,
                'adv_test_f1': adv_f1,
                'best_val_acc': best_val_acc
            }

        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nAll results saved to: {results_path}")

        if os.path.exists(temp_clean_meta_path):
            os.remove(temp_clean_meta_path)

        return all_results

    except Exception as e:
        print(f"Error in train_lora_for_model_and_attack: {e}")
        print(traceback.format_exc())
        return {}


def load_lora_and_apply(model_path, lora_adapter_path, device):
    model_dir = os.path.dirname(model_path)
    class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

    class_to_idx = {}
    with open(class_mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                idx, cls_name = parts
                class_to_idx[cls_name] = int(idx)

    num_classes = len(class_to_idx)

    base_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))

    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    return model


def train_lora_adversarial_defense():
    parser = argparse.ArgumentParser(
        description='Train LoRA for adversarial defense using PEFT')
    parser.add_argument('--models', nargs='+', default=['google_vit'],
                        help='Model architectures to train (focusing on google_vit)')
    parser.add_argument('--sources', nargs='+', default=['mapillary'],
                        help='Source dataset names (focusing on mapillary)')
    parser.add_argument('--attacks', nargs='+', default=['patch_circle','patch_square','pgd','fgsm'], help='Attack types to train on')
    parser.add_argument('--model_base_path',
                        default='./train24/{model}/{source}/{model}_best_model_finetuned.pth',
                        help='Base path to model checkpoints')
    parser.add_argument('--adv_root', required=True, help='Root directory for adversarial examples')
    parser.add_argument('--data_root', required=True, help='Root directory for clean examples')
    parser.add_argument('--output_dir', required=True, help='Base directory to save LoRA parameters')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--ranks', nargs='+', type=int, default=[8, 16, 32],
                        help='List of ranks to train LoRAs for')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    global_results = {}

    for model_name in args.models:
        for source in args.sources:
            for attack in args.attacks:
                model_path = args.model_base_path.format(model=model_name, source=source)

                print(f"\n{'#' * 80}")
                print(f"Training {model_name} on {source} with {attack} attack")
                print(f"Ranks: {args.ranks}")
                print(f"{'#' * 80}")

                try:
                    results = train_lora_for_model_and_attack(model_name, source, attack, model_path, args, device)
                    if model_name not in global_results:
                        global_results[model_name] = {}
                    if source not in global_results[model_name]:
                        global_results[model_name][source] = {}
                    global_results[model_name][source][attack] = results

                except Exception as e:
                    print(f"Error training {model_name} on {source} with {attack}: {e}")
                    print(traceback.format_exc())
                    continue

    global_results_path = os.path.join(args.output_dir, 'global_results.json')
    with open(global_results_path, 'w') as f:
        json.dump(global_results, f, indent=4)
    print(f"\nGlobal results saved to: {global_results_path}")


if __name__ == '__main__':
    train_lora_adversarial_defense()