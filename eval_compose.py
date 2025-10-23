import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from peft import PeftModel
from transformers import ViTForImageClassification
import argparse
from tqdm import tqdm
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from Utils import TrafficSignDataset, get_normalization
import itertools


def test_model(model, test_loader, device, description=""):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc=description):
            images, labels = images.to(device), labels.to(device)

            if hasattr(model, 'base_model'):
                outputs = model.base_model(pixel_values=images)
            else:
                outputs = model(pixel_values=images)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
                cls_token = last_hidden_state[:, 0, :]

                if hasattr(model, 'classifier'):
                    logits = model.classifier(cls_token)
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'classifier'):
                    logits = model.base_model.classifier(cls_token)
                else:
                    for name, module in model.named_modules():
                        if 'classifier' in name or 'head' in name:
                            logits = module(cls_token)
                            break
                    else:
                        raise ValueError("Could not find classifier in model")
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return acc, f1


def load_base_model(model_path, num_classes, device):
    try:
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading model with transformers: {e}")
        print("Trying alternative loading method...")

        from transformers import ViTConfig, ViTModel
        import torch.nn as nn

        class CustomViTForImageClassification(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
                self.classifier = nn.Linear(768, num_classes)

            def forward(self, pixel_values):
                outputs = self.vit(pixel_values=pixel_values)
                cls_token = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_token)
                return logits

        model = CustomViTForImageClassification(num_classes).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return model


def load_lora_model(base_model, lora_adapter_path, device):
    return PeftModel.from_pretrained(base_model, lora_adapter_path)


def merge_lora_adapters(base_model, lora_adapter_paths, device):
    current_model = base_model

    for i, adapter_path in enumerate(lora_adapter_paths):
        print(f"Loading adapter {i + 1}/{len(lora_adapter_paths)} from {adapter_path}")

        current_model = PeftModel.from_pretrained(current_model, adapter_path)

        current_model = current_model.merge_and_unload()

        current_model = current_model.to(device)

    return current_model


def get_class_mapping(model_dir):
    class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')
    class_to_idx = {}

    with open(class_mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                idx, cls_name = parts
                class_to_idx[cls_name] = int(idx)

    return class_to_idx, len(class_to_idx)


def create_test_dataloaders(args, class_to_idx, device):
    mean, std = get_normalization('google_vit')

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataloaders = {}

    clean_test_meta_path = os.path.join(args.data_root, 'test', 'metadata.csv')
    clean_test_meta = pd.read_csv(clean_test_meta_path)
    clean_test_meta = clean_test_meta[clean_test_meta['source'] == 'mapillary']

    temp_clean_meta_path = 'temp_clean_test_metadata.csv'
    clean_test_meta.to_csv(temp_clean_meta_path, index=False)

    clean_dataset = TrafficSignDataset(
        root_dir=args.data_root,
        metadata_file=temp_clean_meta_path,
        transform=transform,
        class_to_idx=class_to_idx
    )

    dataloaders['clean'] = DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    # Get all available attacks from the adversarial root directory
    test_adv_base_dir = os.path.join(args.adv_root, 'google_vit', 'mapillary', 'test')
    if os.path.exists(test_adv_base_dir):
        for attack_name in os.listdir(test_adv_base_dir):
            test_adv_dir = os.path.join(test_adv_base_dir, attack_name)
            test_adv_meta_path = os.path.join(test_adv_dir, 'metadata.csv')

            if os.path.exists(test_adv_meta_path):
                test_adv_dataset = TrafficSignDataset(
                    root_dir=test_adv_dir,
                    metadata_file=test_adv_meta_path,
                    transform=transform,
                    class_to_idx=class_to_idx
                )

                dataloaders[attack_name] = DataLoader(
                    test_adv_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=min(4, os.cpu_count()),
                    pin_memory=True
                )
                print(f"Loaded test data for attack: {attack_name}")
            else:
                print(f"Warning: Adversarial metadata not found for {attack_name} at {test_adv_meta_path}")

    if os.path.exists(temp_clean_meta_path):
        os.remove(temp_clean_meta_path)

    return dataloaders


def find_lora_adapters(lora_root, attacks, rank):
    lora_adapters = {}

    for attack in attacks:
        lora_path = os.path.join(lora_root, 'google_vit', 'mapillary', attack, f'rank{rank}_best_adapter')
        if os.path.exists(lora_path):
            lora_adapters[attack] = lora_path
            print(f"Found LoRA adapter for {attack} (rank {rank}): {lora_path}")
        else:
            print(f"Warning: LoRA adapter not found for {attack} (rank {rank}) at {lora_path}")

    return lora_adapters


def test_base_model(args, model_path, num_classes, dataloaders, device, results):
    print("\n" + "=" * 60)
    print("TESTING BASE MODEL (NO LoRA)")
    print("=" * 60)

    try:
        base_model = load_base_model(model_path, num_classes, device)
        base_results = {}

        for dataset_name in dataloaders.keys():
            acc, f1 = test_model(base_model, dataloaders[dataset_name], device,
                                 f"Base model on {dataset_name}")
            base_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
            print(f"Base model on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

        results['base_model'] = base_results

        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error testing base model: {e}")
        import traceback
        traceback.print_exc()
        results['base_model'] = {'error': str(e)}


def test_individual_loras(args, model_path, num_classes, dataloaders, lora_adapters, device, results):
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL LoRA ADAPTERS")
    print("=" * 60)

    attacks = list(lora_adapters.keys())

    for attack, lora_path in lora_adapters.items():
        print(f"\n--- Testing {attack.upper()} LoRA adapter ---")

        try:
            base_model = load_base_model(model_path, num_classes, device)
            lora_model = load_lora_model(base_model, lora_path, device)

            attack_results = {}
            for dataset_name in dataloaders.keys():
                acc, f1 = test_model(lora_model, dataloaders[dataset_name], device,
                                     f"{attack} LoRA on {dataset_name}")
                attack_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
                print(f"{attack} LoRA on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

            results[f'{attack}_lora'] = attack_results

        except Exception as e:
            print(f"Error testing {attack} LoRA: {e}")
            results[f'{attack}_lora'] = {'error': str(e)}

        # Clean up to free memory
        if 'base_model' in locals():
            del base_model
        if 'lora_model' in locals():
            del lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_lora_combinations(args, model_path, num_classes, dataloaders, lora_adapters, device, results):
    attacks = list(lora_adapters.keys())

    if len(attacks) < 2:
        print("Not enough LoRA adapters for combination testing")
        return

    print("\n" + "=" * 60)
    print("TESTING LoRA COMBINATIONS")
    print("=" * 60)

    adapter_combinations_2 = list(itertools.combinations(attacks, 2))

    for combo in adapter_combinations_2:
        combo_name = '+'.join(combo)
        print(f"\n--- Testing {combo_name} LoRA combination (2 adapters) ---")

        if all(attack in lora_adapters for attack in combo):
            try:
                base_model = load_base_model(model_path, num_classes, device)

                adapter_paths = [lora_adapters[attack] for attack in combo]
                merged_model = merge_lora_adapters(base_model, adapter_paths, device)

                combo_results = {}
                for dataset_name in dataloaders.keys():
                    acc, f1 = test_model(merged_model, dataloaders[dataset_name], device,
                                         f"{combo_name} on {dataset_name}")
                    combo_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
                    print(f"{combo_name} on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

                results[f'{combo_name}_combo_2'] = combo_results

            except Exception as e:
                print(f"Error testing {combo_name} combination: {e}")
                import traceback
                traceback.print_exc()
                results[f'{combo_name}_combo_2'] = {'error': str(e)}

            # Clean up
            if 'base_model' in locals():
                del base_model
            if 'merged_model' in locals():
                del merged_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"Skipping {combo_name} - not all adapters available")

    if len(attacks) >= 3:
        print("\n" + "=" * 60)
        print("TESTING 3-LoRA COMBINATIONS")
        print("=" * 60)

        adapter_combinations_3 = list(itertools.combinations(attacks, 3))

        for combo in adapter_combinations_3:
            combo_name = '+'.join(combo)
            print(f"\n--- Testing {combo_name} LoRA combination (3 adapters) ---")

            if all(attack in lora_adapters for attack in combo):
                try:
                    base_model = load_base_model(model_path, num_classes, device)

                    adapter_paths = [lora_adapters[attack] for attack in combo]
                    merged_model = merge_lora_adapters(base_model, adapter_paths, device)

                    combo_results = {}
                    for dataset_name in dataloaders.keys():
                        acc, f1 = test_model(merged_model, dataloaders[dataset_name], device,
                                             f"{combo_name} on {dataset_name}")
                        combo_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
                        print(f"{combo_name} on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

                    results[f'{combo_name}_combo_3'] = combo_results

                except Exception as e:
                    print(f"Error testing {combo_name} combination: {e}")
                    import traceback
                    traceback.print_exc()
                    results[f'{combo_name}_combo_3'] = {'error': str(e)}

                # Clean up
                if 'base_model' in locals():
                    del base_model
                if 'merged_model' in locals():
                    del merged_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"Skipping {combo_name} - not all adapters available")

    if len(attacks) >= 4:
        print("\n" + "=" * 60)
        print("TESTING ALL LoRA COMBINATION")
        print("=" * 60)

        combo_name = '+'.join(attacks)
        print(f"\n--- Testing {combo_name} LoRA combination (all adapters) ---")

        try:
            base_model = load_base_model(model_path, num_classes, device)

            adapter_paths = [lora_adapters[attack] for attack in attacks]
            merged_model = merge_lora_adapters(base_model, adapter_paths, device)

            combo_results = {}
            for dataset_name in dataloaders.keys():
                acc, f1 = test_model(merged_model, dataloaders[dataset_name], device,
                                     f"{combo_name} on {dataset_name}")
                combo_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
                print(f"{combo_name} on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

            results[f'{combo_name}_combo_all'] = combo_results

        except Exception as e:
            print(f"Error testing {combo_name} combination: {e}")
            import traceback
            traceback.print_exc()
            results[f'{combo_name}_combo_all'] = {'error': str(e)}

        # Clean up
        if 'base_model' in locals():
            del base_model
        if 'merged_model' in locals():
            del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif len(attacks) >= 3:
        combo_name = '+'.join(attacks)
        print(f"\n--- Testing {combo_name} LoRA combination (all 3 adapters) ---")

        try:
            base_model = load_base_model(model_path, num_classes, device)

            adapter_paths = [lora_adapters[attack] for attack in attacks]
            merged_model = merge_lora_adapters(base_model, adapter_paths, device)

            combo_results = {}
            for dataset_name in dataloaders.keys():
                acc, f1 = test_model(merged_model, dataloaders[dataset_name], device,
                                     f"{combo_name} on {dataset_name}")
                combo_results[dataset_name] = {'accuracy': acc, 'f1_score': f1}
                print(f"{combo_name} on {dataset_name}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

            results[f'{combo_name}_combo_3'] = combo_results

        except Exception as e:
            print(f"Error testing {combo_name} combination: {e}")
            import traceback
            traceback.print_exc()
            results[f'{combo_name}_combo_3'] = {'error': str(e)}

        if 'base_model' in locals():
            del base_model
        if 'merged_model' in locals():
            del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Test base model and LoRA adapters')
    parser.add_argument('--model_path', required=True, help='Path to base fine-tuned model')
    parser.add_argument('--lora_root', required=True, help='Root directory containing LoRA adapters')
    parser.add_argument('--adv_root', required=True, help='Root directory for adversarial examples')
    parser.add_argument('--data_root', required=True, help='Root directory for clean examples')
    parser.add_argument('--attacks', nargs='+', required=True, help='List of attacks to evaluate')
    parser.add_argument('--rank', type=int, required=True, help='Rank value to evaluate (e.g., 16)')
    parser.add_argument('--output_file', default='test_results.json', help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_mode', choices=['all', 'base_only', 'individual_only', 'combinations_only'],
                        default='all', help='What to test: all, base_only, individual_only, or combinations_only')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Test mode: {args.test_mode}")
    print(f"Evaluating rank: {args.rank}")
    print(f"Attacks to evaluate: {args.attacks}")

    model_dir = os.path.dirname(args.model_path)
    class_to_idx, num_classes = get_class_mapping(model_dir)
    print(f"Number of classes: {num_classes}")

    print("Creating test dataloaders...")
    dataloaders = create_test_dataloaders(args, class_to_idx, device)
    print(f"Available test datasets: {list(dataloaders.keys())}")

    lora_adapters = find_lora_adapters(args.lora_root, args.attacks, args.rank)

    if not lora_adapters:
        print("No LoRA adapters found for the specified attacks and rank!")
        return

    results = {
        'rank': args.rank,
        'attacks_evaluated': args.attacks,
        'test_datasets': list(dataloaders.keys())
    }

    if args.test_mode in ['all', 'base_only']:
        test_base_model(args, args.model_path, num_classes, dataloaders, device, results)

    if args.test_mode in ['all', 'individual_only']:
        test_individual_loras(args, args.model_path, num_classes, dataloaders, lora_adapters, device, results)

    if args.test_mode in ['all', 'combinations_only']:
        test_lora_combinations(args, args.model_path, num_classes, dataloaders, lora_adapters, device, results)

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {args.output_file}")

    print("\n" + "=" * 80)
    print(f"SUMMARY RESULTS (Rank {args.rank})")
    print("=" * 80)

    test_datasets = list(dataloaders.keys())

    model_keys = [key for key in results.keys() if key not in ['rank', 'attacks_evaluated', 'test_datasets']]

    print("\n" + "Model".ljust(35) + "".join([f"{dataset:>12}" for dataset in test_datasets]))
    print("-" * (35 + 12 * len(test_datasets)))

    for model in model_keys:
        if model in results and isinstance(results[model], dict) and not results[model].get('error'):
            line = f"{model:<35}"
            for dataset in test_datasets:
                if dataset in results[model] and 'accuracy' in results[model][dataset]:
                    acc = results[model][dataset]['accuracy']
                    line += f"{acc:>12.4f}"
                else:
                    line += f"{'N/A':>12}"
            print(line)
        else:
            print(f"{model:<35} {'ERROR':>12}")


if __name__ == '__main__':
    main()