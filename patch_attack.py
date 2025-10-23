import argparse
import torch
import numpy as np
from tqdm import tqdm
from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier
import random
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import torch.nn as nn
from Utils import create_vit_model, get_normalization, TrafficSignDataset, get_filtered_metadata, save_images, \
    create_adv_metadata


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.model = model

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)


def get_model_output(outputs):
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    else:
        return outputs


class LogitsModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return get_model_output(outputs)


def create_patch_attack(model, device, num_classes, args, patch_type):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        device_type=str(device))

    attack = AdversarialPatchPyTorch(
        estimator=classifier,
        rotation_max=args.rotation_max,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        distortion_scale_max=args.distortion_scale_max,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        patch_shape=(3, args.patch_size, args.patch_size),
        patch_location=args.patch_location,
        patch_type=patch_type,
        optimizer=args.optimizer,
        targeted=args.targeted,
        verbose=args.verbose
    )
    return attack


def main():
    parser = argparse.ArgumentParser(description='Generate Adversarial Patch Attacks')
    # Basic arguments
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=24)
    parser.add_argument('--patch_sample_size', type=int, default=500)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])

    # Patch generation parameters (for create_patch_attack)
    parser.add_argument('--scale_min', type=float, default=0.05)
    parser.add_argument('--scale_max', type=float, default=1.0)
    parser.add_argument('--rotation_max', type=float, default=22.5)
    parser.add_argument('--distortion_scale_max', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=5.0)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--patch_type', nargs='+', default=['circle', 'square'], choices=['circle', 'square'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'pgd'])
    parser.add_argument('--targeted', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)

    # Patch application parameters
    parser.add_argument('--scale_min_apply', type=float, default=0.1)
    parser.add_argument('--scale_max_apply', type=float, default=0.5)

    # Advanced patch location control (optional)
    parser.add_argument('--patch_location_x', type=int, default=None)
    parser.add_argument('--patch_location_y', type=int, default=None)

    args = parser.parse_args()

    # Handle patch location
    if args.patch_location_x is not None and args.patch_location_y is not None:
        args.patch_location = (args.patch_location_x, args.patch_location_y)
    else:
        args.patch_location = None

    args.data_root = os.path.abspath(args.data_root)
    args.model_path = os.path.abspath(args.model_path)
    args.output_dir = os.path.abspath(args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.dirname(args.model_path)
    class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    with open(class_mapping_path, 'r') as f:
        lines = f.readlines()
        num_classes = len(lines)
        class_to_idx = {}
        for line in lines:
            idx, cls_name = line.strip().split(': ')
            class_to_idx[cls_name] = int(idx)

    base_model = create_vit_model(num_classes).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))

    mean, std = get_normalization(args.model)
    model = NormalizedModel(base_model, mean, std).to(device).eval()

    wrapped_model = LogitsModel(model)

    # Loop through each patch type
    for patch_type in args.patch_type:
        print(f"\n{'=' * 50}")
        print(f"Generating patches for shape: {patch_type}")
        print(f"{'=' * 50}")

        for split in args.splits:
            print(f"\nProcessing {split} split for {patch_type} patches...")

            # Create folder name with patch shape
            patch_folder_name = f"patch_{patch_type}"
            base_output = os.path.join(args.output_dir, args.model, args.source, split, patch_folder_name)
            output_dir = os.path.join(base_output, 'images')
            os.makedirs(output_dir, exist_ok=True)

            clean_meta_path = os.path.join(args.data_root, split, 'metadata.csv')
            filtered_meta = get_filtered_metadata(clean_meta_path, [args.source])

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

            dataset = TrafficSignDataset(
                root_dir=args.data_root,
                metadata_file=filtered_meta,
                transform=transform,
                class_to_idx=class_to_idx
            )

            indices = list(range(len(dataset)))
            random.shuffle(indices)
            subset_indices = indices[:args.patch_sample_size]
            subset = torch.utils.data.Subset(dataset, subset_indices)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

            images_list, labels_list = [], []
            for img, lbl, _ in loader:
                images_list.append(img.numpy())
                labels_list.append(lbl.numpy())

            x_train = np.concatenate(images_list)
            y_train = np.concatenate(labels_list)

            # Create attack with specific patch type
            attack = create_patch_attack(wrapped_model, device, num_classes, args, patch_type)
            attack.generate(x=x_train, y=y_train)

            full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            all_filenames = []
            for images, labels, filenames in tqdm(full_loader, desc=f"Applying {patch_type} Patch"):
                images_np = images.numpy()

                # Apply patch with random scale for each batch
                scale = random.uniform(args.scale_min_apply, args.scale_max_apply)
                patched_np = attack.apply_patch(images_np, scale=scale)

                patched_tensor = torch.from_numpy(patched_np).to(device)
                all_filenames.extend(filenames)
                save_images(patched_tensor, filenames, output_dir)

            patch_meta = create_adv_metadata(clean_meta_path, all_filenames, output_dir)

            patch_meta['image_path'] = patch_meta['image_path'].apply(
                lambda x: os.path.abspath(x) if not os.path.isabs(x) else x
            )

            meta_path = os.path.join(base_output, 'metadata.csv')
            patch_meta.to_csv(meta_path, index=False)
            print(f"{patch_type.capitalize()} patch attack results saved to: {base_output}")

            if os.path.exists(filtered_meta):
                os.remove(filtered_meta)


if __name__ == '__main__':
    main()