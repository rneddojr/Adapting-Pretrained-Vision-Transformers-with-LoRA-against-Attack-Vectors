import argparse
from tqdm import tqdm
from art.attacks.evasion import RobustPhysicalPatch
from art.estimators.classification import PyTorchClassifier
import torch
import numpy as np
from Utils import (create_model, get_dataloader, save_images, get_normalization,
                   TrafficSignDataset, calculate_sign_mask)
import random
from torch.utils.data import Subset, DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch.nn as nn


class SignConstrainedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std
        return self.model(x)


def create_rp2_attack(model, device, num_classes, patch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        device_type=str(device)
    )

    attack = RobustPhysicalPatch(
        classifier,
        patch_location=(0.5, 0.5),
        patch_type="circle",
        patch_scale=(patch_size / 224, patch_size / 224),
        rotation_weights=[1.0, 0.0, 0.0],
        brightness_range=(0.8, 1.2),
        scale_range=(0.4, 1.0),
        learning_rate=0.1,
        max_iter=500,
        batch_size=16,
        verbose=False,
        targeted=False
    )

    return attack


def apply_sign_constrained_patch(images, patch_attack, mean, std, device):
    sign_mask = calculate_sign_mask(images).to(device)

    images_np = images.cpu().numpy()
    patched_np = patch_attack.apply_patch(images_np, scale=0.4)
    patched_tensor = torch.from_numpy(patched_np).to(device)

    return images * (1 - sign_mask) + patched_tensor * sign_mask


def save_mask_debug(images, masks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(5, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i][0].cpu().numpy()

        # Create overlay (green mask)
        overlay = img.copy()
        overlay[mask > 0.5] = [0, 1, 0]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title('Original')
        ax[1].imshow(overlay)
        ax[1].set_title('Sign Mask')
        plt.savefig(os.path.join(output_dir, f'mask_{i}.png'), bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate RP2 Physical Attacks')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--sample_per_class', type=int, default=50)
    parser.add_argument('--datasets', nargs='+', default=['test'])
    parser.add_argument('--debug', action='store_true', help='Enable mask debugging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, num_classes = get_dataloader(args.data_root, 'test', args.model)
    base_model = create_model(args.model, num_classes).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))

    mean, std = get_normalization(args.model)
    model = SignConstrainedModel(base_model, mean, std).to(device).eval()

    for dataset_name in args.datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_path = os.path.join(args.data_root, dataset_name)

        full_dataset = TrafficSignDataset(
            os.path.join(dataset_path, 'images'),
            os.path.join(dataset_path, 'metadata.csv'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        )

        class_patches = {}
        for class_idx, class_name in full_dataset.idx_to_class.items():
            class_indices = [i for i, (_, label, _) in enumerate(full_dataset) if label == class_idx]
            if not class_indices:
                print(f"Skipping class {class_name} - no samples")
                continue

            print(f"Generating RP2 patch for {class_name}...")
            random.shuffle(class_indices)
            sample_indices = class_indices[:args.sample_per_class]
            subset = Subset(full_dataset, sample_indices)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

            images_list, labels_list = [], []
            for img, lbl, _ in loader:
                images_list.append(img)
                labels_list.append(lbl)

            x_train = torch.cat(images_list).numpy()
            y_train = torch.cat(labels_list).numpy()

            attack = create_rp2_attack(model, device, num_classes, args.patch_size)
            attack.generate(x_train, y_train)
            class_patches[class_idx] = attack

            # Save patch visualization
            patch_img = (attack.patch * 255).astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(patch_img).save(
                os.path.join(args.output_dir, f"rp2_patch_{class_name.replace(' ', '_')}.png")
            )

        # Apply patches to full dataset
        dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Applying RP2 perturbations to {len(full_dataset)} images...")

        for images, labels, filenames in tqdm(dataloader, desc="Applying patches"):
            images = images.to(device)

            # Debug mask visualization
            if args.debug:
                sign_mask = calculate_sign_mask(images)
                debug_dir = os.path.join(args.output_dir, 'mask_debug', dataset_name)
                save_mask_debug(images, sign_mask, debug_dir)

            patched_images = []
            for i in range(len(images)):
                if labels[i] in class_patches:
                    img_tensor = images[i].unsqueeze(0)
                    patched = apply_sign_constrained_patch(
                        img_tensor,
                        class_patches[labels[i]],
                        mean, std, device
                    )
                    patched_images.append(patched)
                else:
                    patched_images.append(images[i].unsqueeze(0))

            patched_batch = torch.cat(patched_images)
            save_images(patched_batch, filenames, 'rp2', dataset_name, args.output_dir, mean, std)

    print("\nRP2 attack generation complete!")


if __name__ == '__main__':
    main()