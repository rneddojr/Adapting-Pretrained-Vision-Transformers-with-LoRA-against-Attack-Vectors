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
from Utils import create_model, get_normalization, TrafficSignDataset, get_filtered_metadata, save_images, create_adv_metadata


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.model = model

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)

def create_patch_attack(model, device, num_classes, args):
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
        classifier,
        rotation_max=22.5,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        patch_shape=(3, args.patch_size, args.patch_size))
    return attack

def main():
    parser = argparse.ArgumentParser(description='Generate Adversarial Patch Attacks')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patch_sample_size', type=int, default=100)
    parser.add_argument('--splits', nargs='+', default=['train','val','test'])
    parser.add_argument('--scale_min', type=float, default=0.1)
    parser.add_argument('--scale_max', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=5.0)
    parser.add_argument('--max_iter', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.dirname(args.model_path)
    class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    with open(class_mapping_path, 'r') as f:
        num_classes = len(f.readlines())

    base_model = create_model(args.model, num_classes).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))

    mean, std = get_normalization(args.model)
    model = NormalizedModel(base_model, mean, std).to(device).eval()

    for split in args.splits:
        print(f"\nProcessing {split} split...")

        output_dir = os.path.join(args.output_dir, args.model, args.source, split, 'patch', 'images')
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
            transform=transform
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

        attack = create_patch_attack(model, device, num_classes, args)
        attack.generate(x=x_train, y=y_train)

        full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_filenames = []
        for images, labels, filenames in tqdm(full_loader, desc="Applying Patch"):
            images_np = images.numpy()
            patched_np = attack.apply_patch(images_np, scale=0.4)
            patched_tensor = torch.from_numpy(patched_np).to(device)
            all_filenames.extend(filenames)
            save_images(patched_tensor, filenames, output_dir)

        patch_meta = create_adv_metadata(filtered_meta, all_filenames, output_dir)
        patch_meta.to_csv(os.path.join(output_dir, '../metadata.csv'), index=False)
        print(f"Patch attack results saved to: {os.path.dirname(output_dir)}")

if __name__ == '__main__':
    main()