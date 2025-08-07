import argparse
import torch
import numpy as np
from tqdm import tqdm
from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier
from Utils import create_model, get_dataloader, save_images, get_normalization, TrafficSignDataset
import random
from torch.utils.data import Subset, DataLoader
import os
from torchvision import transforms
import torch.nn as nn


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)


def create_patch_attack(model, device, num_classes, patch_size=32):
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

    attack = AdversarialPatchPyTorch(
        classifier,
        rotation_max=22.5,
        scale_min=0.1,
        scale_max=1.0,
        learning_rate=5.0,
        max_iter=500,
        batch_size=16,
        patch_shape=(3, patch_size, patch_size),
    )
    return attack


def main():
    parser = argparse.ArgumentParser(description='Generate Adversarial Patch Attacks')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--patch_sample_size', type=int, default=100)
    parser.add_argument('--datasets', nargs='+', default=['test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, num_classes = get_dataloader(args.data_root, 'test', args.model)
    base_model = create_model(args.model, num_classes).to(device)
    base_model.load_state_dict(torch.load(args.model_path, map_location=device))

    mean, std = get_normalization(args.model)
    model = NormalizedModel(base_model, mean, std).to(device).eval()

    for dataset_name in args.datasets:
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

        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        subset_indices = indices[:args.patch_sample_size]
        subset = Subset(full_dataset, subset_indices)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

        images_list, labels_list = [], []
        for img, lbl, _ in loader:
            images_list.append(img.numpy())
            labels_list.append(lbl.numpy())

        x_train = np.concatenate(images_list)
        y_train = np.concatenate(labels_list)

        attack = create_patch_attack(model, device, num_classes, args.patch_size)
        attack.generate(x=x_train, y=y_train)

        loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
        for images, labels, filenames in tqdm(loader, desc="Applying Patch"):
            images_np = images.numpy()
            patched_np = attack.apply_patch(images_np, scale=0.4)
            patched_tensor = torch.from_numpy(patched_np).to(device)
            save_images(patched_tensor, filenames, 'patch', dataset_name, args.output_dir, mean, std)


if __name__ == '__main__':
    main()