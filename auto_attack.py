import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from Utils import create_model, get_normalization, TrafficSignDataset, get_filtered_metadata, save_images


def main():
    parser = argparse.ArgumentParser(description='Generate AutoAttack')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True, help='Model architecture (e.g., swin)')
    parser.add_argument('--source', required=True, help='Source dataset (e.g., gtsrb)')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.031)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError("AutoAttack not installed. Run: pip install git+https://github.com/fra31/auto-attack")

    model_dir = os.path.dirname(args.model_path)
    class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    with open(class_mapping_path, 'r') as f:
        num_classes = len(f.readlines())

    model = create_model(args.model, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    mean_tensor = torch.tensor(get_normalization(args.model)[0]).view(1, 3, 1, 1).to(device)
    std_tensor = torch.tensor(get_normalization(args.model)[1]).view(1, 3, 1, 1).to(device)

    class NormalizedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model((x - mean_tensor) / std_tensor)

    normalized_model = NormalizedModel(model).eval()

    for split in args.splits:
        print(f"\nProcessing {split} split...")

        output_dir = os.path.join(
            args.output_dir,
            args.model,
            args.source,
            split,
            'auto',
            'images'
        )
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

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )

        all_filenames = []
        for images, labels, filenames in tqdm(loader, desc=f"AutoAttack {split}"):
            images, labels = images.to(device), labels.to(device)
            all_filenames.extend(filenames)

            adversary = AutoAttack(
                normalized_model,
                norm='Linf',
                eps=args.epsilon,
                device=device,
                version='standard',
                seed=42,
                verbose=False
            )

            x_adv = adversary.run_standard_evaluation(images, labels, bs=args.batch_size)

            save_images(x_adv, filenames, output_dir)

        auto_meta = pd.read_csv(filtered_meta)
        auto_meta['image_path'] = auto_meta['image_path'].apply(
            lambda x: os.path.join(output_dir, os.path.basename(x)))
        auto_meta.to_csv(os.path.join(output_dir, '../metadata.csv'), index=False)
        print(f"AutoAttack results saved to: {os.path.dirname(output_dir)}")


if __name__ == '__main__':
    main()