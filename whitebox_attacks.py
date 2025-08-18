import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import os
from Utils import create_model, get_normalization, calculate_sign_mask, \
    TrafficSignDataset, get_filtered_metadata, save_images, create_adv_metadata


def batched_fgsm_attack(model, images, labels, epsilon, mean, std):
    sign_mask = calculate_sign_mask(images)
    perturbed = images.clone().detach().requires_grad_(True)

    normalized = (perturbed - mean) / std
    outputs = model(normalized)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    data_grad = perturbed.grad.data
    with torch.no_grad():
        perturbation = epsilon * data_grad.sign() * sign_mask
        perturbed = perturbed + perturbation
        perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


def batched_pgd_attack(model, images, labels, epsilon, alpha, num_steps, mean, std):
    sign_mask = calculate_sign_mask(images)
    orig = images.clone().detach()
    perturbed = images.clone().detach()

    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        normalized = (perturbed - mean) / std
        outputs = model(normalized)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        data_grad = perturbed.grad.data
        with torch.no_grad():
            perturbation = alpha * data_grad.sign() * sign_mask
            perturbed = perturbed + perturbation
            delta = torch.clamp(perturbed - orig, -epsilon, epsilon)
            perturbed = torch.clamp(orig + delta, 0, 1).detach()

    return perturbed


def main():
    parser = argparse.ArgumentParser(description='Generate FGSM and PGD Attacks')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True, help='Model architecture (e.g., swin)')
    parser.add_argument('--source', required=True, help='Source dataset (e.g., gtsrb)')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.10)
    parser.add_argument('--pgd_alpha', type=float, default=0.01)
    parser.add_argument('--pgd_iters', type=int, default=10)
    parser.add_argument('--splits', nargs='+', default=['train','val','test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    for split in args.splits:
        print(f"\nProcessing {split} split...")

        base_output = os.path.join(args.output_dir,args.model,args.source,split)
        fgsm_dir = os.path.join(base_output, 'fgsm', 'images')
        pgd_dir = os.path.join(base_output, 'pgd', 'images')
        os.makedirs(fgsm_dir, exist_ok=True)
        os.makedirs(pgd_dir, exist_ok=True)

        clean_meta_path = os.path.join(args.data_root, split, 'metadata.csv')
        filtered_meta = get_filtered_metadata(clean_meta_path, [args.source])

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
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
        for images, labels, filenames in tqdm(loader, desc=split):
            images, labels = images.to(device), labels.to(device)
            all_filenames.extend(filenames)

            fgsm_images = batched_fgsm_attack(model, images, labels, args.epsilon, mean_tensor, std_tensor)
            pgd_images = batched_pgd_attack(model, images, labels, args.epsilon, args.pgd_alpha,
                args.pgd_iters, mean_tensor, std_tensor)

            save_images(fgsm_images, filenames, fgsm_dir)
            save_images(pgd_images, filenames, pgd_dir)

        fgsm_meta = create_adv_metadata(filtered_meta, all_filenames, fgsm_dir)
        pgd_meta = create_adv_metadata(filtered_meta, all_filenames, pgd_dir)

        fgsm_meta.to_csv(os.path.join(base_output, 'fgsm', 'metadata.csv'), index=False)
        pgd_meta.to_csv(os.path.join(base_output, 'pgd', 'metadata.csv'), index=False)

        print(f"FGSM results saved to: {os.path.join(base_output, 'fgsm')}")
        print(f"PGD results saved to: {os.path.join(base_output, 'pgd')}")


if __name__ == '__main__':
    main()