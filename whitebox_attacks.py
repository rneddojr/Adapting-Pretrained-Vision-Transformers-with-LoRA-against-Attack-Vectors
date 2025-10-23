import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import os
from Utils import TrafficSignDataset, create_vit_model, get_normalization, get_filtered_metadata, save_images, \
    create_adv_metadata
from torchattacks import FGSM, PGD


def get_model_output(outputs):
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, dict) and 'logits' in outputs:
        return outputs['logits']
    else:
        return outputs


def batched_fgsm_attack(model, images, labels, epsilon, mean, std):
    uniform_mask = torch.ones_like(images)
    perturbed = images.clone().detach().requires_grad_(True)

    normalized = (perturbed - mean) / std
    outputs = model(normalized)
    logits = get_model_output(outputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    data_grad = perturbed.grad.data
    with torch.no_grad():
        perturbation = epsilon * data_grad.sign() * uniform_mask
        perturbed = perturbed + perturbation
        perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


class LogitsModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return get_model_output(outputs)


def main():
    parser = argparse.ArgumentParser(description='Generate FGSM and PGD Attacks')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--models', nargs='+', required=True, help='Model architectures (e.g., google_vit)')
    parser.add_argument('--sources', nargs='+', required=True, help='Source datasets (e.g., mapillary)')
    parser.add_argument('--model_base_path', default='./Train24', help='Base path for models')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=8 / 255)
    parser.add_argument('--pgd_alpha', type=float, default=3 / 255)
    parser.add_argument('--pgd_iters', type=int, default=30)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--attacks', nargs='+', choices=['fgsm', 'pgd'], default=['fgsm', 'pgd'],
                        help='Which attacks to run (default: both)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_name in args.models:
        for source_name in args.sources:
            print(f"\nProcessing model: {model_name}, source: {source_name}")

            model_path = os.path.join(
                args.model_base_path,
                model_name,
                source_name,
                f"{model_name}_best_model_finetuned.pth"
            )

            model_dir = os.path.dirname(model_path)
            class_mapping_path = os.path.join(model_dir, 'class_mappings.txt')

            if not os.path.exists(class_mapping_path):
                print(f"Warning: Class mapping file not found: {class_mapping_path}")
                continue

            with open(class_mapping_path, 'r') as f:
                lines = f.readlines()
                num_classes = len(lines)

            model = create_vit_model(num_classes).to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except FileNotFoundError:
                print(f"Warning: Model file not found: {model_path}")
                continue

            model.eval()

            wrapped_model = LogitsModel(model)
            wrapped_model.eval()

            mean, std = get_normalization(model_name)
            mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(device)
            std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(device)

            attacks = {}
            if 'fgsm' in args.attacks:
                attacks['fgsm'] = FGSM(wrapped_model, eps=args.epsilon)
            if 'pgd' in args.attacks:
                attacks['pgd'] = PGD(wrapped_model, eps=args.epsilon, alpha=args.pgd_alpha,
                                     steps=args.pgd_iters, random_start=True)

            for split in args.splits:
                print(f"  Processing {split} split...")

                base_output = os.path.join(args.output_dir, model_name, source_name, split)

                attack_dirs = {}
                for attack_name in args.attacks:
                    attack_dir = os.path.join(base_output, attack_name, 'images')
                    os.makedirs(attack_dir, exist_ok=True)
                    attack_dirs[attack_name] = attack_dir

                clean_meta_path = os.path.join(args.data_root, split, 'metadata.csv')
                filtered_meta = get_filtered_metadata(clean_meta_path, [source_name])

                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])

                class_to_idx = {}
                with open(class_mapping_path, 'r') as f:
                    for line in f:
                        idx, cls_name = line.strip().split(': ')
                        class_to_idx[cls_name] = int(idx)

                dataset = TrafficSignDataset(
                    root_dir=args.data_root,
                    metadata_file=filtered_meta,
                    transform=transform,
                    class_to_idx=class_to_idx
                )

                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=min(4, os.cpu_count()),
                    pin_memory=True
                )

                all_filenames = []
                for images, labels, filenames in tqdm(loader, desc=f"{model_name}-{source_name}-{split}"):
                    images, labels = images.to(device), labels.to(device)
                    all_filenames.extend(filenames)

                    adv_images = {}
                    for attack_name in args.attacks:
                        if attack_name == 'fgsm':
                            adv_images[attack_name] = batched_fgsm_attack(
                                model, images, labels, args.epsilon, mean_tensor, std_tensor
                            )
                        else:
                            attack = attacks[attack_name]
                            attack.set_normalization_used(mean=mean, std=std)
                            adv_images[attack_name] = attack(images, labels)

                    for attack_name in args.attacks:
                        save_images(adv_images[attack_name], filenames, attack_dirs[attack_name])

                for attack_name in args.attacks:
                    attack_meta = create_adv_metadata(clean_meta_path, all_filenames, attack_dirs[attack_name])
                    meta_path = os.path.join(base_output, attack_name, 'metadata.csv')
                    attack_meta.to_csv(meta_path, index=False)
                    print(f"    {attack_name.upper()} results saved to: {os.path.join(base_output, attack_name)}")

                if os.path.exists(filtered_meta):
                    os.remove(filtered_meta)


if __name__ == '__main__':
    main()