import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Utils import create_model, get_dataloader, save_images, normalize_batch, get_normalization, calculate_sign_mask


def batched_fgsm_attack(model, images, labels, epsilon, mean, std):
    sign_mask = calculate_sign_mask(images)

    perturbed = images.clone().detach().requires_grad_(True)

    normalized = normalize_batch(perturbed, mean, std)
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

        normalized = normalize_batch(perturbed, mean, std)
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
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--pgd_alpha', type=float, default=0.01)
    parser.add_argument('--pgd_iters', type=int, default=40)
    parser.add_argument('--datasets', nargs='+', default=['test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader, num_classes = get_dataloader(args.data_root, 'test', args.model)
    model = create_model(args.model, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    mean, std = get_normalization(args.model)

    for dataset_name in args.datasets:
        dataloader, _ = get_dataloader(args.data_root, dataset_name, args.model, args.batch_size)
        print(f"Attacking {dataset_name} with {args.model}")

        for images, labels, filenames in tqdm(dataloader, desc=f"{dataset_name}"):
            images, labels = images.to(device), labels.to(device)

            fgsm_images = batched_fgsm_attack(model, images, labels, args.epsilon, mean, std)
            save_images(fgsm_images, filenames, 'fgsm', dataset_name, args.output_dir, mean, std)

            pgd_images = batched_pgd_attack(
                model, images, labels, args.epsilon, args.pgd_alpha, args.pgd_iters, mean, std
            )
            save_images(pgd_images, filenames, 'pgd', dataset_name, args.output_dir, mean, std)


if __name__ == '__main__':
    main()