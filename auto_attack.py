import argparse
import torch
from tqdm import tqdm
from Utils import create_model, get_dataloader, save_images, get_normalization


def main():
    parser = argparse.ArgumentParser(description='Generate AutoAttack')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--datasets', nargs='+', default=['test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from autoattack import AutoAttack
    except ImportError:
        raise ImportError("AutoAttack not installed. Run: pip install git+https://github.com/fra31/auto-attack")

    dataloader, num_classes = get_dataloader(args.data_root, 'test', args.model)
    model = create_model(args.model, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, device=device)
    mean, std = get_normalization(args.model)

    for dataset_name in args.datasets:
        dataloader, _ = get_dataloader(args.data_root, dataset_name, args.model, args.batch_size)

        for images, labels, filenames in tqdm(dataloader, desc=f"AutoAttack {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            adv_images = adversary.run_standard(images, labels)
            save_images(adv_images, filenames, 'auto', dataset_name, args.output_dir, mean, std)


if __name__ == '__main__':
    main()