import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import clip
import timm
import torch.nn.functional as F


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.filenames = []

        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(root_dir, row['image_path'])
                label = row['unified_class']
                filename = os.path.basename(img_path)

                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)

                self.samples.append((img_path, self.class_to_idx[label]))
                self.filenames.append(filename)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, self.filenames[idx]


def create_model(model_name, num_classes):
    if model_name == 'dinov2':
        model = timm.create_model('dinov2_vitb14', pretrained=True, num_classes=num_classes)
    elif model_name == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'clip':
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        model = clip_model.visual
        model.proj = None
        model.classifier = nn.Linear(512, num_classes)
    elif model_name == 'googlenet':
        model = timm.create_model('inception_v3', pretrained=True, num_classes=num_classes)
    elif model_name == 'resnet':
        model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    elif model_name == 'convnext':
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_normalization(model_name):
    if model_name == 'clip':
        return [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # Default ImageNet


def normalize_batch(batch, mean, std):
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    return (batch - mean_tensor) / std_tensor


def unnormalize_batch(batch, mean, std):
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    return batch * std_tensor + mean_tensor


def calculate_sign_mask(images):
    batch_size, _, height, width = images.shape
    mask = torch.zeros((batch_size, 1, height, width), device=images.device)
    h_start, h_end = int(height * 0.25), int(height * 0.75)
    w_start, w_end = int(width * 0.25), int(width * 0.75)
    mask[:, :, h_start:h_end, w_start:w_end] = 1.0
    return mask


def calculate_sign_mask(images):
    gray = images.mean(dim=1, keepdim=True)
    normalized_gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    threshold = 0.5 * (normalized_gray.min() + normalized_gray.max())
    mask = (normalized_gray > threshold).float()

    kernel = torch.ones(1, 1, 7, 7, device=images.device)

    mask = F.conv2d(mask, kernel, padding=3)
    mask = (mask > 0).float()
    mask = 1 - F.conv2d(1 - mask, kernel, padding=3)
    mask = (mask > 0).float()

    mask = 1 - F.conv2d(1 - mask, kernel, padding=3)
    mask = (mask > 0).float()
    mask = F.conv2d(mask, kernel, padding=3)
    mask = (mask > 0).float()

    if mask.sum() < 0.25 * mask.numel():
        _, _, H, W = images.shape
        fallback = torch.zeros_like(mask)
        h_start, h_end = int(H * 0.25), int(H * 0.75)
        w_start, w_end = int(W * 0.25), int(W * 0.75)
        fallback[:, :, h_start:h_end, w_start:w_end] = 1
        return fallback

    return mask.repeat(1, 3, 1, 1)


def save_images(images, filenames, attack_type, dataset_name, output_dir, mean, std):
    attack_dir = os.path.join(output_dir, dataset_name, attack_type)
    os.makedirs(attack_dir, exist_ok=True)

    images = unnormalize_batch(images, mean, std)
    images = torch.clamp(images, 0, 1)

    for i, filename in enumerate(filenames):
        img = images[i].permute(1, 2, 0).cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(attack_dir, filename))


def get_dataloader(data_root, dataset_name, model_name, batch_size=32):
    dataset_path = os.path.join(data_root, dataset_name)
    mean, std = get_normalization(model_name)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = TrafficSignDataset(
        os.path.join(dataset_path, 'images'),
        os.path.join(dataset_path, 'metadata.csv'),
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(dataset.class_to_idx)