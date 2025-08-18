import os
import tempfile
import pandas as pd
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
from ultralytics import YOLO


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.filenames = []
        self.available_classes = set()

        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']

                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.root_dir, img_path)

                img_path = os.path.normpath(img_path)
                label = row['unified_class']
                filename = os.path.basename(img_path)

                self.available_classes.add(label)

                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)

                self.samples.append((img_path, self.class_to_idx[label]))
                self.filenames.append(filename)

        self.class_to_idx = {cls: idx for cls, idx in self.class_to_idx.items()
                             if cls in self.available_classes}

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.available_classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label, self.filenames[idx]


def create_model(model_name, num_classes):
    if model_name == 'dinov3':
        "temp"
    elif model_name == 'dinov1':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        model.head = nn.Linear(768, num_classes)
    elif model_name == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'google_vit':
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
        model = timm.create_model('convnext_base', pretrained=True, num_classes=num_classes)
    elif model_name == 'yolov11':
        model = YOLO("yolo11x-cls.pt", task='classify')
        pytorch_model = model.model

        if hasattr(pytorch_model, 'head'):
            in_features = pytorch_model.head.in_features
            pytorch_model.head = nn.Linear(in_features, num_classes)
        elif hasattr(pytorch_model, 'classifier'):
            in_features = pytorch_model.classifier.in_features
            pytorch_model.classifier = nn.Linear(in_features, num_classes)
        else:
            for name, module in pytorch_model.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
                    in_features = last_linear.in_features
                    break
            pytorch_model.fc = nn.Linear(in_features, num_classes)

        class YOLOWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                output = self.model(x)
                if isinstance(output, tuple):
                    for item in output:
                        if isinstance(item, torch.Tensor) and item.dim() == 2:
                            return item
                    return output[0]
                return output

        wrapped_model = YOLOWrapper(pytorch_model)
        for param in wrapped_model.parameters():
            param.requires_grad = True

        return wrapped_model

    return model


def get_normalization(model_name):
    if model_name == 'clip':
        return [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    if model_name.startswith('dinov3_'):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def normalize_batch(batch, mean, std):
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(batch.device)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(batch.device)
    return (batch - mean_tensor) / std_tensor


def unnormalize_batch(batch, mean, std):
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(batch.device)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(batch.device)
    return batch * std_tensor + mean_tensor


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

def get_dataloader(data_root, dataset_name, model_name, batch_size=32):
    dataset_path = os.path.join(data_root, dataset_name)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    if model_name.startswith('dinov3_'):
        transform = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
        ])

    metadata_file = None
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file == 'metadata.csv':
                metadata_file = os.path.join(root, file)
                break
        if metadata_file:
            break

    if not metadata_file:
        raise FileNotFoundError(f"No metadata.csv found in {dataset_path}")

    dataset = TrafficSignDataset(
        root_dir=dataset_path,
        metadata_file=metadata_file,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    return loader, len(dataset.class_to_idx)


def get_filtered_metadata(original_metadata, sources):
    df = pd.read_csv(original_metadata)
    if sources:
        filtered_df = df[df['source'].isin(sources)]
    else:
        filtered_df = df

    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    filtered_df.to_csv(temp_file.name, index=False)
    return temp_file.name


def save_images(images, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = torch.clamp(images, 0, 1)

    for i, filename in enumerate(filenames):
        img = images[i].permute(1, 2, 0).cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, filename))


def create_adv_metadata(clean_meta_path, filenames, adv_dir):
    clean_meta = pd.read_csv(clean_meta_path)
    adv_meta = clean_meta[clean_meta['image_path'].apply(os.path.basename).isin(filenames)].copy()
    adv_meta['image_path'] = adv_meta['image_path'].apply(
        lambda x: os.path.join(adv_dir, os.path.basename(x)))
    return adv_meta