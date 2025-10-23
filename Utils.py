import os
import tempfile
import pandas as pd
import torch
import numpy as np
from PIL import Image
import csv
from torch.utils.data import Dataset
from transformers import ViTForImageClassification


class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.filenames = []
        self.available_classes = set()

        metadata_dir = os.path.dirname(metadata_file)

        with open(metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']

                if os.path.exists(img_path):
                    final_img_path = img_path
                else:
                    relative_to_metadata = os.path.join(metadata_dir, img_path)
                    if os.path.exists(relative_to_metadata):
                        final_img_path = relative_to_metadata
                    else:
                        relative_to_root = os.path.join(self.root_dir, img_path)
                        if os.path.exists(relative_to_root):
                            final_img_path = relative_to_root
                        else:
                            if os.path.exists(img_path):
                                final_img_path = img_path
                            else:
                                print(f"Warning: Image file not found: {img_path}")
                                print(f"  Tried: {img_path}")
                                print(f"  Tried: {relative_to_metadata}")
                                print(f"  Tried: {relative_to_root}")
                                continue

                final_img_path = os.path.normpath(final_img_path)

                label = row['unified_class']
                filename = os.path.basename(final_img_path)

                self.available_classes.add(label)

                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)

                self.samples.append((final_img_path, self.class_to_idx[label]))
                self.filenames.append(filename)

        self.class_to_idx = {cls: idx for cls, idx in self.class_to_idx.items()
                             if cls in self.available_classes}

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.available_classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        print(f"Loaded {len(self.samples)} samples from {metadata_file}")

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

def create_vit_model(num_classes, pretrained=True):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model

def get_normalization(model_name):
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

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