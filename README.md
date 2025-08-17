# LoRA as a Flexible Framework for Securing Large Vision Systems

## Introduction

The official PyTorch implementation of LoRA as a Flexible Framework.

## Datasets
A unified dataset was developed from the following traffic and road sign datasets to facilitate training models on traffic sign classification.

| Dataset Name                 | Splits Supported | Samples | Region | Description |
|------------------------------|------------------|---------|--------|-------------|
| [GTSRB German Traffic Sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)    | train, test      |         | Europe | Standardized European traffic signs   |
| [LISA Road Sign](https://git-disl.github.io/GTDLBench/datasets/lisa_traffic_sign_dataset/)               | train, val, test |         | USA | Variety of USA specific signs       |
| [Mapillary Traffic Sign](https://www.mapillary.com/dataset/trafficsign)       | train, val, test |         | Global | Large and diverse dataset     |
| [CURE-TSD](https://github.com/olivesgatech/CURE-TSD)                     | train, test      |         | USA |  Synthetic and real signs      |
| [Roboflow Traffic Signs](https://universe.roboflow.com/usmanchaudhry622-gmail-com/traffic-and-road-signs)       | train, val, test |         | Global| Diverse dataset     |

In order to have a more robust model, data was pulled from the "Train" split at a rate of 15% per split missing from a particular data subset prior to training.
To facilitate cross-model validation and testing, splits for train, test and validation were predefined in csv files to prevent data leakage in the form of Train-Test Contamination.

The full unified dataset can be found: [Unified Dataset](https://drive.google.com/drive/folders/10_NTgU7khKPmxbTD26_RbU4eiQh7mQDl?usp=drive_link).

The dataset after being restructured can be found: [Global Dataset](https://drive.google.com/drive/folders/1JV6WgSIYHnKQBHy7hzP_yTz7iSHfjjnc?usp=sharing).

The "Global Dataset" is used for all training, testing and attack generation going forward.

---
Breaking down the dataset further and we can view the class distribution.

### Unified Classes Table  
| Unified Class          |  Number of Samples |  
|------------------------|---|  
| speed_limit            |   |  
| no_overtaking          |   |  
| priority_road          |   |  
| yield                  |   |  
| stop                   |   |  
| no_vehicles            |   |  
| goods_vehicles         |   |  
| no_entry               |   |  
| curve                  |   |  
| bump                   |   |  
| slippery_road          |   |  
| warning                |   |  
| road_work              |   |  
| pedestrian_crossing    |   |  
| school_zone            |   |  
| bicycle_crossing       |   |  
| wild_animals           |   |  
| no_left_turn           |   |  
| no_right_turn          |   |  
| directional            |   |  
| keep_right             |   |  
| keep_left              |   |  
| roundabout             |   |  
| ahead_only             |   |  
| turn_left              |   |  
| turn_right             |   |  
| no_parking             |   |  
| no_stopping            |   |  
| parking                |   |  
| bus_stop               |   |  
| rest_area              |   |  
| railway_crossing       |   |  
| other                  |   |

## Supported Models

| Model Name      |                                                     Architecture Type    | Pretrained Source | Input Size | Normalization Parameters |
|-----------------                                                      |-------------------|-------------------|-----------|--------------------------|
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer)     | Vision Transformer | TIMM              | 224x224    | ImageNet standard        |
| [Google ViT](https://github.com/google-research/vision_transformer) | Vision Transformer | TIMM              | 224x224    | ImageNet standard        |
| [Dino V1](https://github.com/facebookresearch/dino)                                | Vision Transformer | TIMM            | 224x224    | ImageNet standard          |
| [Yolo V11(Classification)](https://docs.ultralytics.com/models/yolo11/)         | CNN                | TIMM              | 224x224    | ImageNet standard        |
| [ConvNeXt-Base](https://github.com/facebookresearch/ConvNeXt)         | CNN                | TIMM              | 224x224    | ImageNet standard        |

## Training

| Parameter       | Default Value | Description                           |
|-----------------|---------------|---------------------------------------|
| `--data_root`   | ./Datasets/Adjust_Global_Dataset            | Images for train, val and test |
| `--model`      | all            | Target model                       |
| `--batch_size`  | 64            | Input batch size                      |
| `--epochs`      | 50            | Training epochs                       |
| `--lr`          | 1e-4          | Learning rate                         |
| `--output_dir`| ./results          | Where to save results of training                |
| `--seed`        | 42            | Random seed                           |
| `--sources` | all             | Which subset to train on, see [Datasets]                  |
| Learning Rate Schedule | StepLR (step=5, γ=0.1) | Reduces LR every 5 epochs |

Finetuned models can be found on [Kaggle](https://www.kaggle.com/models/richardneddo/lora-as-a-flexible-framework/).

### Running Examples
#### Train Clip Model
```
python train.py --model swin dinov1 --batch_size 96 --epochs 24 --sources gtsrb
```
Will train swin then dinov1 on GTSRB data.

## Adverserial Attacks

### whitebox_attacks.py
- **Attack Types**: FGSM (Fast Gradient Sign Method) + PGD (Projected Gradient Descent)
- **Characteristics**: 
  - Digital attacks targeting model gradients
  - FGSM: Single-step perturbation
  - PGD: Iterative refinement of FGSM
- **Sign Area Constraint**: Applies perturbations only to sign regions
```
python whitebox_attacks.py \
--data_root ./processed \
--model swin \
--model_path ./results/swin/best_model.pth \
--output_dir ./adv_attacks \
--epsilon 0.03 \
--pgd_alpha 0.01 \
--pgd_iters 40 \
--datasets train val test
```
### auto_attack.py
- **Attack Type**: AutoAttack (Combination of 4 attacks)
- **Characteristics**:
  - State-of-the-art ensemble attack
  - Parameter-free and adaptive

Run prior:
```pip install git+https://github.com/fra31/auto-attack```
```
python auto_attack.py \
--data_root ./processed \
--model swin \
--model_path ./results/swin/best_model.pth \
--output_dir ./adv_attacks \
--epsilon 0.03 \
--datasets test
```
### patch_attack.py
- **Attack Type**: Physical adversarial patch
- **Characteristics**:
  - Printable real-world perturbation
  - Optimized for robustness to transformations
  - Generated per-split (train, val, test) using random samples
- **Parameters**:
  - `--patch_size`: Physical size of patch
  - `--patch_sample_size`: Images for patch generation
```
python patch_attack.py \
--data_root ./processed \
--model swin \
--model_path ./results/swin/best_model.pth \
--output_dir ./adv_attacks \
--patch_size 32 \
--patch_sample_size 100 \
--datasets train val test
```
### rp2_attack.py
- **Attack Type**: Robust physical perturbation
- **Characteristics**:
  - Per-class physical perturbations
  - Optimized for real-world conditions
  - Non-targeted misclassification
- **Parameters**:
  - `--sample_per_class`: Images per class for perturbation
```
python rp2_attack.py \
--data_root ./processed \
--model swin \
--model_path ./results/swin/best_model.pth \
--output_dir ./adv_attacks \
--patch_size 32 \
--sample_per_class 50 \
--datasets test
```
## LoRA Training



## Composability
TBD


