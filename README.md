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

In order to have a more robust model, data was pulled from the "Train" split at a rate of 15% per split from a particular data subset prior to training.
To facilitate cross-model validation and testing, splits for train, test and validation were predefined in csv files to prevent data leakage in the form of Train-Test Contamination.

---
Breaking down the dataset further and we can view the class distribution.

### Unified Classes Table  
| Class            |  Number of Samples |  
|------------------|------|  
| ahead_only       | 1438  |
| curve            | 2596  |
| goods_vehicles   | 587   |
| keep_left        | 1345  |
| keep_right       | 2691  |
| no_entry         | 2089  |
| no_left_turn     | 1231  |
| no_overtaking    | 1613  |
| no_parking       | 3518  |
| no_right_turn    | 1004  |
| no_stopping      | 3079  |
| no_u_turn        | 1297  |
| parking          | 2690  |
| priority_road    | 1657  |
| roundabout       | 1827  |
| speed_limit      | 14910  |
| stop             | 2552  |
| turn_left        | 2016  |
| turn_right       | 2029  |
| warning          | 2063  |
| yield            | 4067  |
 

## Supported Models

| Model Name      |                                                        Architecture Type    | Pretrained Source | Input Size | Normalization Parameters |
|-----------------                                                        |-------------------|-------------------|-----------|--------------------------|
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer)       | Vision Transformer | TIMM              | 224x224    | ImageNet standard        |
| [Google ViT](https://github.com/google-research/vision_transformer)     | Vision Transformer | TIMM              | 224x224    | ImageNet standard        |
| [Dino V1](https://github.com/facebookresearch/dino)                     | Vision Transformer | TIMM            | 224x224    | ImageNet standard          |
| [Yolo V11(Classification)](https://docs.ultralytics.com/models/yolo11/) | CNN                | TIMM              | 224x224    | ImageNet standard        |
| [ConvNeXt-Base](https://github.com/facebookresearch/ConvNeXt)           | CNN                | TIMM              | 224x224    | ImageNet standard        |

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
OUTDATED

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
```
python whitebox_attacks.py \
--data_root ./path_to_top_of_dataset \
--models google_vit \
--sources gtsrb \
--model_base_path ./Models/ \
--output_dir ./adv_attacks \
```

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
--data_root ./path_to_top_of_dataset \
--model swin \
--source lisa \
--model_path ./results/swin/xxxx_best_model.pth \
--output_dir ./adv_attacks \
--batch_size 32 \
--patch_size 32 \
--patch_sample_size 100 \
--scale_min 0.2 \
--scale_max 0.8 \
--patch_type circle square
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
**In Dev**
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

python train_loras.py \
--data_root ./path_to_data \
--adv_root ./path_to_adv_data/ \
--models swin \
--sources mapillary \
--model_base_path ./results/swin/best_model.pth \
--output_dir ./loras \
--batch_size 32 \
--ranks 8 16 32 \
--attacks patch_circle patch_square pgd fgsm
```

## Composability
    parser.add_argument('--model_path', required=True, help='Path to base fine-tuned model')
    parser.add_argument('--lora_root', required=True, help='Root directory containing LoRA adapters')
    parser.add_argument('--adv_root', required=True, help='Root directory for adversarial examples')
    parser.add_argument('--data_root', required=True, help='Root directory for clean examples')
    parser.add_argument('--attacks', nargs='+', required=True, help='List of attacks to evaluate')
    parser.add_argument('--rank', type=int, required=True, help='Rank value to evaluate (e.g., 16)')
    parser.add_argument('--output_file', default='test_results.json', help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_mode', choices=['all', 'base_only', 'individual_only', 'combinations_only'],
                        default='all', help='What to test: all, base_only, individual_only, or combinations_only')

