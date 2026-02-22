#!/usr/bin/env python3
"""
Training script for Object Detection with CNN
Designed to run on local machine or Google Colab
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from Trainer import ModelTrainer
from Datasets import CUB200

# Augmentation imports
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BboxIOU(nn.Module):
    """Computes IoU for bounding box predictions"""
    
    def xyhw_to_xyxy(self, bbox):
        """Convert (x, y, w, h) to (x_min, y_min, x_max, y_max) format"""
        return torch.cat((bbox[:, 0:1], bbox[:, 1:2],
                          bbox[:, 2:3] + bbox[:, 0:1],
                          bbox[:, 3:4] + bbox[:, 1:2]), 1)

    def bb_intersection_over_union(self, pred_xyhw, target_xyhw):
        """Calculate IoU between predicted and target bounding boxes"""
        pred_xyxy = self.xyhw_to_xyxy(pred_xyhw)
        target_xyxy = self.xyhw_to_xyxy(target_xyhw)

        # Intersection rectangle
        xA = torch.cat((pred_xyxy[:, 0:1], target_xyxy[:, 0:1]), 1).max(dim=1)[0].unsqueeze(1)
        yA = torch.cat((pred_xyxy[:, 1:2], target_xyxy[:, 1:2]), 1).max(dim=1)[0].unsqueeze(1)
        xB = torch.cat((pred_xyxy[:, 2:3], target_xyxy[:, 2:3]), 1).min(dim=1)[0].unsqueeze(1)
        yB = torch.cat((pred_xyxy[:, 3:4], target_xyxy[:, 3:4]), 1).min(dim=1)[0].unsqueeze(1)

        # Intersection and union
        x_len = nn.functional.relu(xB - xA)
        y_len = nn.functional.relu(yB - yA)
        interArea = x_len * y_len
        
        area1 = pred_xyhw[:, 2:3] * pred_xyhw[:, 3:4]
        area2 = target_xyhw[:, 2:3] * target_xyhw[:, 3:4]
        iou = interArea / (area1 + area2 - interArea + 1e-5)

        return iou

    def forward(self, predictions, data):
        pred_bbox = torch.sigmoid(predictions[:, :4])
        target_bbox = data[1].to(pred_bbox.device)
        return self.bb_intersection_over_union(pred_bbox, target_bbox)


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seeds set to {seed}")


def get_transforms(image_size=128):
    """Get data augmentation transforms"""
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['class_labels']))

    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.RandomCrop(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['class_labels']))
    
    return train_transform, test_transform


def main(args):
    """Main training function"""
    
    # Set seeds
    set_seeds(seed=42)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get transforms
    train_transform, test_transform = get_transforms(image_size=args.image_size)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_data = CUB200(args.data_root, image_size=args.image_size, transform=train_transform, test_train=0)
    test_data = CUB200(args.data_root, image_size=args.image_size, transform=test_transform, test_train=1)
    
    # Split training data
    validation_split = 0.9
    n_train_examples = int(len(train_data) * validation_split)
    n_valid_examples = len(train_data) - n_train_examples
    
    train_data, valid_data = torch.utils.data.random_split(
        train_data, [n_train_examples, n_valid_examples],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_data):,} | Val: {len(valid_data):,} | Test: {len(test_data):,}\n")
    
    # Display configuration
    print("🔒 Deterministic mode ENABLED (Reproducibility optimized). Benchmark mode DISABLED.")
    print("🚀 Configuration:")
    print(f"   EPOCHS: {args.epochs}")
    print(f"   BATCH_SIZE: {args.batch_size}")
    print(f"   LEARNING_RATE: {args.learning_rate}")
    print(f"   DEVICE: {device}")
    print(f"   IMAGE_SIZE: {args.image_size}")
    print(f"   MODEL: {args.model_name}")
    print(f"   OVERWRITE: {args.overwrite}")
    
    # Handle checkpoint before creating trainer
    checkpoint_path = os.path.join(args.save_dir, args.model_name + ".pt")
    if os.path.isfile(checkpoint_path):
        if args.overwrite:
            print(f"\n⚠️  Checkpoint exists at {checkpoint_path}")
            print(f"🔄 OVERWRITE MODE: Removing old checkpoint...")
            os.remove(checkpoint_path)
            print(f"✓ Old checkpoint removed. Starting fresh training.\n")
        elif not args.start_from_checkpoint:
            print(f"\n⚠️  WARNING: Checkpoint exists at {checkpoint_path}")
            print(f"Use --overwrite flag to delete it, or --start-from-checkpoint to continue training.\n")
            return
    
    # Initialize model
    print(f"\n📊 Initializing {args.model_name}...")
    res_net = models.resnet34(weights="IMAGENET1K_V1")
    
    # Create model trainer
    model_trainer = ModelTrainer(
        model=res_net.to(device),
        output_size=4,
        device=device,
        loss_fun=nn.BCEWithLogitsLoss(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        model_name=args.model_name,
        start_from_checkpoint=args.start_from_checkpoint,
        eval_metric=BboxIOU(),
        num_workers=args.num_workers,
    )
    
    # Set data
    model_trainer.set_data(train_set=train_data, test_set=test_data, val_set=valid_data)
    
    # Set learning rate scheduler
    model_trainer.set_lr_schedule(
        optim.lr_scheduler.StepLR(model_trainer.optimizer, step_size=1, gamma=0.95)
    )
    
    # Display model info
    num_params = sum(p.flatten().shape[0] for p in model_trainer.model.parameters())
    trainable_params = sum(p.flatten().shape[0] for p in model_trainer.model.parameters() if p.requires_grad)
    print(f"📊 Model ({args.model_name}): {num_params:,} total params, {trainable_params:,} trainable")
    
    print(f"\n🚀 TRAINING START | Device: {device} | Epochs: {args.epochs}\n")
    
    # Run training
    model_trainer.run_training(num_epochs=args.epochs)
    
    print("\n✅ Training completed!")
    print(f"📁 Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Object Detection Model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="./datasets", help="Path to datasets")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="ResNet34_CUB", help="Model checkpoint name")
    parser.add_argument("--save-dir", type=str, default="Checkpoints", help="Save directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    
    # Checkpoint arguments
    parser.add_argument("--start-from-checkpoint", action="store_true", help="Start from checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint")
    
    # DataLoader arguments
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for DataLoader (0=disable multiprocessing)")
    
    args = parser.parse_args()
    main(args)
