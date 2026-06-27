"""
===========================================================
VALIDATED MIXED-PRECISION SCHEDULER (VGG-16)
===========================================================

VGG-16 adapted for CIFAR-10 — a classic deep CNN with
13 conv layers + 3 FC layers (~15M parameters).
No skip connections means quantization errors propagate
without residual correction — interesting contrast to ResNet.

This script compares:
  OLD: Greedy scheduler (additive estimation — may exceed budget)
  NEW: Validated scheduler (real inference — guaranteed within budget)

Expected runtime on Kaggle T4:
  Training (100 epochs):  ~25 min
  Sensitivity analysis:   ~5 min
  Validated scheduling:   ~10 min (3 budgets x ~16 layers)
  Total:                  ~40 min

===========================================================
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

# --- Kaggle: add dataset path for nvfp4_utils ---
sys.path.append('/kaggle/input/nvfp4-utils')

from nvfp4_utils import (
    run_sensitivity_analysis,
    greedy_precision_scheduler,
    validated_greedy_scheduler,
    apply_precision_map,
    compute_memory_savings,
    evaluate_model,
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")

BUDGETS = [1.0, 2.0, 5.0]


# =========================================================
# VGG-16 MODEL (adapted for CIFAR-10, 32x32 input)
# =========================================================

class VGG16_CIFAR(nn.Module):
    """
    VGG-16 with BatchNorm for CIFAR-10.
    13 conv layers + 3 FC layers = 16 quantizable layers.
    ~15M parameters.
    """

    def __init__(self, num_classes=10):
        super(VGG16_CIFAR, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 4: 4x4 -> 2x2
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Block 5: 2x2 -> 1x1
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =========================================================
# DATA LOADING
# =========================================================

print("\n=================================================")
print("LOADING CIFAR-10 DATASET")
print("=================================================\n")

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train)

test_dataset = datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2)

print(f"Device: {DEVICE}")


# =========================================================
# TRAINING
# =========================================================

print("\n=================================================")
print("TRAINING VGG-16 (100 epochs)")
print("=================================================\n")

model = VGG16_CIFAR().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
n_quant = sum(
    1 for n, p in model.named_parameters()
    if 'weight' in n and p.dim() >= 2
)
print(f"Total parameters: {total_params:,}")
print(f"Quantizable layers: {n_quant}\n")

optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE,
    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

lr_sched = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[50, 75], gamma=0.1)

criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = total = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
    lr_sched.step()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS} | "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Acc: {100.*correct/total:.2f}%")


# =========================================================
# SENSITIVITY ANALYSIS
# =========================================================

print("\n=================================================")
print("SENSITIVITY ANALYSIS")
print("=================================================\n")

fp32_acc, sensitivity = run_sensitivity_analysis(
    model, test_loader, DEVICE)

print(f"\nFP32 Baseline: {fp32_acc:.2f}%")
print(f"Total quantizable layers: {len(sensitivity)}\n")

for r in sensitivity:
    print(f"  {r['layer']:<45} "
          f"Drop: {r['acc_drop']:>6.2f}% | "
          f"CosSim: {r['cos_sim']:.4f}")


# =========================================================
# COMPARISON: OLD vs NEW SCHEDULER
# =========================================================

for budget in BUDGETS:

    print(f"\n{'='*70}")
    print(f"  BUDGET: {budget:.0f}%")
    print(f"{'='*70}")

    # ---- OLD: Greedy (additive estimation) ----
    print(f"\n  --- OLD: Greedy Scheduler (additive estimation) ---")

    old_map, old_est_drop = greedy_precision_scheduler(
        sensitivity, budget)

    old_model = apply_precision_map(model, old_map)
    old_acc, _ = evaluate_model(old_model, test_loader, DEVICE)
    old_drop = fp32_acc - old_acc

    old_mem = compute_memory_savings(sensitivity, old_map)

    old_nvfp4 = sum(1 for v in old_map.values() if v == 'NVFP4')
    old_bf16 = sum(1 for v in old_map.values() if v == 'BF16')
    old_fp32 = sum(1 for v in old_map.values() if v == 'FP32')

    print(f"  Accuracy:     {old_acc:.2f}%")
    print(f"  Est. Drop:    {old_est_drop:.2f}%")
    print(f"  Actual Drop:  {old_drop:.2f}%")
    print(f"  Within Budget: "
          f"{'YES' if old_drop <= budget else 'NO'}")
    print(f"  Compression:  {old_mem['compression_ratio']:.2f}x")
    print(f"  Distribution: "
          f"{old_nvfp4} NVFP4 / {old_bf16} BF16 / {old_fp32} FP32")

    # ---- NEW: Validated (real inference) ----
    print(f"\n  --- NEW: Validated Scheduler (guaranteed) ---")

    new_map, history = validated_greedy_scheduler(
        model, sensitivity, test_loader, DEVICE, budget)

    new_model = apply_precision_map(model, new_map)
    new_acc, _ = evaluate_model(new_model, test_loader, DEVICE)
    new_drop = fp32_acc - new_acc

    new_mem = compute_memory_savings(sensitivity, new_map)

    new_nvfp4 = sum(1 for v in new_map.values() if v == 'NVFP4')
    new_bf16 = sum(1 for v in new_map.values() if v == 'BF16')
    new_fp32 = sum(1 for v in new_map.values() if v == 'FP32')

    print(f"\n  Accuracy:     {new_acc:.2f}%")
    print(f"  Actual Drop:  {new_drop:.2f}%")
    print(f"  Within Budget: "
          f"{'YES' if new_drop <= budget else 'NO'}")
    print(f"  Compression:  {new_mem['compression_ratio']:.2f}x")
    print(f"  Distribution: "
          f"{new_nvfp4} NVFP4 / {new_bf16} BF16 / {new_fp32} FP32")

    # ---- Side-by-side comparison ----
    print(f"\n  --- COMPARISON ---")
    print(f"  {'Metric':<25} {'OLD (greedy)':>15} "
          f"{'NEW (validated)':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<25} {old_acc:>14.2f}% "
          f"{new_acc:>14.2f}%")
    print(f"  {'Drop':<25} {old_drop:>14.2f}% "
          f"{new_drop:>14.2f}%")
    print(f"  {'Within Budget':<25} "
          f"{'YES' if old_drop <= budget else 'NO':>15} "
          f"{'YES' if new_drop <= budget else 'NO':>15}")
    print(f"  {'Compression':<25} "
          f"{old_mem['compression_ratio']:>14.2f}x "
          f"{new_mem['compression_ratio']:>14.2f}x")
    print(f"  {'Memory (KB)':<25} "
          f"{old_mem['mixed_KB']:>14.2f} "
          f"{new_mem['mixed_KB']:>14.2f}")
    print(f"  {'NVFP4 layers':<25} {old_nvfp4:>15} "
          f"{new_nvfp4:>15}")
    print(f"  {'BF16 layers':<25} {old_bf16:>15} "
          f"{new_bf16:>15}")
    print(f"  {'FP32 layers':<25} {old_fp32:>15} "
          f"{new_fp32:>15}")


print(f"\n{'='*70}")
print(f"  DONE — VGG-16")
print(f"{'='*70}")
