"""
===========================================================
RESNET-20 LAYER SENSITIVITY ANALYSIS (CIFAR-10)
===========================================================

This script:
1. Trains ResNet-20 on CIFAR-10 (~91% accuracy)
2. Evaluates FP32 baseline accuracy
3. Evaluates full NVFP4 quantization
4. Performs layer-by-layer sensitivity analysis (22 layers)
5. Assigns precision (NVFP4 / BF16 / FP32) per layer

Expected: A clear mix of NVFP4, BF16, and FP32 assignments
because early layers (near input) are more sensitive than
late layers.

===========================================================
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

# --- Kaggle: add dataset path for nvfp4_utils ---
sys.path.append('/kaggle/input/nvfp4-utils')

from nvfp4_utils import (
    evaluate_model,
    quantize_single_layer,
    quantize_all_layers,
    compute_cosine_sim,
    compute_kl_divergence,
    compute_activation_mse,
    assign_precision,
    print_sensitivity_table,
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
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")


# =========================================================
# RESNET-20 MODEL
# =========================================================

class BasicBlock(nn.Module):
    """Residual block with two 3x3 conv layers."""

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10.

    Architecture:
        Conv1 (3→16) → 3×BasicBlock(16) →
        3×BasicBlock(32, stride=2) →
        3×BasicBlock(64, stride=2) →
        AvgPool → FC(64→10)

    Total: ~272K parameters, 22 quantizable layers
    """

    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_planes, planes,
                    num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_planes = in_planes
        for s in strides:
            layers.append(
                BasicBlock(current_planes, planes, s)
            )
            current_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
        (0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Device:        {DEVICE}")


# =========================================================
# TRAINING
# =========================================================

print("\n=================================================")
print("TRAINING RESNET-20")
print("=================================================\n")

model = ResNet20().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[50, 75], gamma=0.1
)

criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    scheduler.step()
    train_acc = 100.0 * correct / total

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Loss: {running_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.4f}"
        )


# =========================================================
# FP32 BASELINE
# =========================================================

print("\n=================================================")
print("FP32 BASELINE EVALUATION")
print("=================================================\n")

fp32_acc, fp32_logits = evaluate_model(
    model, test_loader, DEVICE
)

print(f"FP32 Baseline Accuracy: {fp32_acc:.2f}%")


# =========================================================
# FULL NVFP4 EVALUATION
# =========================================================

print("\n=================================================")
print("FULL NVFP4 EVALUATION")
print("=================================================\n")

full_nvfp4_model = quantize_all_layers(model)
full_nvfp4_acc, full_nvfp4_logits = evaluate_model(
    full_nvfp4_model, test_loader, DEVICE
)

print(f"Full NVFP4 Accuracy:    {full_nvfp4_acc:.2f}%")
print(f"Accuracy Drop:          "
      f"{fp32_acc - full_nvfp4_acc:.2f}%")


# =========================================================
# LAYER-BY-LAYER SENSITIVITY ANALYSIS
# =========================================================

print("\n=================================================")
print("LAYER-BY-LAYER SENSITIVITY ANALYSIS")
print("=================================================\n")

results = []

quantizable_layers = [
    name for name, param in model.named_parameters()
    if 'weight' in name and param.dim() >= 2
]

print(f"Quantizable layers ({len(quantizable_layers)}):")
for name in quantizable_layers:
    param = dict(model.named_parameters())[name]
    print(f"  {name:<45} shape={list(param.shape)}")

print()

for i, layer_name in enumerate(quantizable_layers):

    print(f"  [{i+1:>2}/{len(quantizable_layers)}] "
          f"Analyzing: {layer_name} ...", end=" ",
          flush=True)

    q_model = quantize_single_layer(model, layer_name)

    q_acc, q_logits = evaluate_model(
        q_model, test_loader, DEVICE
    )

    acc_drop = fp32_acc - q_acc

    cos_sim = compute_cosine_sim(
        fp32_logits, q_logits
    )

    kl_div = compute_kl_divergence(
        fp32_logits, q_logits
    )

    mse = compute_activation_mse(
        fp32_logits, q_logits
    )

    precision = assign_precision(acc_drop, cos_sim)

    results.append({
        'layer': layer_name,
        'accuracy': q_acc,
        'acc_drop': acc_drop,
        'cos_sim': cos_sim,
        'kl_div': kl_div,
        'mse': mse,
        'precision': precision,
    })

    print(f"Acc: {q_acc:.2f}% (Drop: {acc_drop:.2f}%)")


# =========================================================
# RESULTS TABLE
# =========================================================

print_sensitivity_table(results, "ResNet-20 on CIFAR-10")


# =========================================================
# SUMMARY
# =========================================================

print("\n=================================================")
print("PRECISION ASSIGNMENT SUMMARY")
print("=================================================\n")

nvfp4_count = sum(
    1 for r in results if r['precision'] == 'NVFP4'
)
bf16_count = sum(
    1 for r in results if r['precision'] == 'BF16'
)
fp32_count = sum(
    1 for r in results if r['precision'] == 'FP32'
)

for r in results:
    print(f"  {r['layer']:<45} -> {r['precision']}")

print(f"\n  Assignment Distribution:")
print(f"    NVFP4: {nvfp4_count} layers")
print(f"    BF16:  {bf16_count} layers")
print(f"    FP32:  {fp32_count} layers")

print(f"\n  FP32 Baseline:     {fp32_acc:.2f}%")
print(f"  Full NVFP4:        {full_nvfp4_acc:.2f}%")
print(f"  Full NVFP4 Drop:   "
      f"{fp32_acc - full_nvfp4_acc:.2f}%")
