"""
===========================================================
VALIDATED MIXED-PRECISION SCHEDULER (RESNET-20)
===========================================================

This script compares:
  OLD: Greedy scheduler (additive estimation — may exceed budget)
  NEW: Validated scheduler (real inference — guaranteed within budget)

The validated scheduler evaluates actual accuracy after each
layer assignment, so the result ALWAYS stays within budget.

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
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")

BUDGETS = [1.0, 2.0, 5.0]


# =========================================================
# RESNET-20 MODEL
# =========================================================

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):

    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_planes, planes,
                    num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current = in_planes
        for s in strides:
            layers.append(BasicBlock(current, planes, s))
            current = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)


# =========================================================
# DATA LOADING
# =========================================================

print("\n=================================================")
print("LOADING CIFAR-10")
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
    shuffle=True, num_workers=0)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0)

print(f"Device: {DEVICE}")


# =========================================================
# TRAINING
# =========================================================

print("\n=================================================")
print("TRAINING RESNET-20 (100 epochs)")
print("=================================================\n")

model = ResNet20().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

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
# SENSITIVITY ANALYSIS (shared by both schedulers)
# =========================================================

print("\n=================================================")
print("SENSITIVITY ANALYSIS")
print("=================================================\n")

fp32_acc, sensitivity = run_sensitivity_analysis(
    model, test_loader, DEVICE)

print(f"\nFP32 Baseline: {fp32_acc:.2f}%\n")

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
print(f"  DONE")
print(f"{'='*70}")
