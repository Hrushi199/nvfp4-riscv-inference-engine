"""
===========================================================
VALIDATED MIXED-PRECISION SCHEDULER: MLP ON MNIST
===========================================================

Compares OLD (additive estimation) vs NEW (validated) scheduler.
The validated scheduler evaluates actual accuracy after each
layer, guaranteeing the result stays within budget.

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
    evaluate_model,
    compute_memory_savings,
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

BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")

BUDGETS = [1.0, 2.0, 5.0]


# =========================================================
# MLP MODEL
# =========================================================

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# =========================================================
# DATA LOADING
# =========================================================

print("\n=================================================")
print("LOADING MNIST")
print("=================================================\n")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform)

test_dataset = datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transform)

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
print("TRAINING MLP (5 epochs)")
print("=================================================\n")

model = MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = correct = total = 0
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
    print(f"Epoch {epoch}/{EPOCHS} | "
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

print(f"\nFP32 Baseline: {fp32_acc:.2f}%\n")

for r in sensitivity:
    print(f"  {r['layer']:<25} "
          f"Drop: {r['acc_drop']:>6.2f}% | "
          f"CosSim: {r['cos_sim']:.4f}")


# =========================================================
# OLD vs NEW SCHEDULER COMPARISON
# =========================================================

for budget in BUDGETS:

    print(f"\n{'='*70}")
    print(f"  BUDGET: {budget:.0f}%")
    print(f"{'='*70}")

    # ---- OLD ----
    print(f"\n  --- OLD: Greedy (additive estimation) ---")
    old_map, old_est = greedy_precision_scheduler(
        sensitivity, budget)
    old_model = apply_precision_map(model, old_map)
    old_acc, _ = evaluate_model(old_model, test_loader, DEVICE)
    old_drop = fp32_acc - old_acc
    old_mem = compute_memory_savings(sensitivity, old_map)
    old_n = sum(1 for v in old_map.values() if v == 'NVFP4')
    old_b = sum(1 for v in old_map.values() if v == 'BF16')
    old_f = sum(1 for v in old_map.values() if v == 'FP32')
    print(f"  Acc: {old_acc:.2f}% | Drop: {old_drop:.2f}% | "
          f"Budget OK: {'YES' if old_drop <= budget else 'NO'} | "
          f"Compress: {old_mem['compression_ratio']:.2f}x | "
          f"{old_n}/{old_b}/{old_f}")

    # ---- NEW ----
    print(f"\n  --- NEW: Validated (guaranteed) ---")
    new_map, history = validated_greedy_scheduler(
        model, sensitivity, test_loader, DEVICE, budget)
    new_model = apply_precision_map(model, new_map)
    new_acc, _ = evaluate_model(new_model, test_loader, DEVICE)
    new_drop = fp32_acc - new_acc
    new_mem = compute_memory_savings(sensitivity, new_map)
    new_n = sum(1 for v in new_map.values() if v == 'NVFP4')
    new_b = sum(1 for v in new_map.values() if v == 'BF16')
    new_f = sum(1 for v in new_map.values() if v == 'FP32')
    print(f"\n  Acc: {new_acc:.2f}% | Drop: {new_drop:.2f}% | "
          f"Budget OK: {'YES' if new_drop <= budget else 'NO'} | "
          f"Compress: {new_mem['compression_ratio']:.2f}x | "
          f"{new_n}/{new_b}/{new_f}")

    # ---- Compare ----
    print(f"\n  {'Metric':<20} {'OLD':>12} {'NEW':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Accuracy':<20} {old_acc:>11.2f}% {new_acc:>11.2f}%")
    print(f"  {'Drop':<20} {old_drop:>11.2f}% {new_drop:>11.2f}%")
    print(f"  {'Within Budget':<20} "
          f"{'YES' if old_drop <= budget else 'NO':>12} "
          f"{'YES' if new_drop <= budget else 'NO':>12}")
    print(f"  {'Compression':<20} "
          f"{old_mem['compression_ratio']:>11.2f}x "
          f"{new_mem['compression_ratio']:>11.2f}x")

print(f"\n{'='*70}")
print(f"  DONE — MLP")
print(f"{'='*70}")
