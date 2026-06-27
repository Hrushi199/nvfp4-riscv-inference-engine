"""
===========================================================
CNN LAYER SENSITIVITY ANALYSIS (CIFAR-10)
===========================================================

This script:
1. Trains a small CNN on CIFAR-10
2. Evaluates FP32 baseline accuracy
3. Evaluates full NVFP4 quantization
4. Performs layer-by-layer sensitivity analysis
5. Assigns precision (NVFP4 / BF16 / FP32) per layer

===========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random

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
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")


# =========================================================
# CNN MODEL DEFINITION
# =========================================================

class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10 classification.

    Architecture:
        Conv1 (3->32, 3x3) -> BN -> ReLU
        Conv2 (32->64, 3x3) -> BN -> ReLU -> MaxPool
        Conv3 (64->128, 3x3) -> BN -> ReLU -> MaxPool
        FC1 (128*6*6 -> 256) -> ReLU -> Dropout
        FC2 (256 -> 10)
    """

    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool2(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)

        return x


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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Device:        {DEVICE}")


# =========================================================
# TRAINING
# =========================================================

print("\n=================================================")
print("TRAINING CNN")
print("=================================================\n")

model = SmallCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch:>2}/{EPOCHS} | "
        f"Loss: {running_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.2f}%"
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
print(f"Accuracy Drop:          {fp32_acc - full_nvfp4_acc:.2f}%")


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

print(f"Quantizable layers: {quantizable_layers}\n")

for layer_name in quantizable_layers:

    print(f"  Analyzing: {layer_name} ...", end=" ")

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

print_sensitivity_table(results, "CNN on CIFAR-10")


# =========================================================
# SUMMARY
# =========================================================

print("\n=================================================")
print("PRECISION ASSIGNMENT SUMMARY")
print("=================================================\n")

for r in results:
    print(f"  {r['layer']:<25} -> {r['precision']}")

print(f"\n  FP32 Baseline:     {fp32_acc:.2f}%")
print(f"  Full NVFP4:        {full_nvfp4_acc:.2f}%")
print(f"  Full NVFP4 Drop:   {fp32_acc - full_nvfp4_acc:.2f}%")
