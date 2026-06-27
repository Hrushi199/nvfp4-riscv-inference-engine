"""
===========================================================
MLP LAYER SENSITIVITY ANALYSIS (MNIST)
===========================================================

This script:
1. Trains a 3-layer MLP on MNIST
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

BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")


# =========================================================
# MLP MODEL DEFINITION
# =========================================================

class MLP(nn.Module):
    """
    3-layer MLP for MNIST classification.

    Architecture:
        Input (784) -> FC1 (256) -> ReLU
                    -> FC2 (128) -> ReLU
                    -> FC3 (10)
    """

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
        x = self.fc3(x)

        return x


# =========================================================
# DATA LOADING
# =========================================================

print("\n=================================================")
print("LOADING MNIST DATASET")
print("=================================================\n")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transform
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
print("TRAINING MLP")
print("=================================================\n")

model = MLP().to(DEVICE)
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
        f"Epoch {epoch}/{EPOCHS} | "
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

# Get all quantizable layers (weight matrices)
quantizable_layers = [
    name for name, param in model.named_parameters()
    if 'weight' in name and param.dim() >= 2
]

print(f"Quantizable layers: {quantizable_layers}\n")

for layer_name in quantizable_layers:

    print(f"  Analyzing: {layer_name} ...", end=" ")

    # Quantize only this layer
    q_model = quantize_single_layer(model, layer_name)

    # Evaluate
    q_acc, q_logits = evaluate_model(
        q_model, test_loader, DEVICE
    )

    # Compute metrics
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

print_sensitivity_table(results, "MLP on MNIST")


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
