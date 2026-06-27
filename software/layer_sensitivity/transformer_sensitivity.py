"""
===========================================================
TINY TRANSFORMER LAYER SENSITIVITY ANALYSIS (MNIST)
===========================================================

This script:
1. Trains a tiny Transformer encoder on MNIST
2. Evaluates FP32 baseline accuracy
3. Evaluates full NVFP4 quantization
4. Performs layer-by-layer sensitivity analysis
5. Assigns precision (NVFP4 / BF16 / FP32) per layer

Architecture:
    Patch Embedding (7x7 patches -> 16 patches)
    -> Positional Encoding
    -> 2x Transformer Encoder Layers
    -> Classification Head

===========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
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
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")

# Transformer config
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
PATCH_SIZE = 7
NUM_PATCHES = 16   # 28/7 = 4, 4x4 = 16 patches


# =========================================================
# PATCH EMBEDDING
# =========================================================

class PatchEmbedding(nn.Module):
    """
    Split image into patches and project to d_model.
    """

    def __init__(self, patch_size=7, d_model=64):
        super().__init__()

        self.patch_size = patch_size

        # Each patch is patch_size * patch_size pixels
        self.projection = nn.Linear(
            patch_size * patch_size, d_model
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        b, c, h, w = x.shape
        p = self.patch_size

        # Reshape into patches
        x = x.unfold(2, p, p).unfold(3, p, p)
        # x: (batch, 1, h/p, w/p, p, p)

        x = x.contiguous().view(b, -1, p * p)
        # x: (batch, num_patches, patch_dim)

        x = self.projection(x)
        # x: (batch, num_patches, d_model)

        return x


# =========================================================
# TINY TRANSFORMER MODEL
# =========================================================

class TinyTransformer(nn.Module):
    """
    Tiny Transformer for MNIST classification.

    Architecture:
        Patch Embedding -> Positional Encoding
        -> TransformerEncoder (2 layers)
        -> Mean Pooling -> Classification Head
    """

    def __init__(self, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128,
                 num_patches=16, num_classes=10):

        super().__init__()

        self.patch_embed = PatchEmbedding(
            patch_size=PATCH_SIZE, d_model=d_model
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer norm before classifier
        self.ln = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoder
        x = self.transformer(x)

        # Mean pooling over patches
        x = x.mean(dim=1)

        # Layer norm + classify
        x = self.ln(x)
        x = self.classifier(x)

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
print("TRAINING TINY TRANSFORMER")
print("=================================================\n")

model = TinyTransformer(
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_ff=DIM_FF,
    num_patches=NUM_PATCHES
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\n")

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

print(f"Quantizable layers ({len(quantizable_layers)}):")
for ln in quantizable_layers:
    param = dict(model.named_parameters())[ln]
    print(f"  {ln:<45} shape={list(param.shape)}")
print()

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

print_sensitivity_table(results, "Tiny Transformer on MNIST")


# =========================================================
# SUMMARY
# =========================================================

print("\n=================================================")
print("PRECISION ASSIGNMENT SUMMARY")
print("=================================================\n")

for r in results:
    print(f"  {r['layer']:<45} -> {r['precision']}")

print(f"\n  FP32 Baseline:     {fp32_acc:.2f}%")
print(f"  Full NVFP4:        {full_nvfp4_acc:.2f}%")
print(f"  Full NVFP4 Drop:   {fp32_acc - full_nvfp4_acc:.2f}%")
