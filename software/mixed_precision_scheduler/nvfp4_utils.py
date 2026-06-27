"""
===========================================================
NVFP4 UTILITIES + MIXED-PRECISION SCHEDULER
===========================================================

Shared utilities for the Mixed-Precision Scheduler:

1. NVFP4 quantization (E2M1, sign-aware)
2. BF16 simulation
3. Greedy precision scheduler algorithm
4. Mixed-precision model application
5. Memory savings computation

===========================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# =========================================================
# FP4 LOOKUP TABLE (E2M1)
# =========================================================

fp4_table = {
    0b0000: 0.0,
    0b0001: 0.5,
    0b0010: 1.0,
    0b0011: 1.5,
    0b0100: 2.0,
    0b0101: 3.0,
    0b0110: 4.0,
    0b0111: 6.0,

    0b1000: -0.0,
    0b1001: -0.5,
    0b1010: -1.0,
    0b1011: -1.5,
    0b1100: -2.0,
    0b1101: -3.0,
    0b1110: -4.0,
    0b1111: -6.0,
}

FP4_MIN = -6.0
FP4_MAX = 6.0


# =========================================================
# SATURATION
# =========================================================

def saturate(x, min_val=FP4_MIN, max_val=FP4_MAX):
    if x > max_val:
        return max_val
    if x < min_val:
        return min_val
    return x


# =========================================================
# FP4 ENCODER (SIGN-AWARE)
# =========================================================

def encode_fp4(x):
    x = saturate(x)
    if x >= 0:
        search_codes = {
            k: v for k, v in fp4_table.items()
            if k <= 0b0111
        }
    else:
        search_codes = {
            k: v for k, v in fp4_table.items()
            if k >= 0b1000
        }
    encoded_bits = min(
        search_codes.keys(),
        key=lambda k: abs(search_codes[k] - x)
    )
    return encoded_bits


def decode_fp4(bits):
    return fp4_table[bits]


# =========================================================
# NVFP4 WEIGHT QUANTIZATION (SIMULATE)
# =========================================================

def quantize_weight_nvfp4(weight_tensor, block_size=16):
    """
    Simulate NVFP4 quantization on a PyTorch weight tensor.
    Block size = 16 (NVIDIA micro-block).
    """
    w = weight_tensor.detach().cpu().numpy()
    shape = w.shape
    flat = w.flatten()
    result = np.zeros_like(flat)

    for i in range(0, len(flat), block_size):
        block = flat[i:i + block_size]
        scale = np.max(np.abs(block))
        if scale == 0:
            result[i:i + len(block)] = 0.0
            continue
        normalized = block / scale
        for j, val in enumerate(normalized):
            enc = encode_fp4(val)
            dec = decode_fp4(enc)
            result[i + j] = dec * scale

    return torch.tensor(
        result.reshape(shape),
        dtype=weight_tensor.dtype
    )


# =========================================================
# BF16 WEIGHT QUANTIZATION (SIMULATE)
# =========================================================

def quantize_weight_bf16(weight_tensor):
    """
    Simulate BF16 quantization by casting to bfloat16
    and back to float32.
    """
    return weight_tensor.to(torch.bfloat16).to(torch.float32)


# =========================================================
# APPLY MIXED-PRECISION MAP TO MODEL
# =========================================================

def apply_precision_map(model, precision_map):
    """
    Clone model and apply a precision map.

    precision_map: dict mapping layer_name -> 'NVFP4'|'BF16'|'FP32'
    """
    mixed_model = copy.deepcopy(model)

    for name, param in mixed_model.named_parameters():
        if name in precision_map:
            prec = precision_map[name]

            if prec == "NVFP4":
                param.data = quantize_weight_nvfp4(param.data)

            elif prec == "BF16":
                param.data = quantize_weight_bf16(param.data)

            # FP32 -> no change

    return mixed_model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(model, test_loader, device):
    """Evaluate accuracy and collect output logits."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    all_logits = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            all_logits.append(output.cpu())
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    all_logits = torch.cat(all_logits, dim=0)
    return accuracy, all_logits


# =========================================================
# LAYER-WISE SENSITIVITY ANALYSIS
# =========================================================

def run_sensitivity_analysis(model, test_loader, device):
    """
    Run layer-by-layer sensitivity analysis.
    Returns list of dicts with per-layer metrics.
    """
    fp32_acc, fp32_logits = evaluate_model(
        model, test_loader, device
    )

    quantizable = [
        name for name, p in model.named_parameters()
        if 'weight' in name and p.dim() >= 2
    ]

    results = []
    for layer_name in quantizable:
        q_model = copy.deepcopy(model)
        target_numel = 0
        for name, param in q_model.named_parameters():
            if name == layer_name:
                param.data = quantize_weight_nvfp4(param.data)
                target_numel = param.numel()

        q_acc, q_logits = evaluate_model(
            q_model, test_loader, device
        )

        acc_drop = fp32_acc - q_acc
        cos_sim = F.cosine_similarity(
            fp32_logits, q_logits, dim=1
        ).mean().item()

        results.append({
            'layer': layer_name,
            'accuracy': q_acc,
            'acc_drop': acc_drop,
            'cos_sim': cos_sim,
            'num_params': target_numel,
        })

    return fp32_acc, results


# =========================================================
# GREEDY MIXED-PRECISION SCHEDULER
# =========================================================

def greedy_precision_scheduler(sensitivity_results,
                               accuracy_budget):
    """
    Greedy scheduler: assign the lowest possible precision
    to each layer while staying within the accuracy budget.

    Algorithm:
    1. Sort layers by acc_drop (ascending = least sensitive)
    2. Try assigning NVFP4 to each layer
    3. If estimated cumulative drop > budget, try BF16
    4. If still over budget, keep FP32

    BF16 is estimated to cause ~20% of NVFP4's drop.

    Returns:
        precision_map: dict {layer_name: precision}
    """

    # Sort: least sensitive first
    sorted_layers = sorted(
        sensitivity_results,
        key=lambda x: x['acc_drop']
    )

    precision_map = {}
    cumulative_drop = 0.0

    for layer in sorted_layers:

        name = layer['layer']
        nvfp4_drop = layer['acc_drop']

        # BF16 causes much less error than NVFP4
        # Estimate ~20% of NVFP4 drop
        bf16_drop = nvfp4_drop * 0.2

        # Try NVFP4 first (maximum compression)
        if cumulative_drop + nvfp4_drop <= accuracy_budget:
            precision_map[name] = "NVFP4"
            cumulative_drop += nvfp4_drop

        # Try BF16 (medium compression)
        elif cumulative_drop + bf16_drop <= accuracy_budget:
            precision_map[name] = "BF16"
            cumulative_drop += bf16_drop

        # Keep FP32 (no compression)
        else:
            precision_map[name] = "FP32"

    return precision_map, cumulative_drop


# =========================================================
# MEMORY SAVINGS COMPUTATION
# =========================================================

BITS_PER_PRECISION = {
    "FP32": 32,
    "BF16": 16,
    "NVFP4": 4,
}


def compute_memory_savings(sensitivity_results,
                           precision_map):
    """
    Compute memory usage for FP32 baseline vs mixed-precision.
    Accounts for NVFP4 scale factor overhead (32 bits per 16 elements).
    """

    fp32_total_bits = 0
    mixed_total_bits = 0

    details = []

    for layer in sensitivity_results:

        name = layer['layer']
        num_params = layer['num_params']
        prec = precision_map.get(name, "FP32")

        fp32_bits = num_params * 32

        if prec == "NVFP4":
            # 4 bits per param + 32-bit scale per 16 elements
            import math
            num_blocks = math.ceil(num_params / 16)
            mixed_bits = num_params * 4 + num_blocks * 32
        elif prec == "BF16":
            mixed_bits = num_params * 16
        else:
            mixed_bits = num_params * 32

        fp32_total_bits += fp32_bits
        mixed_total_bits += mixed_bits

        details.append({
            'layer': name,
            'params': num_params,
            'precision': prec,
            'fp32_KB': fp32_bits / 8 / 1024,
            'mixed_KB': mixed_bits / 8 / 1024,
        })

    compression = fp32_total_bits / mixed_total_bits

    return {
        'fp32_total_bits': fp32_total_bits,
        'mixed_total_bits': mixed_total_bits,
        'fp32_KB': fp32_total_bits / 8 / 1024,
        'mixed_KB': mixed_total_bits / 8 / 1024,
        'compression_ratio': compression,
        'savings_pct': (1 - mixed_total_bits / fp32_total_bits) * 100,
        'details': details,
    }


# =========================================================
# PRINT SCHEDULER RESULTS
# =========================================================

def print_scheduler_results(budget, precision_map,
                            est_drop, actual_acc,
                            fp32_acc, memory):
    """Print formatted scheduler output for one budget."""

    print(f"\n{'='*70}")
    print(f"  BUDGET: {budget:.1f}% max accuracy drop")
    print(f"{'='*70}")

    print(f"\n  {'Layer':<45} {'Precision':>10}")
    print(f"  {'-'*55}")

    for name, prec in precision_map.items():
        print(f"  {name:<45} {prec:>10}")

    actual_drop = fp32_acc - actual_acc

    print(f"\n  Estimated Drop:    {est_drop:.2f}%")
    print(f"  Actual Drop:       {actual_drop:.2f}%")
    print(f"  Actual Accuracy:   {actual_acc:.2f}%")
    print(f"  Within Budget:     "
          f"{'YES' if actual_drop <= budget else 'NO'}")

    print(f"\n  FP32 Memory:       {memory['fp32_KB']:.2f} KB")
    print(f"  Mixed Memory:      {memory['mixed_KB']:.2f} KB")
    print(f"  Compression:       {memory['compression_ratio']:.2f}x")
    print(f"  Savings:           {memory['savings_pct']:.1f}%")


# =========================================================
# VALIDATED GREEDY SCHEDULER (GUARANTEED WITHIN BUDGET)
# =========================================================

def validated_greedy_scheduler(model, sensitivity_results,
                                test_loader, device,
                                accuracy_budget):
    """
    Validated greedy scheduler with actual inference feedback.

    Unlike the basic greedy scheduler which estimates drops
    additively, this version ACTUALLY evaluates accuracy after
    each layer assignment. This guarantees the final precision
    map stays within the accuracy budget.

    Algorithm:
    1. Start with all FP32 (guaranteed 0% drop)
    2. Sort layers by acc_drop (least sensitive first)
    3. For each layer, try NVFP4:
       - Apply precision map and evaluate on test set
       - If actual drop <= budget: keep NVFP4
       - If not, try BF16:
         - If actual drop <= budget: keep BF16
         - If not: keep FP32 and move to next layer
    4. Continue until all layers are processed

    Returns:
        precision_map: dict {layer_name: precision}
        history: list of dicts tracking each step
    """

    # Sort: least sensitive first (most compressible)
    sorted_layers = sorted(
        sensitivity_results,
        key=lambda x: x['acc_drop']
    )

    # Start with all FP32
    precision_map = {
        r['layer']: 'FP32' for r in sensitivity_results
    }

    # Get FP32 baseline
    fp32_acc, _ = evaluate_model(model, test_loader, device)

    history = []
    current_acc = fp32_acc

    print(f"\n  FP32 Baseline: {fp32_acc:.2f}%")
    print(f"  Budget: {accuracy_budget:.1f}% max drop")
    print(f"  Processing {len(sorted_layers)} layers...\n")

    for i, layer_info in enumerate(sorted_layers):

        name = layer_info['layer']
        short = name.replace('.weight', '.w')

        # --- Try NVFP4 ---
        precision_map[name] = 'NVFP4'
        mixed_model = apply_precision_map(model, precision_map)
        nvfp4_acc, _ = evaluate_model(
            mixed_model, test_loader, device
        )
        nvfp4_drop = fp32_acc - nvfp4_acc

        if nvfp4_drop <= accuracy_budget:
            current_acc = nvfp4_acc
            history.append({
                'step': i + 1,
                'layer': name,
                'tried': 'NVFP4',
                'accepted': 'NVFP4',
                'accuracy': nvfp4_acc,
                'drop': nvfp4_drop,
            })
            print(f"  [{i+1:>2}/{len(sorted_layers)}] "
                  f"{short:<40} -> NVFP4  "
                  f"(acc: {nvfp4_acc:.2f}%, "
                  f"drop: {nvfp4_drop:.2f}%)")
            continue

        # --- NVFP4 exceeded budget, try BF16 ---
        precision_map[name] = 'BF16'
        mixed_model = apply_precision_map(model, precision_map)
        bf16_acc, _ = evaluate_model(
            mixed_model, test_loader, device
        )
        bf16_drop = fp32_acc - bf16_acc

        if bf16_drop <= accuracy_budget:
            current_acc = bf16_acc
            history.append({
                'step': i + 1,
                'layer': name,
                'tried': 'BF16',
                'accepted': 'BF16',
                'accuracy': bf16_acc,
                'drop': bf16_drop,
            })
            print(f"  [{i+1:>2}/{len(sorted_layers)}] "
                  f"{short:<40} -> BF16   "
                  f"(acc: {bf16_acc:.2f}%, "
                  f"drop: {bf16_drop:.2f}%)")
            continue

        # --- Both exceeded budget, keep FP32 ---
        precision_map[name] = 'FP32'
        history.append({
            'step': i + 1,
            'layer': name,
            'tried': 'FP32',
            'accepted': 'FP32',
            'accuracy': current_acc,
            'drop': fp32_acc - current_acc,
        })
        print(f"  [{i+1:>2}/{len(sorted_layers)}] "
              f"{short:<40} -> FP32   "
              f"(budget limit reached)")

    return precision_map, history

