"""
===========================================================
NVFP4 QUANTIZATION UTILITIES (PyTorch Compatible)
===========================================================

Shared utility module for NVFP4 quantization used across
all sensitivity analysis scripts.

Provides:
1. FP4 encode/decode (sign-aware)
2. Block-wise quantization for PyTorch tensors
3. Metrics: cosine similarity, KL divergence

===========================================================
"""

import numpy as np
import torch
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


# =========================================================
# FP4 DECODER
# =========================================================

def decode_fp4(bits):

    return fp4_table[bits]


# =========================================================
# PYTORCH WEIGHT QUANTIZATION (SIMULATE NVFP4)
# =========================================================

def quantize_weight_nvfp4(weight_tensor, block_size=16):
    """
    Simulate NVFP4 quantization on a PyTorch weight tensor.

    Process:
    1. Flatten weights
    2. Split into blocks
    3. Per-block: scale -> normalize -> encode -> decode -> denormalize
    4. Reshape back to original dimensions

    Returns a new FP32 tensor with NVFP4-quantized values.
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

    quantized = torch.tensor(
        result.reshape(shape),
        dtype=weight_tensor.dtype
    )

    return quantized


# =========================================================
# CLONE MODEL AND QUANTIZE A SINGLE LAYER
# =========================================================

def quantize_single_layer(model, target_layer_name,
                          block_size=16):
    """
    Clone a model and quantize only the specified layer.
    """

    quantized_model = copy.deepcopy(model)

    for name, param in quantized_model.named_parameters():

        if name == target_layer_name:

            quantized_weights = quantize_weight_nvfp4(
                param.data, block_size
            )

            param.data = quantized_weights

    return quantized_model


# =========================================================
# CLONE MODEL AND QUANTIZE ALL LAYERS
# =========================================================

def quantize_all_layers(model, block_size=16):
    """
    Clone a model and quantize all weight parameters.
    """

    quantized_model = copy.deepcopy(model)

    for name, param in quantized_model.named_parameters():

        if 'weight' in name and param.dim() >= 2:

            quantized_weights = quantize_weight_nvfp4(
                param.data, block_size
            )

            param.data = quantized_weights

    return quantized_model


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy and collect output logits.
    """

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
# METRICS
# =========================================================

def compute_cosine_sim(logits_fp32, logits_quant):
    """
    Compute average cosine similarity between two sets
    of output logits.
    """

    cos_sim = F.cosine_similarity(
        logits_fp32, logits_quant, dim=1
    )

    return cos_sim.mean().item()


def compute_kl_divergence(logits_fp32, logits_quant):
    """
    Compute average KL divergence between softmax outputs.
    """

    p = F.log_softmax(logits_quant, dim=1)
    q = F.softmax(logits_fp32, dim=1)

    kl_div = F.kl_div(p, q, reduction='batchmean')

    return kl_div.item()


def compute_activation_mse(logits_fp32, logits_quant):
    """
    Compute MSE between output activations.
    """

    mse = F.mse_loss(logits_fp32, logits_quant)

    return mse.item()


# =========================================================
# PRECISION ASSIGNMENT
# =========================================================

def assign_precision(acc_drop, cos_sim):
    """
    Assign precision based on sensitivity thresholds.

    Rules:
        acc_drop < 1% and cos_sim > 0.99  -> NVFP4
        acc_drop < 3% and cos_sim > 0.95  -> BF16
        otherwise                          -> FP32
    """

    if acc_drop < 1.0 and cos_sim > 0.99:
        return "NVFP4"

    elif acc_drop < 3.0 and cos_sim > 0.95:
        return "BF16"

    else:
        return "FP32"


# =========================================================
# PRINT RESULTS TABLE
# =========================================================

def print_sensitivity_table(results, model_name):
    """
    Print formatted sensitivity analysis results.
    """

    print("\n" + "=" * 85)
    print(f"  LAYER SENSITIVITY ANALYSIS: {model_name}")
    print("=" * 85)

    header = (
        f"{'Layer':<25} "
        f"{'Acc%':>7} "
        f"{'Drop%':>7} "
        f"{'CosSim':>8} "
        f"{'KL-Div':>10} "
        f"{'MSE':>10} "
        f"{'Precision':>10}"
    )

    print(header)
    print("-" * 85)

    for r in results:

        row = (
            f"{r['layer']:<25} "
            f"{r['accuracy']:>7.2f} "
            f"{r['acc_drop']:>7.2f} "
            f"{r['cos_sim']:>8.4f} "
            f"{r['kl_div']:>10.4f} "
            f"{r['mse']:>10.4f} "
            f"{r['precision']:>10}"
        )

        print(row)

    print("=" * 85)
