"""
===========================================================
NVFP4 ERROR DISTRIBUTION PLOTS
===========================================================

This file generates:

1. Quantization error histogram
2. Original vs Reconstructed scatter plot
3. Per-element error bar chart
4. Error distribution across random tensors

Author: Siddhant Deore
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# FP4 LOOKUP TABLE
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

    # Sign-aware search to avoid -0.0 ambiguity
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

    if bits < 0 or bits > 15:
        raise ValueError(
            "FP4 value must be between 0 and 15"
        )

    return fp4_table[bits]


# =========================================================
# BLOCK-WISE QUANTIZATION
# =========================================================

def compute_scale(tensor):

    return np.max(np.abs(tensor))


def quantize_tensor_nvfp4(tensor):

    tensor = np.array(tensor)
    scale = compute_scale(tensor)

    if scale == 0:
        normalized = tensor
    else:
        normalized = tensor / scale

    encoded = [encode_fp4(v) for v in normalized]

    return np.array(encoded), scale


def dequantize_tensor_nvfp4(encoded_tensor, scale):

    decoded = [decode_fp4(b) for b in encoded_tensor]

    return np.array(decoded) * scale


# =========================================================
# PLOT 1: QUANTIZATION ERROR HISTOGRAM
# =========================================================

np.random.seed(42)

# Generate 1000 random values in [-5, 5]
random_values = np.random.uniform(-5.0, 5.0, 1000)

# Quantize and dequantize each value
errors = []

for val in random_values:
    enc = encode_fp4(val)
    dec = decode_fp4(enc)
    errors.append(abs(val - dec))

errors = np.array(errors)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "NVFP4 Quantization Error Analysis",
    fontsize=16,
    fontweight='bold'
)


# ---------------------------------------------------------
# Plot 1: Error Histogram
# ---------------------------------------------------------

axes[0, 0].hist(
    errors,
    bins=50,
    color='steelblue',
    edgecolor='black',
    alpha=0.8
)

axes[0, 0].set_title("Quantization Error Distribution")
axes[0, 0].set_xlabel("Absolute Error")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(
    np.mean(errors),
    color='red',
    linestyle='--',
    label=f"Mean = {np.mean(errors):.3f}"
)
axes[0, 0].legend()


# ---------------------------------------------------------
# Plot 2: Original vs Reconstructed (Scatter)
# ---------------------------------------------------------

reconstructed_vals = []

for val in random_values:
    enc = encode_fp4(val)
    dec = decode_fp4(enc)
    reconstructed_vals.append(dec)

reconstructed_vals = np.array(reconstructed_vals)

axes[0, 1].scatter(
    random_values,
    reconstructed_vals,
    s=3,
    alpha=0.5,
    color='steelblue'
)

axes[0, 1].plot(
    [-5, 5], [-5, 5],
    'r--',
    linewidth=1,
    label="Ideal (y=x)"
)

axes[0, 1].set_title(
    "Original vs Quantized (No Block Scaling)"
)
axes[0, 1].set_xlabel("Original FP32 Value")
axes[0, 1].set_ylabel("Quantized FP4 Value")
axes[0, 1].legend()
axes[0, 1].set_xlim(-5, 5)
axes[0, 1].set_ylim(-5, 5)


# ---------------------------------------------------------
# Plot 3: Block-wise Quantization Error
# ---------------------------------------------------------

test_tensor = np.array([
    1.2, -0.8, 2.5, -1.7, 3.9, -2.6,
    0.3, -0.1, 1.8, -3.2
])

encoded, scale = quantize_tensor_nvfp4(test_tensor)
reconstructed = dequantize_tensor_nvfp4(encoded, scale)
per_element_error = np.abs(test_tensor - reconstructed)

x_indices = np.arange(len(test_tensor))

axes[1, 0].bar(
    x_indices,
    per_element_error,
    color='coral',
    edgecolor='black',
    alpha=0.8
)

axes[1, 0].set_title(
    f"Per-Element Error (Block Scale = {scale:.2f})"
)
axes[1, 0].set_xlabel("Element Index")
axes[1, 0].set_ylabel("Absolute Error")
axes[1, 0].set_xticks(x_indices)


# ---------------------------------------------------------
# Plot 4: MSE vs Block Size
# ---------------------------------------------------------

large_tensor = np.random.uniform(-5.0, 5.0, 128)

block_sizes = [4, 8, 16, 32, 64, 128]
mse_values = []

for bs in block_sizes:

    block_mses = []

    for i in range(0, len(large_tensor), bs):

        block = large_tensor[i:i + bs]

        enc, sc = quantize_tensor_nvfp4(block)
        rec = dequantize_tensor_nvfp4(enc, sc)

        block_mse = np.mean((block - rec) ** 2)
        block_mses.append(block_mse)

    mse_values.append(np.mean(block_mses))


axes[1, 1].plot(
    block_sizes,
    mse_values,
    'o-',
    color='darkgreen',
    linewidth=2,
    markersize=8
)

axes[1, 1].set_title("MSE vs Block Size")
axes[1, 1].set_xlabel("Block Size")
axes[1, 1].set_ylabel("Mean Squared Error")
axes[1, 1].set_xticks(block_sizes)


# ---------------------------------------------------------
# Save and Show
# ---------------------------------------------------------

plt.tight_layout()
plt.savefig("nvfp4_error_distribution.png", dpi=150)
plt.show()


print("\n=================================================")
print("ERROR STATISTICS")
print("=================================================\n")

print(f"Mean Error        : {np.mean(errors):.4f}")
print(f"Max Error         : {np.max(errors):.4f}")
print(f"Min Error         : {np.min(errors):.4f}")
print(f"Std Dev           : {np.std(errors):.4f}")
print(f"Median Error      : {np.median(errors):.4f}")
