"""
===========================================================
NVFP4 COMPLETE QUANTIZATION PIPELINE
===========================================================

This file implements the complete NVFP4 quantization
pipeline for neural network tensors.

Implemented Features:
---------------------
1. FP4 lookup table
2. Saturation handling
3. FP4 encoding
4. FP4 decoding
5. Block-wise scaling
6. Tensor normalization
7. Tensor quantization
8. Tensor reconstruction
9. Quantization error analysis
10. Mean Squared Error (MSE)
11. Cosine Similarity

Author: Siddhant Deore
===========================================================
"""

import numpy as np


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


# Reverse lookup table
value_to_fp4 = {v: k for k, v in fp4_table.items()}


# =========================================================
# FP4 RANGE
# =========================================================

FP4_MIN = -6.0
FP4_MAX = 6.0


# =========================================================
# SATURATION
# =========================================================

def saturate(x, min_val=FP4_MIN, max_val=FP4_MAX):
    """
    Saturate scalar value into FP4 range.
    """

    if x > max_val:
        return max_val

    if x < min_val:
        return min_val

    return x


# =========================================================
# FP4 ENCODER
# =========================================================

def encode_fp4(x):
    """
    Encode FP32 value into FP4 representation.
    """

    # Saturate value
    x = saturate(x)

    # Sign-aware search to avoid -0.0 ambiguity
    # In Python, -0.0 == 0.0 which corrupts the
    # reverse lookup table
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

    # Find nearest representable value
    encoded_bits = min(
        search_codes.keys(),
        key=lambda k: abs(search_codes[k] - x)
    )

    return encoded_bits


# =========================================================
# FP4 DECODER
# =========================================================

def decode_fp4(bits):
    """
    Decode FP4 bits into FP32 value.
    """

    if bits < 0 or bits > 15:
        raise ValueError(
            "FP4 value must be between 0 and 15"
        )

    return fp4_table[bits]


# =========================================================
# BLOCK-WISE SCALING
# =========================================================

def compute_scale(tensor):
    """
    Compute block-wise scaling factor.
    """

    tensor = np.array(tensor)

    scale = np.max(np.abs(tensor))

    return scale


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_tensor(tensor, scale):
    """
    Normalize tensor using scale factor.
    """

    if scale == 0:
        return tensor

    return tensor / scale


# =========================================================
# DENORMALIZATION
# =========================================================

def denormalize_tensor(tensor, scale):
    """
    Restore tensor using scale factor.
    """

    return tensor * scale


# =========================================================
# TENSOR QUANTIZATION
# =========================================================

def quantize_tensor_nvfp4(tensor):
    """
    Complete NVFP4 quantization pipeline.

    Pipeline:
    ----------
    FP32 Tensor
          ↓
    Compute Scale
          ↓
    Normalize
          ↓
    Saturate
          ↓
    FP4 Encode
    """

    tensor = np.array(tensor)

    # -----------------------------------------------------
    # Step 1: Compute scale
    # -----------------------------------------------------
    scale = compute_scale(tensor)

    # -----------------------------------------------------
    # Step 2: Normalize tensor
    # -----------------------------------------------------
    normalized_tensor = normalize_tensor(
        tensor,
        scale
    )

    # -----------------------------------------------------
    # Step 3: Encode tensor
    # -----------------------------------------------------
    encoded_tensor = []

    for value in normalized_tensor:

        encoded_bits = encode_fp4(value)

        encoded_tensor.append(encoded_bits)

    encoded_tensor = np.array(encoded_tensor)

    return encoded_tensor, scale


# =========================================================
# TENSOR DEQUANTIZATION
# =========================================================

def dequantize_tensor_nvfp4(encoded_tensor, scale):
    """
    Reconstruct FP32 tensor from NVFP4 tensor.

    Pipeline:
    ----------
    FP4 Bits
         ↓
    FP4 Decode
         ↓
    Denormalize
         ↓
    Reconstructed FP32 Tensor
    """

    decoded_tensor = []

    # -----------------------------------------------------
    # Step 1: Decode FP4 values
    # -----------------------------------------------------
    for bits in encoded_tensor:

        decoded_value = decode_fp4(bits)

        decoded_tensor.append(decoded_value)

    decoded_tensor = np.array(decoded_tensor)

    # -----------------------------------------------------
    # Step 2: Denormalize tensor
    # -----------------------------------------------------
    reconstructed_tensor = denormalize_tensor(
        decoded_tensor,
        scale
    )

    return reconstructed_tensor


# =========================================================
# QUANTIZATION ERROR
# =========================================================

def compute_quantization_error(original,
                               reconstructed):
    """
    Compute absolute quantization error.
    """

    error = np.abs(original - reconstructed)

    return error


# =========================================================
# MEAN SQUARED ERROR
# =========================================================

def compute_mse(original, reconstructed):
    """
    Compute Mean Squared Error.
    """

    mse = np.mean(
        (original - reconstructed) ** 2
    )

    return mse


# =========================================================
# COSINE SIMILARITY
# =========================================================

def compute_cosine_similarity(original,
                              reconstructed):
    """
    Compute cosine similarity.
    """

    dot_product = np.dot(original, reconstructed)

    norm_original = np.linalg.norm(original)

    norm_reconstructed = np.linalg.norm(
        reconstructed
    )

    cosine_similarity = (
        dot_product /
        (norm_original * norm_reconstructed)
    )

    return cosine_similarity


# =========================================================
# COMPLETE PIPELINE TEST
# =========================================================

input_tensor = np.array([
    1.2,
    -0.8,
    2.5,
    -1.7,
    3.9,
    -2.6
])


print("\n=================================================")
print("ORIGINAL FP32 TENSOR")
print("=================================================\n")

print(input_tensor)


# =========================================================
# QUANTIZATION
# =========================================================

encoded_tensor, scale = quantize_tensor_nvfp4(
    input_tensor
)


print("\n=================================================")
print("BLOCK SCALE")
print("=================================================\n")

print(scale)


print("\n=================================================")
print("ENCODED NVFP4 TENSOR")
print("=================================================\n")

for bits in encoded_tensor:

    print(format(bits, '04b'))


# =========================================================
# DEQUANTIZATION
# =========================================================

reconstructed_tensor = dequantize_tensor_nvfp4(
    encoded_tensor,
    scale
)


print("\n=================================================")
print("RECONSTRUCTED FP32 TENSOR")
print("=================================================\n")

print(reconstructed_tensor)


# =========================================================
# QUANTIZATION ERROR
# =========================================================

error = compute_quantization_error(
    input_tensor,
    reconstructed_tensor
)


print("\n=================================================")
print("QUANTIZATION ERROR")
print("=================================================\n")

print(error)


# =========================================================
# MSE
# =========================================================

mse = compute_mse(
    input_tensor,
    reconstructed_tensor
)


print("\n=================================================")
print("MEAN SQUARED ERROR")
print("=================================================\n")

print(mse)


# =========================================================
# COSINE SIMILARITY
# =========================================================

cosine_similarity = compute_cosine_similarity(
    input_tensor,
    reconstructed_tensor
)


print("\n=================================================")
print("COSINE SIMILARITY")
print("=================================================\n")

print(cosine_similarity)


# =========================================================
# MEMORY COMPARISON
# =========================================================

fp32_bits = len(input_tensor) * 32

scale_bits = 32  # One FP32 scale factor per block

nvfp4_bits = (
    len(encoded_tensor) * 4 + scale_bits
)

compression_ratio = (
    fp32_bits / nvfp4_bits
)


print("\n=================================================")
print("MEMORY ANALYSIS")
print("=================================================\n")

print("FP32 Memory (bits) :", fp32_bits)

print("NVFP4 Memory(bits) :", nvfp4_bits)

print("Compression Ratio  :", compression_ratio)