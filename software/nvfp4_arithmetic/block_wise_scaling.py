"""
===========================================================
NVFP4 BLOCK-WISE SCALING IMPLEMENTATION
===========================================================

This file implements:

1. Block-wise scaling
2. Tensor normalization
3. Scale factor computation
4. Saturation handling
5. FP4 encoding
6. FP4 decoding

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
# SATURATION
# =========================================================

def saturate(x, min_val=-6.0, max_val=6.0):

    if x > max_val:
        return max_val

    if x < min_val:
        return min_val

    return x


# =========================================================
# FP4 ENCODER
# =========================================================

def encode_fp4(x):

    # Saturate
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
# BLOCK-WISE SCALING
# =========================================================

def blockwise_scale(tensor):
    """
    Compute block-wise scale factor.

    Scale = maximum absolute value in tensor block
    """

    tensor = np.array(tensor)

    scale = np.max(np.abs(tensor))

    return scale


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_tensor(tensor, scale):
    """
    Normalize tensor using block scale.
    """

    if scale == 0:
        return tensor

    normalized = tensor / scale

    return normalized


# =========================================================
# BLOCK-WISE NVFP4 ENCODING
# =========================================================

def encode_block_nvfp4(tensor):
    """
    Complete block-wise NVFP4 encoding pipeline.

    Steps:
    1. Compute scale
    2. Normalize tensor
    3. Encode each value into FP4
    """

    tensor = np.array(tensor)

    # -----------------------------------------------------
    # Step 1: Compute scale
    # -----------------------------------------------------
    scale = blockwise_scale(tensor)

    # -----------------------------------------------------
    # Step 2: Normalize tensor
    # -----------------------------------------------------
    normalized_tensor = normalize_tensor(tensor, scale)

    # -----------------------------------------------------
    # Step 3: Encode values
    # -----------------------------------------------------
    encoded_tensor = []

    for value in normalized_tensor:

        encoded_bits = encode_fp4(value)

        encoded_tensor.append(encoded_bits)

    return encoded_tensor, scale


# =========================================================
# BLOCK-WISE NVFP4 DECODING
# =========================================================

def decode_block_nvfp4(encoded_tensor, scale):
    """
    Decode NVFP4 tensor back to FP32 values.

    Steps:
    1. Decode FP4 values
    2. Multiply by scale
    """

    decoded_tensor = []

    for bits in encoded_tensor:

        decoded_value = decode_fp4(bits)

        reconstructed_value = decoded_value * scale

        decoded_tensor.append(reconstructed_value)

    return np.array(decoded_tensor)


# =========================================================
# TEST EXAMPLE
# =========================================================

input_tensor = np.array([
    1.2,
    -0.8,
    2.5,
    -1.7
])


print("\n=================================================")
print("ORIGINAL TENSOR")
print("=================================================\n")

print(input_tensor)


# ---------------------------------------------------------
# Encode
# ---------------------------------------------------------

encoded_tensor, scale = encode_block_nvfp4(input_tensor)


print("\n=================================================")
print("BLOCK-WISE SCALING")
print("=================================================\n")

print("Scale Factor:", scale)


print("\n=================================================")
print("ENCODED NVFP4 VALUES")
print("=================================================\n")

for bits in encoded_tensor:

    print(format(bits, '04b'))


# ---------------------------------------------------------
# Decode
# ---------------------------------------------------------

decoded_tensor = decode_block_nvfp4(
    encoded_tensor,
    scale
)


print("\n=================================================")
print("RECONSTRUCTED TENSOR")
print("=================================================\n")

print(decoded_tensor)


# ---------------------------------------------------------
# Quantization Error
# ---------------------------------------------------------

error = np.abs(input_tensor - decoded_tensor)

print("\n=================================================")
print("QUANTIZATION ERROR")
print("=================================================\n")

print(error)


# ---------------------------------------------------------
# Mean Squared Error
# ---------------------------------------------------------

mse = np.mean((input_tensor - decoded_tensor) ** 2)

print("\n=================================================")
print("MEAN SQUARED ERROR")
print("=================================================\n")

print(mse)