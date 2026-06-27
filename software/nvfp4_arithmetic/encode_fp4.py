"""
===========================================================
NVFP4 ENCODING IMPLEMENTATION
===========================================================

Format Used:
-------------
E2M1-style custom NVFP4 format

4-bit representation:
[S][E1 E0][M]

This implementation includes:
1. FP4 lookup table
2. Saturation handling
3. Nearest-value quantization
4. Encoding function
5. Test examples

Author: Siddhant Deore
===========================================================
"""

import numpy as np


# =========================================================
# FP4 LOOKUP TABLE
# =========================================================

# Binary code -> FP4 value
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


# Reverse lookup:
# FP4 value -> Binary code
value_to_fp4 = {v: k for k, v in fp4_table.items()}


# =========================================================
# SATURATION FUNCTION
# =========================================================

def saturate(x, min_val=-6.0, max_val=6.0):
    """
    Saturate input value into representable FP4 range.
    """

    if x > max_val:
        return max_val

    if x < min_val:
        return min_val

    return x


# =========================================================
# ENCODING FUNCTION
# =========================================================

def encode_fp4(x):
    """
    Encode FP32 number into 4-bit NVFP4 representation.

    Steps:
    1. Saturate
    2. Find nearest representable FP4 value (sign-aware)
    3. Return encoded 4-bit binary
    """

    # -----------------------------------------------------
    # Step 1: Saturation
    # -----------------------------------------------------
    x = saturate(x)

    # -----------------------------------------------------
    # Step 2: Find nearest FP4 value (sign-aware)
    # Sign-aware search avoids -0.0 ambiguity in Python
    # where -0.0 == 0.0, which corrupts the reverse lookup
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Step 3: Get binary encoding
    # -----------------------------------------------------
    encoded_bits = min(
        search_codes.keys(),
        key=lambda k: abs(search_codes[k] - x)
    )

    return encoded_bits


# =========================================================
# PRETTY PRINT FUNCTION
# =========================================================

def print_encoded_result(x):
    """
    Print encoding result in readable form.
    """

    encoded = encode_fp4(x)

    print("Input Value        :", x)
    print("Encoded FP4 Bits   :", format(encoded, '04b'))
    print("Quantized Value    :", fp4_table[encoded])
    print("Quantization Error :", abs(x - fp4_table[encoded]))
    print("-" * 45)


# =========================================================
# TEST CASES
# =========================================================

test_values = [
    0.1,
    0.7,
    1.2,
    1.7,
    2.3,
    3.6,
    -0.9,
    -2.7,
    -5.0,   # saturation test
    10.0    # saturation test
]


print("\n=================================================")
print("NVFP4 ENCODING TEST RESULTS")
print("=================================================\n")

for val in test_values:
    print_encoded_result(val)