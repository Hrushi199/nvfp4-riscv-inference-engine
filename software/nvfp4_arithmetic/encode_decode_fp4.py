"""
FP32 -> NVFP4 (Encode) and NVFP4 -> FP32 (Decode)

===========================================================
NVFP4 DECODING IMPLEMENTATION
===========================================================

This file implements:

1. FP4 lookup table
2. encode_fp4()
3. decode_fp4()
4. Saturation handling
5. Test examples

Format Used:
-------------
Custom E2M1-style NVFP4

4-bit representation:
[S][E1 E0][M]

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
    Encode FP32 value into 4-bit NVFP4 representation.
    """

    # ---------------------------------------------
    # Step 1: Saturation
    # ---------------------------------------------
    x = saturate(x)

    # ---------------------------------------------
    # Step 2: Find nearest representable FP4 value
    # Sign-aware to avoid -0.0 ambiguity
    # ---------------------------------------------
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

    # ---------------------------------------------
    # Step 3: Return 4-bit encoded value
    # ---------------------------------------------
    encoded_bits = min(
        search_codes.keys(),
        key=lambda k: abs(search_codes[k] - x)
    )

    return encoded_bits


# =========================================================
# DECODING FUNCTION
# =========================================================

def decode_fp4(bits):
    """
    Decode 4-bit NVFP4 representation into FP32 value.

    Input:
        bits -> integer (0 to 15)

    Output:
        decoded floating-point value
    """

    # ---------------------------------------------
    # Input validation
    # ---------------------------------------------
    if bits < 0 or bits > 15:
        raise ValueError(
            "FP4 value must be a 4-bit number (0 to 15)"
        )

    # ---------------------------------------------
    # Lookup decoded value
    # ---------------------------------------------
    decoded_value = fp4_table[bits]

    return decoded_value


# =========================================================
# PRETTY PRINT FUNCTION
# =========================================================

def print_encode_decode_result(x):
    """
    Print complete encode-decode flow.
    """

    # ---------------------------------------------
    # Encode
    # ---------------------------------------------
    encoded = encode_fp4(x)

    # ---------------------------------------------
    # Decode
    # ---------------------------------------------
    decoded = decode_fp4(encoded)

    # ---------------------------------------------
    # Error calculation
    # ---------------------------------------------
    error = abs(x - decoded)

    # ---------------------------------------------
    # Display
    # ---------------------------------------------
    print("Original Value     :", x)
    print("Encoded FP4 Bits   :", format(encoded, '04b'))
    print("Decoded Value      :", decoded)
    print("Quantization Error :", error)
    print("-" * 50)


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
print("NVFP4 ENCODE-DECODE TEST RESULTS")
print("=================================================\n")

for val in test_values:
    print_encode_decode_result(val)


# =========================================================
# MANUAL DECODING TEST
# =========================================================

print("\n=================================================")
print("MANUAL DECODING TEST")
print("=================================================\n")

manual_test_bits = [
    0b0000,
    0b0011,
    0b0101,
    0b0111,
    0b1011,
    0b1111
]

for bits in manual_test_bits:

    decoded = decode_fp4(bits)

    print(
        "FP4 Bits:",
        format(bits, '04b'),
        " --> Decoded Value:",
        decoded
    )