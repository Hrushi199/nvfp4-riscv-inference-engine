"""
===========================================================
NVFP4 SATURATION HANDLING IMPLEMENTATION
===========================================================

This file implements:

1. Saturation handling
2. FP4 range clipping
3. Overflow protection
4. Underflow protection
5. Saturation-aware encoding

Author: Siddhant Deore
===========================================================
"""

import numpy as np


# =========================================================
# FP4 REPRESENTABLE RANGE
# =========================================================

FP4_MIN = -6.0
FP4_MAX =  6.0


# =========================================================
# SATURATION FUNCTION
# =========================================================

def saturate(x, min_val=FP4_MIN, max_val=FP4_MAX):
    """
    Saturate a single FP32 value into FP4 range.

    If value exceeds representable range:
        x > max_val --> max_val
        x < min_val --> min_val

    Otherwise:
        return x unchanged
    """

    # -----------------------------------------------------
    # Overflow condition
    # -----------------------------------------------------
    if x > max_val:
        return max_val

    # -----------------------------------------------------
    # Underflow condition
    # -----------------------------------------------------
    if x < min_val:
        return min_val

    # -----------------------------------------------------
    # Value within range
    # -----------------------------------------------------
    return x


# =========================================================
# VECTOR/TENSOR SATURATION
# =========================================================

def saturate_tensor(tensor,
                    min_val=FP4_MIN,
                    max_val=FP4_MAX):
    """
    Saturate an entire tensor/vector.
    """

    tensor = np.array(tensor)

    saturated_tensor = np.clip(
        tensor,
        min_val,
        max_val
    )

    return saturated_tensor


# =========================================================
# SATURATION STATUS CHECK
# =========================================================

def check_saturation(x,
                     min_val=FP4_MIN,
                     max_val=FP4_MAX):
    """
    Check whether saturation occurred.
    """

    if x > max_val:
        return "OVERFLOW"

    elif x < min_val:
        return "UNDERFLOW"

    else:
        return "WITHIN RANGE"


# =========================================================
# DETAILED SATURATION REPORT
# =========================================================

def print_saturation_result(x):

    saturated_value = saturate(x)

    status = check_saturation(x)

    print("Original Value   :", x)
    print("Saturated Value  :", saturated_value)
    print("Status           :", status)
    print("-" * 45)


# =========================================================
# TEST CASES
# =========================================================

test_values = [
    -10.0,
    -5.2,
    -4.0,
    -2.5,
    0.0,
    1.7,
    3.9,
    4.0,
    5.5,
    12.0
]


print("\n=================================================")
print("NVFP4 SATURATION TEST")
print("=================================================\n")

for val in test_values:

    print_saturation_result(val)


# =========================================================
# TENSOR SATURATION TEST
# =========================================================

input_tensor = np.array([
    -9.0,
    -4.5,
    -1.2,
    0.5,
    2.3,
    4.8,
    10.0
])


print("\n=================================================")
print("ORIGINAL TENSOR")
print("=================================================\n")

print(input_tensor)


saturated_tensor = saturate_tensor(input_tensor)


print("\n=================================================")
print("SATURATED TENSOR")
print("=================================================\n")

print(saturated_tensor)


# =========================================================
# SATURATION ERROR
# =========================================================

error = np.abs(input_tensor - saturated_tensor)


print("\n=================================================")
print("SATURATION ERROR")
print("=================================================\n")

print(error)