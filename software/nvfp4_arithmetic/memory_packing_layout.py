"""
===========================================================
NVFP4 MEMORY PACKING LAYOUT
===========================================================

This file implements the memory layout design for packed
NVFP4 tensors, defining how data is stored at the byte
level for hardware consumption.

Packing Format:
--------------
- Two 4-bit NVFP4 values are packed per byte
  [High Nibble | Low Nibble]

- Scale factors (FP32, 32-bit) are stored at the
  start of each block

Block Memory Layout:
--------------------
For a block of N elements:

  Byte 0-3  : Scale factor (FP32, little-endian)
  Byte 4    : [Element 0 (4-bit) | Element 1 (4-bit)]
  Byte 5    : [Element 2 (4-bit) | Element 3 (4-bit)]
  ...
  Byte 4+N/2: [Element N-2 (4-bit) | Element N-1 (4-bit)]

This layout is designed for direct hardware consumption
by the NVFP4 Dot-Product Accelerator (Week 5) and
NVFP4.LOAD/NVFP4.STORE ISA instructions (Week 6).

Author: Siddhant Deore
===========================================================
"""

import numpy as np
import struct


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
# PACKING: TWO FP4 VALUES INTO ONE BYTE
# =========================================================

def pack_two_fp4(high_val, low_val):
    """
    Pack two 4-bit FP4 codes into a single byte.

    Layout:
        [high_val (bits 7-4) | low_val (bits 3-0)]

    Input:
        high_val -> integer (0 to 15)
        low_val  -> integer (0 to 15)

    Output:
        packed byte (0 to 255)
    """

    packed_byte = ((high_val & 0x0F) << 4) | \
                  (low_val & 0x0F)

    return packed_byte


# =========================================================
# UNPACKING: ONE BYTE INTO TWO FP4 VALUES
# =========================================================

def unpack_two_fp4(packed_byte):
    """
    Unpack a single byte into two 4-bit FP4 codes.

    Input:
        packed_byte -> integer (0 to 255)

    Output:
        (high_val, low_val) tuple
    """

    high_val = (packed_byte >> 4) & 0x0F
    low_val = packed_byte & 0x0F

    return high_val, low_val


# =========================================================
# BLOCK PACKING: FULL TENSOR TO PACKED BYTES
# =========================================================

def pack_block(tensor):
    """
    Pack an entire tensor block into the NVFP4 memory
    format.

    Memory Layout:
    ---------------
    Bytes 0-3 : Scale factor (FP32, little-endian)
    Bytes 4+  : Packed FP4 pairs (2 values per byte)

    If tensor has odd length, the last byte's low nibble
    is padded with zero (0b0000).

    Returns:
        packed_bytes -> bytearray
    """

    tensor = np.array(tensor, dtype=np.float32)

    # -------------------------------------------------
    # Step 1: Compute block scale factor
    # -------------------------------------------------
    scale = float(np.max(np.abs(tensor)))

    # -------------------------------------------------
    # Step 2: Normalize tensor
    # -------------------------------------------------
    if scale == 0:
        normalized = tensor
    else:
        normalized = tensor / scale

    # -------------------------------------------------
    # Step 3: Encode each value to FP4
    # -------------------------------------------------
    encoded = [encode_fp4(v) for v in normalized]

    # -------------------------------------------------
    # Step 4: Pack scale factor as 4 bytes (FP32 LE)
    # -------------------------------------------------
    scale_bytes = struct.pack('<f', scale)

    # -------------------------------------------------
    # Step 5: Pack FP4 pairs into bytes
    # -------------------------------------------------
    packed_data = bytearray(scale_bytes)

    for i in range(0, len(encoded), 2):

        high = encoded[i]

        # Pad with zero if odd number of elements
        if i + 1 < len(encoded):
            low = encoded[i + 1]
        else:
            low = 0b0000

        packed_data.append(
            pack_two_fp4(high, low)
        )

    return packed_data, scale, encoded


# =========================================================
# BLOCK UNPACKING: PACKED BYTES TO TENSOR
# =========================================================

def unpack_block(packed_data, num_elements):
    """
    Unpack NVFP4 memory format back to FP32 tensor.

    Input:
        packed_data  -> bytearray
        num_elements -> original tensor length

    Returns:
        reconstructed FP32 tensor
    """

    # -------------------------------------------------
    # Step 1: Extract scale factor (first 4 bytes)
    # -------------------------------------------------
    scale = struct.unpack('<f', packed_data[0:4])[0]

    # -------------------------------------------------
    # Step 2: Unpack FP4 pairs
    # -------------------------------------------------
    decoded_values = []

    for byte in packed_data[4:]:

        high, low = unpack_two_fp4(byte)

        decoded_values.append(decode_fp4(high))
        decoded_values.append(decode_fp4(low))

    # Trim to original length (remove padding)
    decoded_values = decoded_values[:num_elements]

    # -------------------------------------------------
    # Step 3: Denormalize
    # -------------------------------------------------
    reconstructed = np.array(decoded_values) * scale

    return reconstructed


# =========================================================
# MEMORY ANALYSIS
# =========================================================

def memory_analysis(num_elements):
    """
    Compare FP32 vs packed NVFP4 memory usage.
    """

    fp32_bytes = num_elements * 4
    fp32_bits = fp32_bytes * 8

    # NVFP4: 4 bytes scale + ceil(N/2) bytes data
    import math
    nvfp4_data_bytes = math.ceil(num_elements / 2)
    nvfp4_scale_bytes = 4
    nvfp4_total_bytes = nvfp4_data_bytes + nvfp4_scale_bytes
    nvfp4_bits = nvfp4_total_bytes * 8

    compression = fp32_bits / nvfp4_bits

    return {
        'fp32_bytes': fp32_bytes,
        'fp32_bits': fp32_bits,
        'nvfp4_data_bytes': nvfp4_data_bytes,
        'nvfp4_scale_bytes': nvfp4_scale_bytes,
        'nvfp4_total_bytes': nvfp4_total_bytes,
        'nvfp4_bits': nvfp4_bits,
        'compression_ratio': compression,
    }


# =========================================================
# TEST: FULL PACK/UNPACK PIPELINE
# =========================================================

input_tensor = np.array([
    1.2, -0.8, 2.5, -1.7, 3.9, -2.6
])


print("\n=================================================")
print("ORIGINAL FP32 TENSOR")
print("=================================================\n")

print(input_tensor)


# ---------------------------------------------------------
# Pack
# ---------------------------------------------------------

packed_data, scale, encoded = pack_block(input_tensor)


print("\n=================================================")
print("BLOCK SCALE FACTOR")
print("=================================================\n")

print(f"Scale: {scale}")


print("\n=================================================")
print("ENCODED FP4 VALUES")
print("=================================================\n")

for i, bits in enumerate(encoded):
    print(f"  Element {i}: {format(bits, '04b')} "
          f"-> {decode_fp4(bits)}")


print("\n=================================================")
print("PACKED MEMORY LAYOUT (BYTE LEVEL)")
print("=================================================\n")

print(f"  Total packed size: {len(packed_data)} bytes\n")

# Scale factor bytes
print("  Bytes 0-3 (Scale Factor FP32):")
for i in range(4):
    print(f"    Byte {i}: 0x{packed_data[i]:02X} "
          f"({format(packed_data[i], '08b')})")

# Data bytes
print(f"\n  Bytes 4+ (Packed FP4 Pairs):")
for i in range(4, len(packed_data)):

    high, low = unpack_two_fp4(packed_data[i])

    pair_idx = (i - 4) * 2

    print(
        f"    Byte {i}: 0x{packed_data[i]:02X} "
        f"({format(packed_data[i], '08b')}) "
        f"-> Element {pair_idx}: {format(high, '04b')}, "
        f"Element {pair_idx + 1}: {format(low, '04b')}"
    )


# ---------------------------------------------------------
# Unpack
# ---------------------------------------------------------

reconstructed = unpack_block(
    packed_data,
    len(input_tensor)
)


print("\n=================================================")
print("RECONSTRUCTED FP32 TENSOR")
print("=================================================\n")

print(reconstructed)


# ---------------------------------------------------------
# Error
# ---------------------------------------------------------

error = np.abs(input_tensor - reconstructed)
mse = np.mean((input_tensor - reconstructed) ** 2)


print("\n=================================================")
print("QUANTIZATION ERROR")
print("=================================================\n")

print(f"Per-element error: {error}")
print(f"MSE:               {mse:.6f}")


# ---------------------------------------------------------
# Memory Analysis
# ---------------------------------------------------------

print("\n=================================================")
print("MEMORY ANALYSIS")
print("=================================================\n")

block_sizes = [6, 16, 32, 64, 128]

for bs in block_sizes:

    stats = memory_analysis(bs)

    print(
        f"  Block Size {bs:>3d}: "
        f"FP32={stats['fp32_bytes']:>4d}B, "
        f"NVFP4={stats['nvfp4_total_bytes']:>3d}B "
        f"(data={stats['nvfp4_data_bytes']}B + "
        f"scale={stats['nvfp4_scale_bytes']}B), "
        f"Compression={stats['compression_ratio']:.2f}x"
    )
