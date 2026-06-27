"""
===========================================================
GOLDEN REFERENCE TEST VECTOR GENERATOR
===========================================================

Generates test vectors for the NVFP4 accelerator RTL using
the same nvfp4_utils.py from the software phase. This ensures
bit-exact correspondence between Python and Verilog.

Output: test_vectors.hex (readable by Verilog $readmemh)
        test_vectors_readable.txt (human-readable version)

Usage:
    python generate_test_vectors.py

===========================================================
"""

import sys
import os
import struct
import random

# Add parent directory to import nvfp4_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nvfp4_utils import fp4_table, encode_fp4, decode_fp4


# =========================================================
# HELPERS
# =========================================================

def float_to_hex32(f):
    """Convert Python float to IEEE 754 FP32 hex string (8 chars)."""
    packed = struct.pack('>f', f)
    return packed.hex()


def float_to_uint32(f):
    """Convert Python float to IEEE 754 FP32 as unsigned int."""
    packed = struct.pack('>f', f)
    return struct.unpack('>I', packed)[0]


def pack_nvfp4_block(codes):
    """
    Pack 16 NVFP4 4-bit codes into a 64-bit integer.
    Element i occupies bits [i*4+3 : i*4] (little-endian nibble).
    """
    assert len(codes) == 16
    packed = 0
    for i, code in enumerate(codes):
        packed |= (code & 0xF) << (i * 4)
    return packed


def encode_block(values):
    """Encode 16 float values to NVFP4 codes."""
    return [encode_fp4(v) for v in values]


def compute_integer_dot(w_codes, a_codes):
    """
    Compute the integer dot product (x4) matching the RTL.
    
    The RTL decodes each NVFP4 code to (value x 2), multiplies,
    then sums. So the integer result = true_dot_product x 4.
    """
    # x2 integer lookup (matching nvfp4_decoder.v)
    x2_table = {
        0: 0,  1: 1,  2: 2,  3: 3,
        4: 4,  5: 6,  6: 8,  7: 12,
        8: 0,  9: -1, 10: -2, 11: -3,
        12: -4, 13: -6, 14: -8, 15: -12,
    }
    
    dot = 0
    for wc, ac in zip(w_codes, a_codes):
        dot += x2_table[wc] * x2_table[ac]
    return dot


def compute_expected_fp32(w_codes, a_codes, combined_scale_float):
    """
    Compute the expected FP32 result matching the RTL.
    
    RTL computes: int_to_float(dot_int) * combined_scale
    This function replicates that exactly.
    """
    dot_int = compute_integer_dot(w_codes, a_codes)
    
    if dot_int == 0 or combined_scale_float == 0.0:
        return 0.0
    
    # The RTL converts dot_int to float then multiplies
    return float(dot_int) * combined_scale_float


# =========================================================
# MAIN
# =========================================================

def main():
    random.seed(42)
    
    # All possible NVFP4 decoded values
    pos_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    neg_values = [-v for v in pos_values if v != 0.0]
    all_values = pos_values + neg_values
    
    test_vectors = []
    
    # --- Structured test vectors ---
    
    # Test: all zeros
    test_vectors.append({
        'name': 'All zeros',
        'w_vals': [0.0] * 16,
        'a_vals': [0.0] * 16,
        'scale_w': 1.0,
        'scale_a': 1.0,
    })
    
    # Test: all ones
    test_vectors.append({
        'name': 'All 1.0 x 1.0',
        'w_vals': [1.0] * 16,
        'a_vals': [1.0] * 16,
        'scale_w': 1.0,
        'scale_a': 1.0,
    })
    
    # Test: all max (6.0 x 6.0)
    test_vectors.append({
        'name': 'All 6.0 x 6.0 (max)',
        'w_vals': [6.0] * 16,
        'a_vals': [6.0] * 16,
        'scale_w': 1.0,
        'scale_a': 1.0,
    })
    
    # Test: mixed positive and negative
    test_vectors.append({
        'name': 'Mixed +/- pattern',
        'w_vals': [6.0, -3.0, 1.5, 0.5] + [0.0] * 12,
        'a_vals': [1.0, 2.0, 4.0, 6.0] + [0.0] * 12,
        'scale_w': 2.0,
        'scale_a': 0.5,
    })
    
    # Test: alternating signs
    test_vectors.append({
        'name': 'Alternating +6/-6',
        'w_vals': [6.0, -6.0] * 8,
        'a_vals': [1.0] * 16,
        'scale_w': 1.0,
        'scale_a': 1.0,
    })
    
    # --- 45 random test vectors ---
    for t in range(45):
        w_vals = [random.choice(all_values) for _ in range(16)]
        a_vals = [random.choice(all_values) for _ in range(16)]
        scale_w = random.uniform(0.1, 10.0)
        scale_a = random.uniform(0.1, 10.0)
        
        test_vectors.append({
            'name': f'Random #{t}',
            'w_vals': w_vals,
            'a_vals': a_vals,
            'scale_w': scale_w,
            'scale_a': scale_a,
        })
    
    # --- Process all vectors ---
    output_dir = os.path.dirname(__file__)
    hex_path = os.path.join(output_dir, 'test_vectors.hex')
    txt_path = os.path.join(output_dir, 'test_vectors_readable.txt')
    
    with open(hex_path, 'w') as fhex, open(txt_path, 'w') as ftxt:
        fhex.write("// NVFP4 Accelerator Golden Reference Test Vectors\n")
        fhex.write("// Format: packed_w(64bit) packed_a(64bit) "
                    "combined_scale(32bit) expected(32bit)\n")
        fhex.write(f"// Total: {len(test_vectors)} vectors\n")
        fhex.write(f"// Generated with SEED=42\n\n")
        
        ftxt.write("NVFP4 Accelerator Golden Reference Test Vectors\n")
        ftxt.write("=" * 70 + "\n\n")
        
        for i, tv in enumerate(test_vectors):
            w_codes = encode_block(tv['w_vals'])
            a_codes = encode_block(tv['a_vals'])
            
            packed_w = pack_nvfp4_block(w_codes)
            packed_a = pack_nvfp4_block(a_codes)
            
            combined_scale = (tv['scale_w'] * tv['scale_a']) / 4.0
            dot_int = compute_integer_dot(w_codes, a_codes)
            expected = compute_expected_fp32(w_codes, a_codes, combined_scale)
            
            combined_hex = float_to_hex32(combined_scale)
            expected_hex = float_to_hex32(expected)
            
            # Hex file line
            fhex.write(f"{packed_w:016x} {packed_a:016x} "
                       f"{combined_hex} {expected_hex}\n")
            
            # Human-readable
            ftxt.write(f"Vector {i}: {tv['name']}\n")
            ftxt.write(f"  Weights:    {tv['w_vals']}\n")
            ftxt.write(f"  Activations:{tv['a_vals']}\n")
            ftxt.write(f"  Scale W/A:  {tv['scale_w']:.4f} / {tv['scale_a']:.4f}\n")
            ftxt.write(f"  Combined:   {combined_scale:.6f} (0x{combined_hex})\n")
            ftxt.write(f"  Dot Int:    {dot_int}\n")
            ftxt.write(f"  Expected:   {expected:.6f} (0x{expected_hex})\n")
            ftxt.write(f"  Packed W:   0x{packed_w:016x}\n")
            ftxt.write(f"  Packed A:   0x{packed_a:016x}\n")
            ftxt.write("\n")
    
    print(f"Generated {len(test_vectors)} test vectors")
    print(f"  Hex file:      {hex_path}")
    print(f"  Readable file: {txt_path}")
    
    # Print first 5 for quick verification
    print(f"\nFirst 5 vectors:")
    for i, tv in enumerate(test_vectors[:5]):
        w_codes = encode_block(tv['w_vals'])
        a_codes = encode_block(tv['a_vals'])
        combined_scale = (tv['scale_w'] * tv['scale_a']) / 4.0
        dot_int = compute_integer_dot(w_codes, a_codes)
        expected = compute_expected_fp32(w_codes, a_codes, combined_scale)
        print(f"  [{i}] {tv['name']}: dot_int={dot_int}, "
              f"expected={expected:.4f}")


if __name__ == '__main__':
    main()
