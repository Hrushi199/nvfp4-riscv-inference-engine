# Generate multi-test hex including BOTH NVFP4 and BF16 tests

def encode_custom0(funct7, rs2, rs1, funct3, rd):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x0B

def encode_lui(rd, imm20):
    return (imm20 << 12) | (rd << 7) | 0x37

def encode_addi(rd, rs1, imm12):
    val = imm12 & 0xFFF
    return (val << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x13

def encode_sw(rs2, base, offset12):
    off = offset12 & 0xFFF
    return ((off >> 5) << 25) | (rs2 << 20) | (base << 15) | (2 << 12) | ((off & 0x1F) << 7) | 0x23

def load_reg(rd, val32):
    """Emit lui+addi to load a 32-bit immediate into rd"""
    instrs = []
    hi = (val32 >> 12) & 0xFFFFF
    lo = val32 & 0xFFF
    # If lo bit 11 is set, add 1 to hi to compensate addi sign-extension
    if lo & 0x800:
        hi = (hi + 1) & 0xFFFFF
    instrs.append(encode_lui(rd, hi))
    instrs.append(encode_addi(rd, rd, lo))
    return instrs

instructions = []

# ── x4 = 0x20000000 (Data RAM base) ────────────────────────────────
instructions += load_reg(4, 0x20000000)

# ════════════════════════════════════════════════════════════════════
# NVFP4 TESTS (7 tests) → data_mem[0..6]
# ════════════════════════════════════════════════════════════════════
nvfp4_tests = [
    # (weight_64, act_64, scale_32,   expected_f32_hex, label)
    (0x2222222222222222, 0x2222222222222222, 0x3E800000, 0x41800000, "NVFP4 T1: 16.0"),
    (0x0000000000000000, 0x0000000000000000, 0x3E800000, 0x00000000, "NVFP4 T2: 0.0"),
    (0x0000000000000007, 0x0000000000000007, 0x3E800000, 0x42100000, "NVFP4 T3: 36.0"),
    (0x00000000000013D7, 0x0000000000007642, 0x3E800000, 0x41100000, "NVFP4 T4: 9.0"),
    (0x000000000000000F, 0x0000000000000006, 0x3E800000, 0xC1C00000, "NVFP4 T5: -24.0"),
    (0x0000000000000001, 0x0000000000000001, 0x3E800000, 0x3E800000, "NVFP4 T6: 0.25"),
    (0x7777777777777777, 0x7777777777777777, 0x3E800000, 0x44100000, "NVFP4 T7: 576.0"),
]

for idx, (w, a, scale, exp, lbl) in enumerate(nvfp4_tests):
    w_lo = w & 0xFFFFFFFF
    w_hi = (w >> 32) & 0xFFFFFFFF
    a_lo = a & 0xFFFFFFFF
    a_hi = (a >> 32) & 0xFFFFFFFF

    instructions += load_reg(1, w_lo)   # x1 = weight_lo
    instructions += load_reg(2, w_hi)   # x2 = weight_hi
    instructions.append(encode_custom0(0, 2, 1, 0, 0))  # NVFP4.LOAD_W

    instructions += load_reg(1, a_lo)   # x1 = act_lo
    instructions += load_reg(2, a_hi)   # x2 = act_hi
    instructions.append(encode_custom0(0, 2, 1, 1, 0))  # NVFP4.LOAD_A

    instructions += load_reg(3, scale)  # x3 = scale
    instructions.append(encode_custom0(0, 0, 3, 2, 5))  # NVFP4.MAC → x5

    instructions.append(encode_sw(5, 4, idx * 4))       # sw x5, offset(x4)
    print(f"  slot {idx}: {lbl}")

# ════════════════════════════════════════════════════════════════════
# BF16 TESTS (4 tests) → data_mem[7..10]
# BF16 packed format: 4x BF16 in 64 bits = {bf16_3, bf16_2, bf16_1, bf16_0}
# BF16 is top 16 bits of IEEE 754 FP32
# ════════════════════════════════════════════════════════════════════
import struct

def fp32_to_bf16(f):
    """Return the BF16 representation of a Python float as a 16-bit int."""
    b = struct.pack('>f', f)
    return (b[0] << 8) | b[1]  # top 16 bits of FP32

def pack_4x_bf16(f0, f1, f2, f3):
    """Pack 4 floats as BF16 into a 64-bit integer: {b3,b2,b1,b0}"""
    b0 = fp32_to_bf16(f0)
    b1 = fp32_to_bf16(f1)
    b2 = fp32_to_bf16(f2)
    b3 = fp32_to_bf16(f3)
    return (b3 << 48) | (b2 << 32) | (b1 << 16) | b0

# BF16 test cases: (w_floats, a_floats, expected_label)
# dot(w, a) = w0*a0 + w1*a1 + w2*a2 + w3*a3
bf16_tests = [
    ([1.0,  1.0,  1.0,  1.0],  [1.0,  1.0,  1.0,  1.0],  "BF16 T1: 4.0"),
    ([2.0,  3.0,  4.0,  5.0],  [1.0,  1.0,  1.0,  1.0],  "BF16 T2: 14.0"),
    ([1.5,  2.5,  0.5,  1.0],  [2.0,  2.0,  2.0,  2.0],  "BF16 T3: 11.0"),
    ([0.0,  0.0,  0.0,  0.0],  [9.0,  9.0,  9.0,  9.0],  "BF16 T4: 0.0"),
    ([-1.0,-2.0, -3.0, -4.0],  [1.0,  1.0,  1.0,  1.0],  "BF16 T5: -10.0"),
    ([0.5,  0.25, 0.125,0.0625],[1.0,  1.0,  1.0,  1.0],  "BF16 T6: 0.9375"),
    ([3.0,  3.0,  3.0,  3.0],  [3.0,  3.0,  3.0,  3.0],  "BF16 T7: 36.0"),
]

bf16_base_slot = 7  # starts at data_mem[7], ends at data_mem[13]

for idx, (w_f, a_f, lbl) in enumerate(bf16_tests):
    w_packed = pack_4x_bf16(*w_f)
    a_packed = pack_4x_bf16(*a_f)
    slot = bf16_base_slot + idx

    w_lo = w_packed & 0xFFFFFFFF
    w_hi = (w_packed >> 32) & 0xFFFFFFFF
    a_lo = a_packed & 0xFFFFFFFF
    a_hi = (a_packed >> 32) & 0xFFFFFFFF

    instructions += load_reg(1, w_lo)   # x1 = bf16_weight_lo
    instructions += load_reg(2, w_hi)   # x2 = bf16_weight_hi
    instructions.append(encode_custom0(0, 2, 1, 4, 0))  # BF16.LOAD_W (funct3=4)

    instructions += load_reg(1, a_lo)   # x1 = bf16_act_lo
    instructions += load_reg(2, a_hi)   # x2 = bf16_act_hi
    instructions.append(encode_custom0(0, 2, 1, 5, 0))  # BF16.LOAD_A (funct3=5)

    instructions.append(encode_custom0(0, 0, 0, 6, 5))  # BF16.MAC → x5 (funct3=6)

    instructions.append(encode_sw(5, 4, slot * 4))      # sw x5, offset(x4)
    
    # Compute expected for display
    expected = sum(w_f[i] * a_f[i] for i in range(4))
    print(f"  slot {slot}: {lbl}  expected={expected}")

# Done flag at data_mem[11]
done_slot = 14  # data_mem[14] = done flag (after 7 NVFP4 + 7 BF16)
instructions += load_reg(6, 1)                          # x6 = 1
instructions.append(encode_sw(6, 4, done_slot * 4))    # sw x6, offset(x4)

# Halt
instructions.append(0x0000006F)  # jal x0, 0

# Write hex
outpath = "c:/Users/hrush/Desktop/SRIP 2026/final/test_nvfp4.hex"
with open(outpath, "w") as f:
    f.write("@00000000\n")
    for instr in instructions:
        f.write(f"{instr:08X}\n")

print(f"\nGenerated {len(instructions)} instructions -> {outpath}")
print("NVFP4 results: data_mem[0..6]")
print("BF16  results: data_mem[7..13]")
print("Done flag:     data_mem[14]")

# Also print BF16 expected hex values for testbench
import struct
print("\nBF16 expected hex values:")
for idx, (w_f, a_f, lbl) in enumerate(bf16_tests):
    expected = sum(w_f[i] * a_f[i] for i in range(4))
    b = struct.pack('>f', expected)
    hexval = int.from_bytes(b, 'big')
    print(f"  {lbl}: 0x{hexval:08X}")
