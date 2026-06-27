# Mixed-Precision RISC-V Custom ISA Extension
## NVFP4 + BF16 Accelerator Integration on CV32E40P

---

## Overview

This project extends the **CV32E40P** open-source RISC-V processor core with **7 custom instructions** that natively dispatch mixed-precision neural network dot-product operations to on-chip hardware accelerators — without any software emulation or memory-mapped I/O overhead.

Two precision formats are supported:
- **NVFP4** — NVIDIA's 4-bit floating-point format with block scaling, used for computation-intensive layers
- **BF16** — Google's Brain Float 16 format (16-bit, same exponent range as FP32), used as a high-accuracy fallback for sensitive layers

The precision assignment per layer is determined upstream by a **layer sensitivity analysis** (using MSE/cosine similarity) and a **mixed-precision scheduler**

---

## Repository Structure

```
final/
├── bf16_mac_unit.sv          # BF16 dot-product hardware unit
├── nvfp4_apu_adapter.sv      # Mixed-precision APU adapter (routes to NVFP4 or BF16)
├── nvfp4_accelerator_top.sv  # NVFP4 accelerator top + sub-modules
├── nvfp4_soc_top.sv          # Full SoC (CPU + Instruction RAM + Data RAM)
├── cv32e40p_nvfp4_top.sv     # CV32E40P core wrapper (FPU replaced by APU adapter)
├── tb_nvfp4_full_soc.sv      # Full SoC testbench (14 tests: 7 NVFP4 + 7 BF16)
├── test_nvfp4.hex            # Assembled test program (188 instructions)
├── generate_multi_test_hex.py # Script to regenerate test_nvfp4.hex
│
├── cv32e40p_rtl/             # CV32E40P RTL (modified decoder included)
├── cv32e40p_include/         # CV32E40P package files
└── cv32e40p_vendor/          # cv32e40p_sim_clock_gate.sv (simulation model)
```

---

## Architecture

```
                ┌─────────────────────────────────────────┐
                │            nvfp4_soc_top                │
                │                                          │
                │  ┌──────────────────────────────────┐   │
                │  │     cv32e40p_nvfp4_top           │   │
                │  │                                   │   │
                │  │  ┌─────────────┐  APU interface  │   │
                │  │  │ cv32e40p    │◄────────────────►│   │
                │  │  │ _core       │                  │   │
                │  │  │ (FPU=1,     │  ┌─────────────┐│   │
                │  │  │  ZFINX=1)   │  │nvfp4_apu    ││   │
                │  │  └─────────────┘  │_adapter     ││   │
                │  │                   │             ││   │
                │  │                   │ ┌──────────┐││   │
                │  │                   │ │ nvfp4_   │││   │
                │  │                   │ │accelera- │││   │
                │  │                   │ │tor_top   │││   │
                │  │                   │ └──────────┘││   │
                │  │                   │ ┌──────────┐││   │
                │  │                   │ │bf16_mac  │││   │
                │  │                   │ │_unit     │││   │
                │  │                   │ └──────────┘││   │
                │  │                   └─────────────┘│   │
                │  └──────────────────────────────────┘   │
                │                                          │
                │  Instruction RAM    Data RAM             │
                │  (loaded from .hex) (results verified)  │
                └─────────────────────────────────────────┘
```

---

## Custom ISA Extension

### Encoding Scheme

All custom instructions use **RISC-V Custom-0 opcode** (`7'b0001011` = `0x0B`), which is a reserved opcode space in the RISC-V specification. The `funct3` field (bits `[14:12]`) selects the operation:

```
 31      25 24   20 19   15 14  12 11    7 6      0
 ┌─────────┬───────┬───────┬──────┬───────┬────────┐
 │ funct7  │  rs2  │  rs1  │funct3│  rd   │0001011 │
 └─────────┴───────┴───────┴──────┴───────┴────────┘
   7 bits    5 bits  5 bits  3 bits  5 bits   7 bits
```

---

### NVFP4 Instructions (`funct3 = 000..011`)

These instructions operate on packed NVFP4 data (16 values of 4-bit float packed into two 32-bit registers).

#### `NVFP4.LOAD_W rd, rs1, rs2` — funct3=000, APU op=`6'b000001`

Loads a packed 64-bit NVFP4 weight vector into the accelerator's weight register.

```
Encoding: funct7=0000000 | rs2 | rs1 | 000 | rd | 0001011
Operands: rs1 = weight_packed[31:0]   (lower 8 weights)
          rs2 = weight_packed[63:32]  (upper 8 weights)
Latency:  1 cycle
```

#### `NVFP4.LOAD_A rd, rs1, rs2` — funct3=001, APU op=`6'b000010`

Loads a packed 64-bit NVFP4 activation vector into the accelerator's activation register.

```
Encoding: funct7=0000000 | rs2 | rs1 | 001 | rd | 0001011
Operands: rs1 = act_packed[31:0]
          rs2 = act_packed[63:32]
Latency:  1 cycle
```

#### `NVFP4.MAC rd, rs1` — funct3=010, APU op=`6'b000100`

Sets the combined FP32 scale factor and triggers the dot-product computation. The CPU pipeline stalls until `apu_rvalid` is asserted by the adapter. The FP32 result is written back to `rd`.

```
Encoding: funct7=0000000 | 00000 | rs1 | 010 | rd | 0001011
Operands: rs1 = combined_scale (FP32, IEEE 754)
Result:   rd  = dot_product(weights, acts) × scale  [FP32]
Latency:  multi-cycle (waits for NVFP4 accelerator FSM)
```

#### `NVFP4.STORE rd` — funct3=011, APU op=`6'b001000`

Returns the last stored result from the accelerator's result register to `rd`. Useful if the result needs to be read again without recomputation.

```
Encoding: funct7=0000000 | 00000 | 00000 | 011 | rd | 0001011
Result:   rd = stored_result_reg [FP32]
Latency:  1 cycle
```

---

### BF16 Instructions (`funct3 = 100..110`)

These instructions operate on packed BF16 data (4 values of BF16 packed into two 32-bit registers). BF16 is identical to the upper 16 bits of IEEE 754 FP32 (1 sign + 8 exponent + 7 mantissa bits), so conversion to FP32 is done by appending `16'h0000`.

#### `BF16.LOAD_W rd, rs1, rs2` — funct3=100, APU op=`6'b010001`

Loads a packed 64-bit BF16 weight vector (4 × BF16) into the BF16 unit's weight register.

```
Encoding: funct7=0000000 | rs2 | rs1 | 100 | rd | 0001011
Operands: rs1 = bf16_weight_packed[31:0]   ({w1,w0} as BF16)
          rs2 = bf16_weight_packed[63:32]  ({w3,w2} as BF16)
Latency:  1 cycle
```

#### `BF16.LOAD_A rd, rs1, rs2` — funct3=101, APU op=`6'b010010`

Loads a packed 64-bit BF16 activation vector into the BF16 unit's activation register.

```
Encoding: funct7=0000000 | rs2 | rs1 | 101 | rd | 0001011
Operands: rs1 = bf16_act_packed[31:0]
          rs2 = bf16_act_packed[63:32]
Latency:  1 cycle
```

#### `BF16.MAC rd` — funct3=110, APU op=`6'b010100`

Triggers the BF16 dot-product computation and writes the FP32 result to `rd`. The CPU pipeline stalls until `apu_rvalid` is returned.

```
Encoding: funct7=0000000 | 00000 | 00000 | 110 | rd | 0001011
Result:   rd = w0×a0 + w1×a1 + w2×a2 + w3×a3  [FP32]
Latency:  1 compute cycle (registered, waits for rvalid)
```

---

### Instruction Summary Table

| Mnemonic        | funct3 | APU Op     | rs1        | rs2        | rd     | Latency    |
|:----------------|:------:|:----------:|:----------:|:----------:|:------:|:----------:|
| `NVFP4.LOAD_W`  | `000`  | `6'b000001`| weight_lo  | weight_hi  | (none) | 1 cycle    |
| `NVFP4.LOAD_A`  | `001`  | `6'b000010`| act_lo     | act_hi     | (none) | 1 cycle    |
| `NVFP4.MAC`     | `010`  | `6'b000100`| scale      | —          | result | multi-cycle|
| `NVFP4.STORE`   | `011`  | `6'b001000`| —          | —          | result | 1 cycle    |
| `BF16.LOAD_W`   | `100`  | `6'b010001`| weight_lo  | weight_hi  | (none) | 1 cycle    |
| `BF16.LOAD_A`   | `101`  | `6'b010010`| act_lo     | act_hi     | (none) | 1 cycle    |
| `BF16.MAC`      | `110`  | `6'b010100`| —          | —          | result | multi-cycle|

---

## How It Works — Step by Step

### Step 1: CV32E40P Decoder Modification (`cv32e40p_decoder.sv`)

The decoder is the part of the CPU that reads each 32-bit instruction and decides what functional unit executes it. We added our custom instruction handling inside the existing `OPCODE_CUSTOM_0` case block.

**Location in file:** Lines ~1626–1695 in `cv32e40p_decoder.sv`

```systemverilog
OPCODE_CUSTOM_0: begin
  // ... existing PULP extensions (only active when COREV_PULP=1) ...
  end else begin   // COREV_PULP=0: our custom block
    if (FPU == 1) begin
      alu_en     = 1'b0;   // do NOT use the ALU
      apu_en     = 1'b1;   // USE the APU interface
      reg_fp_a_o = 1'b0;   // read from integer registers (ZFINX mode)

      unique case (instr_rdata_i[14:12])  // funct3 field
        3'b000: apu_op_o = 6'b000001; ... // NVFP4.LOAD_W
        3'b001: apu_op_o = 6'b000010; ... // NVFP4.LOAD_A
        3'b010: apu_op_o = 6'b000100; ... // NVFP4.MAC
        3'b011: apu_op_o = 6'b001000; ... // NVFP4.STORE
        3'b100: apu_op_o = 6'b010001; ... // BF16.LOAD_W
        3'b101: apu_op_o = 6'b010010; ... // BF16.LOAD_A
        3'b110: apu_op_o = 6'b010100; ... // BF16.MAC
      endcase
    end
  end
end
```

When the decoder sets `apu_en=1`, the CV32E40P execution stage automatically:
- Stalls the pipeline until `apu_rvalid` is returned
- Routes `apu_operands` (register values) to the APU port
- Writes the returned `apu_result` to the destination register `rd`

### Step 2: APU Adapter (`nvfp4_apu_adapter.sv`)

The adapter sits between the CPU's APU port and the two hardware units. It contains:
- **Two separate register banks** — one for NVFP4 state, one for BF16 state
- **A 3-state FSM** (`S_IDLE` → `S_COMPUTE` → `S_IDLE`) to manage multi-cycle operations
- **Instant grant** — always asserts `apu_gnt` in the same cycle as `apu_req` when idle

```
CPU APU port
    │ apu_req + apu_op + apu_operands
    ▼
┌─────────────────────────────────┐
│       nvfp4_apu_adapter         │
│                                  │
│  if op ∈ {NVFP4.*} → nvfp4_reg  │
│  if op ∈ {BF16.*}  → bf16_reg   │
│                                  │
│  on MAC: start accelerator       │
│  wait for done → apu_rvalid=1   │
└───────────┬──────────┬──────────┘
            │          │
    ┌───────▼──┐  ┌────▼────────┐
    │  nvfp4_  │  │  bf16_mac_  │
    │accelera- │  │  unit       │
    │tor_top   │  │  (FP32 MAC) │
    └──────────┘  └─────────────┘
```

### Step 3: NVFP4 Accelerator (`nvfp4_accelerator_top.sv`)

Processes 16 packed NVFP4 values using a 2-cycle pipeline:
1. **Cycle 1:** Decode NVFP4 → 4-bit integer values, run 16 parallel MAC units
2. **Cycle 2:** Scale the 13-bit integer dot-product result to FP32 using `nvfp4_scale_multiply`

Sub-modules: `nvfp4_decoder`, `nvfp4_multiplier`, `nvfp4_mac_unit`, `nvfp4_dot_product_16`, `nvfp4_extractor`, `nvfp4_scale_multiply`

### Step 4: BF16 MAC Unit (`bf16_mac_unit.sv`)

Processes 4 packed BF16 values:
1. **Unpack:** Extend 4× BF16 to FP32 by appending `16'h0000` to each (BF16 = top 16 bits of FP32)
2. **Multiply:** Compute 4 FP32 products using IEEE 754 mantissa multiplication
3. **Accumulate:** Add 4 products in a tree: `(p0+p1) + (p2+p3)`
4. **Return:** FP32 result in a single registered cycle

### Step 5: CV32E40P Core Config (`cv32e40p_nvfp4_top.sv`)

The core is instantiated with two key parameters:

| Parameter | Value | Reason |
|---|---|---|
| `FPU=1` | enabled | Activates the APU dispatch path in the execution stage |
| `ZFINX=1` | enabled | Operands read from integer register file (no separate FP register file needed) |
| `COREV_PULP=0` | disabled | Prevents PULP extension conflicts; enables our custom-0 block |

---

## Example Assembly Program

```asm
; Compute NVFP4 dot-product: weights=all 1.0, acts=all 1.0, scale=0.25 → result=16.0
lui   x1, 0x22222        ; x1 = 0x22222000
addi  x1, x1, 0x222      ; x1 = 0x22222222 (packed NVFP4 weights, lower 32 bits)
add   x2, x1, x0         ; x2 = x1          (upper 32 bits)
lui   x3, 0x3E800        ; x3 = 0x3E800000  (scale = 0.25 in FP32)
lui   x4, 0x20000        ; x4 = 0x20000000  (data RAM base)

.word 0x0020800B         ; NVFP4.LOAD_W x0, x1, x2
.word 0x0020900B         ; NVFP4.LOAD_A x0, x1, x2
.word 0x0001A28B         ; NVFP4.MAC    x5, x3       → x5 = 16.0
sw    x5, 0(x4)          ; store result to data RAM

; Compute BF16 dot-product: weights=[1.0,1.0,1.0,1.0], acts=[1.0,1.0,1.0,1.0] → 4.0
lui   x1, 0x3F80         ; x1 = 0x3F800000 → BF16: {1.0, 1.0}
...
.word 0x0020A00B         ; BF16.LOAD_W x0, x1, x2
.word 0x0020B00B         ; BF16.LOAD_A x0, x1, x2
.word 0x00006..B         ; BF16.MAC    x5            → x5 = 4.0
sw    x5, 4(x4)
```

---

## Verification Results

Simulation run on Xilinx Vivado 2022.2 (XSim behavioral simulation).

```
CPU finished in 270 cycles.

NVFP4 Results (data_mem[0..6]):
  Test 1 | 0x41800000 ( 16.00) |  16.00 | PASS
  Test 2 | 0x00000000 (  0.00) |   0.00 | PASS
  Test 3 | 0x42100000 ( 36.00) |  36.00 | PASS
  Test 4 | 0x41100000 (  9.00) |   9.00 | PASS
  Test 5 | 0xC1C00000 (-24.00) | -24.00 | PASS
  Test 6 | 0x3E800000 (  0.25) |   0.25 | PASS
  Test 7 | 0x44100000 (576.00) | 576.00 | PASS

BF16 Results (data_mem[7..13]):
  Test 1 | 0x40800000 (  4.00) |   4.00 | PASS
  Test 2 | 0x41600000 ( 14.00) |  14.00 | PASS
  Test 3 | 0x41300000 ( 11.00) |  11.00 | PASS
  Test 4 | 0x00000000 (  0.00) |   0.00 | PASS
  Test 5 | 0xC1200000 (-10.00) | -10.00 | PASS
  Test 6 | 0x3F700000 ( 0.937) |  0.937 | PASS
  Test 7 | 0x42100000 ( 36.00) |  36.00 | PASS

>>> ALL 14 MIXED-PRECISION TESTS PASSED! <<<
```

---

## Files Modified in CV32E40P

Only **one file** in the original CV32E40P RTL was modified:

| File | Change |
|---|---|
| `cv32e40p_decoder.sv` | Added 7 custom instruction cases under `OPCODE_CUSTOM_0` at lines ~1651–1695 |

All other CV32E40P files are used **unmodified**. The integration is non-invasive — the existing APU interface was designed to be extended this way.

---

## How to Regenerate the Test Program

```bash
cd "SRIP 2026/final"
python generate_multi_test_hex.py
# Outputs: test_nvfp4.hex (188 instructions)
# Copy test_nvfp4.hex to: <vivado_project>.sim/sim_1/behav/xsim/
```

---

## Dependencies

- **Vivado 2022.2** (or any compatible Xilinx simulator supporting SystemVerilog)
- **Python 3.x** (for test hex generation only — `struct` module, standard library)
- **CV32E40P** RISC-V core (all RTL files included in `final/cv32e40p_rtl/`)
