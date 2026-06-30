# Genus Synthesis Results — cv32e40p_nvfp4_top

**Technology:** GPDK045 (45nm) | **Library:** fast_vdd1v0 (1.1V, 0°C) | **Clock:** 200 MHz (5.0 ns period)

---

## Summary Table (Across Synthesis Stages)

| Metric | Generic | Mapped | Optimized (Final) |
|---|:---:|:---:|:---:|
| **Cell Count** | 54,862 | 19,990 | **19,749** |
| **Cell Area (µm²)** | 165,546 | 56,938 | **56,490** |
| **Total Area (µm²)** | — | 86,735 | **86,128** |
| **WNS / Slack (ps)** | 1,161 | 73 | **0** (met) |
| **TNS (ps)** | 0 | 0 | **0** |
| **Violating Paths** | 0 | 0 | **0** |
| **Total Power (mW)** | 1.317 | 8.803 | **8.716** |
| **Runtime** | 43 min | 46 min | **46 min** |

> [!NOTE]
> Timing is **fully met** — zero violating paths, zero negative slack at 200 MHz.

---

## Area Breakdown (Final — Optimized)

| Block | Cell Count | Cell Area (µm²) | % of Total |
|---|:---:|:---:|:---:|
| **Full SoC** (`cv32e40p_nvfp4_top`) | 19,749 | 56,490 | 100% |
| CV32E40P Core (`core_i`) | 13,574 | 39,516 | **70.0%** |
| Accelerator Adapter (`u_nvfp4_apu`) | 6,175 | 16,974 | **30.0%** |

### Accelerator Adapter Internal Breakdown (from Generic Stage)

| Sub-block | Cell Area (µm²) | Description |
|---|:---:|---|
| 16× NVFP4 MAC array | 7,585 | 16 signed multipliers (474 µm² each) |
| NVFP4 scale multiply | 6,645 | Block-wise FP32 scaling |
| NVFP4 CSA tree (adder) | 3,504 | Dot-product accumulation |
| 4× BF16 FP32 multipliers | 5,940 | 4 unsigned multipliers (1,485 µm² each) |
| 3× BF16 FP32 adders | 1,536 | FP32 accumulation |
| BF16 subtractors | 1,365 | Exponent alignment |
| Remaining logic + routing | ~22,888 | Decoder, muxes, control, registers |

---

## Power Breakdown (Final — Optimized)

| Category | Leakage (µW) | Internal (mW) | Switching (mW) | Total (mW) | % |
|---|:---:|:---:|:---:|:---:|:---:|
| **Register** | 7.04 | 2.860 | 0.527 | **3.394** | 38.9% |
| **Logic** | 15.83 | 2.005 | 3.258 | **5.279** | 60.6% |
| **Clock** | 0.00 | 0.000 | 0.043 | **0.043** | 0.5% |
| Latch | 0.00 | 0.000 | 0.000 | 0.000 | 0.0% |
| **Total** | **22.87** | **4.866** | **3.828** | **8.716** | **100%** |

> [!IMPORTANT]
> Total power: **8.716 mW** at 200 MHz, 45nm, 1.1V fast corner

---

## Instance Breakdown (Final)

| Type | Count | Area (µm²) | Area % |
|---|:---:|:---:|:---:|
| Sequential (FFs) | 2,846 | 20,257 | 35.9% |
| Combinational (logic) | 15,689 | 35,364 | 62.6% |
| Inverters | 1,209 | 858 | 1.5% |
| Buffers | 5 | 10 | 0.0% |
| **Total** | **19,749** | **56,490** | **100%** |

---

## Timing Summary (Final)

| Path Type | Slack (ps) |
|---|:---:|
| Register-to-Register (R2R) | 0 (met exactly) |
| Input-to-Register (I2R) | 1,128 |
| Register-to-Output (R2O) | 1,535 |
| Input-to-Output (I2O) | 1,116 |
| Clock Gating (CG) | 2,450 |

- **Clock period:** 5,000 ps (200 MHz)
- **Critical path slack:** 0 ps → timing **exactly met**
- **No setup violations** across all path groups
