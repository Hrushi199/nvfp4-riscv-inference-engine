# Genus Synthesis — CV32E40P + NVFP4 + BF16 Mixed-Precision SoC
## Technology: GPDK045 / gsclib045 (45nm)

---

## What This Does

This folder runs **Cadence Genus Synthesis** on the mixed-precision RISC-V SoC.
It takes the SystemVerilog RTL source files and maps them to real 45nm standard cells,
producing the gate-level netlist, timing reports, area reports, and power estimates
required for the research paper.

**Top Module:** `cv32e40p_nvfp4_top`  
**Design contains:** CV32E40P RISC-V CPU + NVFP4 APU Adapter + NVFP4 Accelerator + BF16 MAC Unit

---

## Prerequisites

### 1. Cadence Genus Installed
Verify Genus is accessible on your server:
```bash
genus -version
```

### 2. GPDK045 Library Installed
The synthesis uses the **Generic Process Design Kit at 45nm** (`gpdk045 / gsclib045`).
Check that the library is available at your site:
```bash
ls /home/user/pdk/gsclib045_all_v4.8/
```
You should see: `gsclib045/`, `gsclib045_hvt/`, `gsclib045_lvt/`, `gsclib045_tech/`

> **If your library is installed in a different path**, open `run_synthesis.tcl` and  
> update the two variables at the top of the script:
> ```tcl
> set libDir  { /your/actual/path/gsclib045/timing ... }
> ```
> and also update the `read_physical` and `set_db qrc_tech_file` paths.

---

## Folder Structure

```
genus/
├── run_synthesis.tcl      # Main Cadence Genus synthesis script
├── constraints.sdc        # Timing constraints (200 MHz target clock)
├── README.md              # This file
└── src/
    ├── include/           # Package files (read first by Genus)
    │   ├── cv32e40p_apu_core_pkg.sv
    │   ├── cv32e40p_fpu_pkg.sv
    │   └── cv32e40p_pkg.sv
    ├── vendor/            # Clock gate model
    │   └── cv32e40p_sim_clock_gate.sv
    ├── cpu/               # CV32E40P RISC-V core RTL (27 files, modified decoder)
    │   ├── cv32e40p_core.sv
    │   ├── cv32e40p_decoder.sv   ← MODIFIED (custom ISA added)
    │   ├── cv32e40p_nvfp4_top.sv ← Custom wrapper top
    │   └── ... (24 other CPU modules)
    └── accel/             # Custom accelerator RTL (9 files)
        ├── nvfp4_apu_adapter.sv      ← Routes NVFP4/BF16 opcodes
        ├── nvfp4_accelerator_top.sv  ← 16-channel NVFP4 dot-product
        ├── bf16_mac_unit.sv          ← 4-channel BF16 dot-product
        └── ... (6 other NVFP4 sub-modules)
```

> **Note:** `cv32e40p_fp_wrapper.sv` is intentionally **NOT included** in the synthesis.
> It instantiates `fpnew_top` (the PULP floating-point unit) which is not part of our design.
> Our `nvfp4_apu_adapter` replaces the FPU entirely via the APU interface.

---

## Step-by-Step Synthesis Instructions

### Step 1 — Copy the `genus/` folder to your Linux server
```bash
scp -r genus/ user@your-server:/home/user/srip2026/genus/
```

### Step 2 — Set the library path in the synthesis script
Open `run_synthesis.tcl` and update the `libDir` variable to point to where gpdk045 is installed:
```tcl
# Line ~35 in run_synthesis.tcl:
set libDir [list \
  /actual/path/to/gsclib045/timing     \
  /actual/path/to/gsclib045_hvt/timing  \
  /actual/path/to/gsclib045_lvt/timing  \
]
```
Also update the `read_physical` LEF paths and `set_db qrc_tech_file` path on lines ~75–85.

### Step 3 — Adjust the target clock (optional)
The default constraint in `constraints.sdc` targets **200 MHz (5.0 ns period)**.
If you want to target a different frequency:
```tcl
# In constraints.sdc, change:
create_clock -name clk -period 5.0 [get_ports clk_i]
# For 100 MHz: period 10.0
# For 500 MHz: period 2.0
```

### Step 4 — Launch Genus
```bash
cd /home/user/srip2026/genus/
genus -files run_synthesis.tcl | tee genus_run.log
```

> The synthesis will run three phases automatically:
> 1. `syn_generic` — Technology-independent optimization
> 2. `syn_map`     — Map to gpdk045 standard cells
> 3. `syn_opt`     — Final timing/area/power optimization

---

## Output Files

After synthesis completes, outputs will be organized as:

```
genus/
├── cv32e40p_nvfp4_top.db          # Design database (for incremental runs)
└── OUTPUT/
    ├── outputs_<version>/
    │   └── cv32e40p_nvfp4_top/
    │       ├── generic/            # Post-generic snapshot
    │       ├── mapped/             # Post-mapping snapshot
    │       ├── opt/                # Post-optimization snapshot
    │       └── cv32e40p_nvfp4_top_netlist.v   ← Gate-level Verilog
    │           cv32e40p_nvfp4_top_synth.sdc   ← SDC for P&R
    └── reports_<version>/
        └── cv32e40p_nvfp4_top/
            ├── generic/
            │   ├── power.rpt       # Power at generic stage
            │   └── (report_summary files)
            ├── mapped/
            │   └── power.rpt
            └── opt/
                └── power.rpt       ← Use this for paper metrics
```

### Key Report Commands (inside Genus interactively)
```tcl
# Area breakdown per module
report_area

# Worst-case timing paths (setup slack)
report_timing

# Power breakdown (leakage + dynamic)
report_power

# Cell usage counts
report_gates
```

---

## Expected Synthesis Metrics (45nm gpdk045)

These are approximate values — actual numbers depend on synthesis effort and clock target:

| Metric | Expected Range |
|---|---|
| Total Cell Count | ~50,000 – 150,000 gates |
| Core Area | ~0.2 – 0.8 mm² |
| Max Frequency | 100–300 MHz |
| Total Power @ 100 MHz | ~5 – 25 mW |
| NVFP4 Accel Area | ~5–10% of total area |
| BF16 MAC Area | ~2–5% of total area |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot find library file` | Update `libDir` in `run_synthesis.tcl` to your actual library path |
| `Unresolved reference: fpnew_pkg` | Make sure `cv32e40p_fp_wrapper.sv` is NOT in the RTL list (it is excluded by default) |
| `Unresolved module: cv32e40p_fpu_wrap` | Same as above — check the RTL list in `run_synthesis.tcl` |
| `check_design` reports warnings | Most warnings about unconnected ports on unused modules are safe to ignore |
| Timing violations (negative slack) | Relax the clock period in `constraints.sdc` (e.g., change 5.0 ns to 8.0 ns) |
