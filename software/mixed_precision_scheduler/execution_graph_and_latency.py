"""
===========================================================
PRECISION-AWARE EXECUTION GRAPH & LATENCY ANALYSIS
===========================================================

This script (UPDATED for Validated Scheduler results):
1. Generates precision-aware execution graphs for all 6
   models (color-coded layer diagrams showing NVFP4/BF16/FP32)
2. Computes theoretical latency improvement per model
3. Maps each layer to expected hardware execution path
   (NVFP4 Accelerator vs FP32 ALU vs BF16 unit)

===========================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
import io

# Fix Windows encoding for unicode chars (arrows, etc.)
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace'
)

# =========================================================
# PRECISION DATA FROM SCHEDULER RESULTS (SEED=42)
# =========================================================

# MLP on MNIST (Budget 1%)
MLP_DATA = {
    'model_name': 'MLP on MNIST',
    'fp32_baseline': 97.69,
    'mixed_accuracy': 97.27,
    'layers': [
        {'name': 'Input\n(784)',       'type': 'input',  'precision': 'DATA',  'params': 0,       'shape': '784'},
        {'name': 'FC1\n(784→256)',     'type': 'linear', 'precision': 'NVFP4', 'params': 200704,  'shape': '784×256'},
        {'name': 'ReLU',               'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'FC2\n(256→128)',     'type': 'linear', 'precision': 'NVFP4', 'params': 32768,   'shape': '256×128'},
        {'name': 'ReLU',               'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'FC3\n(128→10)',      'type': 'linear', 'precision': 'NVFP4', 'params': 1280,    'shape': '128×10'},
        {'name': 'Output\n(10)',       'type': 'output', 'precision': 'DATA',  'params': 0,       'shape': '10'},
    ],
}

# CNN on CIFAR-10 (Budget 1% — Validated Scheduler)
CNN_DATA = {
    'model_name': 'CNN on CIFAR-10',
    'fp32_baseline': 76.78,
    'mixed_accuracy': 76.46,
    'layers': [
        {'name': 'Input\n(3×32×32)',       'type': 'input',  'precision': 'DATA',  'params': 0,       'shape': '3×32×32'},
        {'name': 'Conv1\n(3→32, 3×3)',     'type': 'conv',   'precision': 'BF16',  'params': 864,     'shape': '3×32×3×3'},
        {'name': 'BN + ReLU',              'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'Conv2\n(32→64, 3×3)',    'type': 'conv',   'precision': 'BF16',  'params': 18432,   'shape': '32×64×3×3'},
        {'name': 'BN + ReLU\n+ Pool',      'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'Conv3\n(64→128, 3×3)',   'type': 'conv',   'precision': 'NVFP4', 'params': 73728,   'shape': '64×128×3×3'},
        {'name': 'BN + ReLU\n+ Pool',      'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'FC1\n(8192→256)',         'type': 'linear', 'precision': 'NVFP4', 'params': 2097152, 'shape': '8192×256'},
        {'name': 'ReLU +\nDropout',        'type': 'activ',  'precision': 'PASS',  'params': 0,       'shape': ''},
        {'name': 'FC2\n(256→10)',           'type': 'linear', 'precision': 'NVFP4', 'params': 2560,    'shape': '256×10'},
        {'name': 'Output\n(10)',            'type': 'output', 'precision': 'DATA',  'params': 0,       'shape': '10'},
    ],
}

# Transformer on MNIST (Budget 1% — Validated Scheduler)
TRANSFORMER_DATA = {
    'model_name': 'Tiny Transformer on MNIST',
    'fp32_baseline': 97.04,
    'mixed_accuracy': 96.38,
    'layers': [
        {'name': 'Input\n(1×28×28)',               'type': 'input',  'precision': 'DATA',  'params': 0,     'shape': '1×28×28'},
        {'name': 'Patch Embed\n(49→64)',            'type': 'embed',  'precision': 'BF16',  'params': 3136,  'shape': '49×64'},
        {'name': '+ Pos Enc',                      'type': 'activ',  'precision': 'PASS',  'params': 0,     'shape': ''},
        {'name': 'L0: Attn\nQKV Proj',             'type': 'attn',   'precision': 'NVFP4', 'params': 12288, 'shape': '192×64'},
        {'name': 'L0: Attn\nOut Proj',             'type': 'attn',   'precision': 'NVFP4', 'params': 4096,  'shape': '64×64'},
        {'name': 'L0: FFN1\n(64→128)',             'type': 'linear', 'precision': 'NVFP4', 'params': 8192,  'shape': '64×128'},
        {'name': 'L0: FFN2\n(128→64)',             'type': 'linear', 'precision': 'NVFP4', 'params': 8192,  'shape': '128×64'},
        {'name': 'L1: Attn\nQKV Proj',             'type': 'attn',   'precision': 'NVFP4', 'params': 12288, 'shape': '192×64'},
        {'name': 'L1: Attn\nOut Proj',             'type': 'attn',   'precision': 'NVFP4', 'params': 4096,  'shape': '64×64'},
        {'name': 'L1: FFN1\n(64→128)',             'type': 'linear', 'precision': 'NVFP4', 'params': 8192,  'shape': '64×128'},
        {'name': 'L1: FFN2\n(128→64)',             'type': 'linear', 'precision': 'NVFP4', 'params': 8192,  'shape': '128×64'},
        {'name': 'LN + Mean\nPool',                'type': 'activ',  'precision': 'PASS',  'params': 0,     'shape': ''},
        {'name': 'Classifier\n(64→10)',            'type': 'linear', 'precision': 'NVFP4', 'params': 640,   'shape': '64×10'},
        {'name': 'Output\n(10)',                   'type': 'output', 'precision': 'DATA',  'params': 0,     'shape': '10'},
    ],
}

# ResNet-20 on CIFAR-10 (Budget 1% — Validated Scheduler)
# 7 NVFP4 + 15 BF16 + 0 FP32
RESNET_DATA = {
    'model_name': 'ResNet-20 on CIFAR-10',
    'fp32_baseline': 92.05,
    'mixed_accuracy': 91.23,
    'layers': [
        {'name': 'Input\n(3x32x32)',            'type': 'input',  'precision': 'DATA',  'params': 0,     'shape': ''},
        {'name': 'Conv1\n(3->16)',               'type': 'conv',   'precision': 'BF16',  'params': 432,   'shape': ''},
        {'name': 'L1.0 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 2304,  'shape': ''},
        {'name': 'L1.0 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 2304,  'shape': ''},
        {'name': 'L1.1 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 2304,  'shape': ''},
        {'name': 'L1.1 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 2304,  'shape': ''},
        {'name': 'L1.2 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 2304,  'shape': ''},
        {'name': 'L1.2 Conv2',                   'type': 'conv',   'precision': 'BF16',  'params': 2304,  'shape': ''},
        {'name': 'L2.0 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 4608,  'shape': ''},
        {'name': 'L2.0 Conv2',                   'type': 'conv',   'precision': 'BF16',  'params': 9216,  'shape': ''},
        {'name': 'L2.0 Skip',                    'type': 'conv',   'precision': 'BF16',  'params': 512,   'shape': ''},
        {'name': 'L2.1 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 9216,  'shape': ''},
        {'name': 'L2.1 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 9216,  'shape': ''},
        {'name': 'L2.2 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 9216,  'shape': ''},
        {'name': 'L2.2 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 9216,  'shape': ''},
        {'name': 'L3.0 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 18432, 'shape': ''},
        {'name': 'L3.0 Conv2',                   'type': 'conv',   'precision': 'BF16',  'params': 36864, 'shape': ''},
        {'name': 'L3.0 Skip',                    'type': 'conv',   'precision': 'BF16',  'params': 2048,  'shape': ''},
        {'name': 'L3.1 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 36864, 'shape': ''},
        {'name': 'L3.1 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 36864, 'shape': ''},
        {'name': 'L3.2 Conv1',                   'type': 'conv',   'precision': 'BF16',  'params': 36864, 'shape': ''},
        {'name': 'L3.2 Conv2',                   'type': 'conv',   'precision': 'NVFP4', 'params': 36864, 'shape': ''},
        {'name': 'FC\n(64->10)',                 'type': 'linear', 'precision': 'NVFP4', 'params': 640,   'shape': ''},
        {'name': 'Output\n(10)',                 'type': 'output', 'precision': 'DATA',  'params': 0,     'shape': ''},
    ],
}

# ResNet-56 on CIFAR-10 (Budget 1% — Validated Scheduler)
# 45 NVFP4 + 13 BF16 + 0 FP32 (summarized by stage)
RESNET56_DATA = {
    'model_name': 'ResNet-56 on CIFAR-10',
    'fp32_baseline': 92.74,
    'mixed_accuracy': 91.84,
    'layers': [
        {'name': 'Input\n(3x32x32)',            'type': 'input',  'precision': 'DATA',  'params': 0,      'shape': ''},
        {'name': 'Conv1\n(3->16)',               'type': 'conv',   'precision': 'BF16',  'params': 432,    'shape': ''},
        {'name': 'L1 x9 blks\n(14 NVFP4/4 BF16)','type': 'conv',  'precision': 'NVFP4', 'params': 41472,  'shape': ''},
        {'name': 'L2.0 Skip',                    'type': 'conv',   'precision': 'BF16',  'params': 512,    'shape': ''},
        {'name': 'L2 x9 blks\n(14 NVFP4/4 BF16)','type': 'conv',  'precision': 'NVFP4', 'params': 165888, 'shape': ''},
        {'name': 'L3.0 Skip',                    'type': 'conv',   'precision': 'BF16',  'params': 2048,   'shape': ''},
        {'name': 'L3 x9 blks\n(15 NVFP4/3 BF16)','type': 'conv',  'precision': 'NVFP4', 'params': 663552, 'shape': ''},
        {'name': 'FC\n(64->10)',                 'type': 'linear', 'precision': 'BF16',  'params': 640,    'shape': ''},
        {'name': 'Output\n(10)',                 'type': 'output', 'precision': 'DATA',  'params': 0,      'shape': ''},
    ],
}

# VGG-16 on CIFAR-10 (Budget 1% — Validated Scheduler)
# 15 NVFP4 + 1 BF16 + 0 FP32
VGG16_DATA = {
    'model_name': 'VGG-16 on CIFAR-10',
    'fp32_baseline': 93.10,
    'mixed_accuracy': 92.19,
    'layers': [
        {'name': 'Input\n(3x32x32)',            'type': 'input',  'precision': 'DATA',  'params': 0,         'shape': ''},
        {'name': 'Conv1\n(3->64)',               'type': 'conv',   'precision': 'BF16',  'params': 1728,      'shape': ''},
        {'name': 'Conv2\n(64->64)',              'type': 'conv',   'precision': 'NVFP4', 'params': 36864,     'shape': ''},
        {'name': 'Pool',                         'type': 'activ',  'precision': 'PASS',  'params': 0,         'shape': ''},
        {'name': 'Conv3-4\n(128)',               'type': 'conv',   'precision': 'NVFP4', 'params': 221184,    'shape': ''},
        {'name': 'Pool',                         'type': 'activ',  'precision': 'PASS',  'params': 0,         'shape': ''},
        {'name': 'Conv5-7\n(256)',               'type': 'conv',   'precision': 'NVFP4', 'params': 1769472,   'shape': ''},
        {'name': 'Pool',                         'type': 'activ',  'precision': 'PASS',  'params': 0,         'shape': ''},
        {'name': 'Conv8-10\n(512)',              'type': 'conv',   'precision': 'NVFP4', 'params': 5309440,   'shape': ''},
        {'name': 'Pool',                         'type': 'activ',  'precision': 'PASS',  'params': 0,         'shape': ''},
        {'name': 'Conv11-13\n(512)',             'type': 'conv',   'precision': 'NVFP4', 'params': 7077888,   'shape': ''},
        {'name': 'Pool',                         'type': 'activ',  'precision': 'PASS',  'params': 0,         'shape': ''},
        {'name': 'FC1-2\n(512)',                 'type': 'linear', 'precision': 'NVFP4', 'params': 524288,    'shape': ''},
        {'name': 'FC3\n(512->10)',               'type': 'linear', 'precision': 'NVFP4', 'params': 5120,      'shape': ''},
        {'name': 'Output\n(10)',                 'type': 'output', 'precision': 'DATA',  'params': 0,         'shape': ''},
    ],
}


# =========================================================
# COLORS & STYLING
# =========================================================

PRECISION_COLORS = {
    'NVFP4': '#2ecc71',   # Green  — NVFP4 accelerator
    'BF16':  '#f39c12',   # Orange — BF16 unit
    'FP32':  '#e74c3c',   # Red    — FP32 ALU
    'PASS':  '#bdc3c7',   # Gray   — passthrough (no weights)
    'DATA':  '#3498db',   # Blue   — input/output data
}

HARDWARE_MAP = {
    'NVFP4': 'NVFP4 MAC Accelerator',
    'BF16':  'BF16 Compute Unit',
    'FP32':  'FP32 ALU (RISC-V Core)',
    'PASS':  'Passthrough (no MAC)',
    'DATA':  'Memory Interface',
}

# Relative throughput: how many MACs per cycle
# FP32 = 1 MAC/cycle (baseline)
# BF16 = 2 MACs/cycle (2× throughput)
# NVFP4 = 8 MACs/cycle (8× throughput, 4-bit ops)
RELATIVE_THROUGHPUT = {
    'FP32':  1.0,
    'BF16':  2.0,
    'NVFP4': 8.0,
}


# =========================================================
# 1. GENERATE EXECUTION GRAPH
# =========================================================

def draw_execution_graph(model_data, ax):
    """Draw a precision-aware execution graph for one model."""

    layers = model_data['layers']
    n = len(layers)

    # Layout: vertical flow, top to bottom
    y_positions = np.linspace(0.95, 0.05, n)
    x_center = 0.5

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(x_center, 1.0, model_data['model_name'],
            ha='center', va='bottom', fontsize=12,
            fontweight='bold')

    box_width = 0.35
    box_height = 0.85 / n * 0.7

    for i, layer in enumerate(layers):

        y = y_positions[i]
        color = PRECISION_COLORS[layer['precision']]

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x_center - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='#2c3e50',
            linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)

        # Layer name
        ax.text(x_center, y, layer['name'],
                ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')

        # Precision label on the right
        if layer['precision'] not in ('PASS', 'DATA'):
            ax.text(x_center + box_width / 2 + 0.02, y,
                    layer['precision'],
                    ha='left', va='center', fontsize=7,
                    fontweight='bold',
                    color=color)

        # Hardware target on the left
        if layer['precision'] not in ('PASS', 'DATA'):
            hw = HARDWARE_MAP[layer['precision']]
            hw_short = hw.split('(')[0].strip()
            ax.text(x_center - box_width / 2 - 0.02, y,
                    f"→ {hw_short}",
                    ha='right', va='center', fontsize=5.5,
                    color='#7f8c8d', style='italic')

        # Arrow to next layer
        if i < n - 1:
            arrow_y_start = y - box_height / 2
            arrow_y_end = y_positions[i + 1] + box_height / 2
            ax.annotate('', xy=(x_center, arrow_y_end),
                        xytext=(x_center, arrow_y_start),
                        arrowprops=dict(
                            arrowstyle='->', color='#2c3e50',
                            lw=1.5))


def generate_all_graphs():
    """Generate execution graphs for all 6 models."""

    fig, axes = plt.subplots(2, 3, figsize=(24, 22))
    fig.suptitle(
        'Precision-Aware Execution Graphs\n'
        'Layer \u2192 Hardware Dispatch Mapping (Budget 1%)',
        fontsize=16, fontweight='bold', y=0.98
    )

    all_data = [MLP_DATA, CNN_DATA, TRANSFORMER_DATA,
                RESNET_DATA, RESNET56_DATA, VGG16_DATA]
    for ax, data in zip(axes.flat, all_data):
        draw_execution_graph(data, ax)

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=PRECISION_COLORS['NVFP4'],
            label='NVFP4 → Accelerator (8× throughput)'
        ),
        mpatches.Patch(
            facecolor=PRECISION_COLORS['BF16'],
            label='BF16 → BF16 Unit (2× throughput)'
        ),
        mpatches.Patch(
            facecolor=PRECISION_COLORS['FP32'],
            label='FP32 → RISC-V ALU (1× baseline)'
        ),
        mpatches.Patch(
            facecolor=PRECISION_COLORS['PASS'],
            label='Passthrough (activation, no MAC)'
        ),
        mpatches.Patch(
            facecolor=PRECISION_COLORS['DATA'],
            label='Data I/O (memory interface)'
        ),
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=3, fontsize=10, frameon=True,
        edgecolor='#2c3e50'
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig('execution_graphs.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("Saved: execution_graphs.png")
    plt.close()


# =========================================================
# 2. THEORETICAL LATENCY ANALYSIS
# =========================================================

def compute_latency(model_data):
    """
    Compute theoretical latency improvement.

    Latency model:
      - Each weight parameter requires 1 MAC operation
      - FP32: 1 MAC/cycle (baseline)
      - BF16: 2 MACs/cycle (2× throughput)
      - NVFP4: 8 MACs/cycle (8× throughput)
      - Latency = total_macs / throughput

    Returns dict with per-layer and total latency.
    """

    results = []
    total_fp32_cycles = 0
    total_mixed_cycles = 0

    for layer in model_data['layers']:

        if layer['params'] == 0:
            continue

        params = layer['params']
        prec = layer['precision']

        fp32_cycles = params / RELATIVE_THROUGHPUT['FP32']
        mixed_cycles = params / RELATIVE_THROUGHPUT[prec]

        total_fp32_cycles += fp32_cycles
        total_mixed_cycles += mixed_cycles

        results.append({
            'name': layer['name'].replace('\n', ' '),
            'params': params,
            'precision': prec,
            'fp32_cycles': fp32_cycles,
            'mixed_cycles': mixed_cycles,
            'speedup': fp32_cycles / mixed_cycles,
            'hardware': HARDWARE_MAP[prec],
        })

    overall_speedup = total_fp32_cycles / total_mixed_cycles

    return {
        'model': model_data['model_name'],
        'layers': results,
        'total_fp32_cycles': total_fp32_cycles,
        'total_mixed_cycles': total_mixed_cycles,
        'overall_speedup': overall_speedup,
        'fp32_acc': model_data['fp32_baseline'],
        'mixed_acc': model_data['mixed_accuracy'],
    }


def print_latency_table(latency):
    """Print formatted latency analysis."""

    print(f"\n{'='*80}")
    print(f"  LATENCY ANALYSIS: {latency['model']}")
    print(f"{'='*80}")

    print(f"\n  {'Layer':<30} {'Params':>10} {'Prec':>6} "
          f"{'FP32 Cyc':>10} {'Mixed Cyc':>10} {'Speedup':>8}")
    print(f"  {'-'*74}")

    for l in latency['layers']:
        print(f"  {l['name']:<30} {l['params']:>10,} "
              f"{l['precision']:>6} {l['fp32_cycles']:>10,.0f} "
              f"{l['mixed_cycles']:>10,.0f} "
              f"{l['speedup']:>7.1f}×")

    print(f"  {'-'*74}")
    print(f"  {'TOTAL':<30} "
          f"{'':>10} {'':>6} "
          f"{latency['total_fp32_cycles']:>10,.0f} "
          f"{latency['total_mixed_cycles']:>10,.0f} "
          f"{latency['overall_speedup']:>7.1f}×")

    print(f"\n  FP32 Accuracy:    {latency['fp32_acc']:.2f}%")
    print(f"  Mixed Accuracy:   {latency['mixed_acc']:.2f}%")
    print(f"  Accuracy Cost:    "
          f"{latency['fp32_acc'] - latency['mixed_acc']:.2f}%")
    print(f"  Latency Speedup:  "
          f"{latency['overall_speedup']:.2f}×")


def generate_latency_chart(all_latency):
    """Generate bar chart comparing latency across models."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    models = [l['model'].split(' on ')[0] for l in all_latency]
    fp32_cyc = [l['total_fp32_cycles'] for l in all_latency]
    mixed_cyc = [l['total_mixed_cycles'] for l in all_latency]
    speedups = [l['overall_speedup'] for l in all_latency]

    # --- Left: Cycle comparison ---
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, fp32_cyc, width,
                    label='FP32 (baseline)',
                    color='#e74c3c', alpha=0.85)
    bars2 = ax1.bar(x + width / 2, mixed_cyc, width,
                    label='Mixed Precision',
                    color='#2ecc71', alpha=0.85)

    ax1.set_ylabel('Total MAC Cycles', fontsize=12)
    ax1.set_title('Theoretical Compute Cycles', fontsize=13,
                  fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9, rotation=15)
    ax1.legend(fontsize=10)
    ax1.ticklabel_format(style='scientific', axis='y',
                         scilimits=(0, 0))

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(), f'{bar.get_height():,.0f}',
                 ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(), f'{bar.get_height():,.0f}',
                 ha='center', va='bottom', fontsize=8)

    # --- Right: Speedup + accuracy trade-off ---
    acc_drops = [l['fp32_acc'] - l['mixed_acc']
                 for l in all_latency]

    colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c',
              '#e74c3c', '#2ecc71']
    bars3 = ax2.bar(x, speedups, 0.5,
                    color=colors[:len(models)], alpha=0.85)

    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Speedup vs Accuracy Trade-off',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9, rotation=15)

    # Add speedup and accuracy drop labels
    for i, (bar, drop) in enumerate(zip(bars3, acc_drops)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'{speedups[i]:.1f}×\n(-{drop:.2f}%)',
                 ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

    ax2.set_ylim(0, max(speedups) * 1.3)
    ax2.axhline(y=1, color='#e74c3c', linestyle='--',
                alpha=0.5, label='FP32 baseline')
    ax2.legend(fontsize=10)

    plt.suptitle(
        'Theoretical Latency Improvement with '
        'Mixed-Precision NVFP4 Scheduling',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig('latency_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    print("Saved: latency_analysis.png")
    plt.close()


# =========================================================
# 3. HARDWARE DISPATCH TABLE
# =========================================================

def print_hardware_dispatch(model_data):
    """Print which hardware unit each layer dispatches to."""

    print(f"\n{'='*80}")
    print(f"  HARDWARE DISPATCH: {model_data['model_name']}")
    print(f"{'='*80}")

    print(f"\n  {'Layer':<35} {'Precision':>8} "
          f"{'Hardware Target':>30}")
    print(f"  {'-'*73}")

    for layer in model_data['layers']:
        if layer['params'] == 0:
            continue

        name = layer['name'].replace('\n', ' ')
        prec = layer['precision']
        hw = HARDWARE_MAP[prec]

        print(f"  {name:<35} {prec:>8} {hw:>30}")


# =========================================================
# MAIN
# =========================================================

ALL_MODELS = [MLP_DATA, CNN_DATA, TRANSFORMER_DATA,
              RESNET_DATA, RESNET56_DATA, VGG16_DATA]


if __name__ == '__main__':

    print("=" * 80)
    print("  PRECISION-AWARE EXECUTION GRAPH & LATENCY ANALYSIS")
    print("  Mixed-Precision Scheduler 4.4 Updated (Validated)")
    print("=" * 80)

    # --- Execution Graphs ---
    print("\n[1/3] Generating execution graphs...")
    generate_all_graphs()

    # --- Hardware Dispatch Tables ---
    print("\n[2/3] Hardware dispatch mapping...")
    for data in ALL_MODELS:
        print_hardware_dispatch(data)

    # --- Latency Analysis ---
    print("\n[3/3] Theoretical latency analysis...")
    all_latency = []
    for data in ALL_MODELS:
        lat = compute_latency(data)
        print_latency_table(lat)
        all_latency.append(lat)

    generate_latency_chart(all_latency)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  CROSS-MODEL SUMMARY (Budget 1%)")
    print(f"{'='*80}")

    print(f"\n  {'Model':<30} {'FP32 Acc':>10} {'Mixed Acc':>10} "
          f"{'Drop':>7} {'Speedup':>9} {'Compress':>10}")
    print(f"  {'-'*76}")

    compress_map = {
        'MLP on MNIST': '5.33\u00d7',
        'CNN on CIFAR-10': '5.26\u00d7',
        'Tiny Transformer on MNIST': '4.96\u00d7',
        'ResNet-20 on CIFAR-10': '2.58\u00d7',
        'ResNet-56 on CIFAR-10': '4.44\u00d7',
        'VGG-16 on CIFAR-10': '5.33\u00d7',
    }

    for lat in all_latency:
        drop = lat['fp32_acc'] - lat['mixed_acc']
        c = compress_map.get(lat['model'], 'N/A')
        print("  {:<30} {:>9.2f}% {:>9.2f}% {:>6.2f}% {:>8.2f}x {:>10}".format(
              lat['model'], lat['fp32_acc'], lat['mixed_acc'],
              drop, lat['overall_speedup'], c))

    print(f"\n  Note: Speedup assumes NVFP4=8× throughput, "
          f"BF16=2×, FP32=1× (baseline)")
    print(f"  Compression = memory savings from scheduler")
    print(f"{'='*80}")
