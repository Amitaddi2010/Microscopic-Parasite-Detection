import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

# Set publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#2D3748'
plt.rcParams['axes.linewidth'] = 0.8

CLASS_NAMES = [
    'Ancylostoma', 'Ascaris', 'Enterobius', 'Fasciola',
    'Hymenolepis', 'Schistosoma', 'Taenia', 'Trichuris', 'Background'
]

DISPLAY_NAMES = [
    'Ancylostoma\n(Hookworm)', 'Ascaris', 'Enterobius\n(Pinworm)',
    'Fasciola\n(Fluke)', 'Hymenolepis', 'Schistosoma',
    'Taenia', 'Trichuris\n(Whipworm)', 'Background\n(Unmatched)'
]

with open('experiments/metrics/confusion_matrices.json', 'r') as f:
    cms = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(18, 16), dpi=300)
models = [
    ('RT-DETR-L', 'RT-DETR-L (Detection Transformer)', axes[0, 0], 'A'),
    ('YOLO11s', 'YOLO11s (Modern Compact CNN)', axes[0, 1], 'B'),
    ('YOLOv8s', 'YOLOv8s (Standard CNN Baseline)', axes[1, 0], 'C'),
    ('YOLO11n', 'YOLO11n (Ultra-Compact Edge CNN)', axes[1, 1], 'D')
]

for key, title, ax, panel_tag in models:
    raw_cm = np.array(cms[key])
    
    # Normalize by true class (row-wise), handling zero rows gracefully
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(raw_cm, row_sums, out=np.zeros_like(raw_cm, dtype=float), where=(row_sums > 0)) * 100.0

    # Custom heatmap
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        cbar=(panel_tag == 'B' or panel_tag == 'D'),
        cbar_kws={'label': 'Row Normalized Rate (%)', 'shrink': 0.8} if (panel_tag in ['B', 'D']) else None,
        ax=ax,
        xticklabels=DISPLAY_NAMES,
        yticklabels=DISPLAY_NAMES,
        linewidths=0.5,
        linecolor='#E2E8F0',
        annot_kws={'size': 8.5, 'weight': 'bold', 'color': '#1A202C'}
    )

    ax.set_title(f'({panel_tag}) {title}', fontsize=13, fontweight='bold', pad=12, color='#1A365D')
    ax.set_xlabel('Predicted Diagnostic Class', fontsize=10.5, fontweight='bold', labelpad=8, color='#2D3748')
    ax.set_ylabel('Ground-Truth Class', fontsize=10.5, fontweight='bold', labelpad=8, color='#2D3748')
    ax.tick_params(axis='x', rotation=45, labelsize=8.5)
    ax.tick_params(axis='y', rotation=0, labelsize=8.5)

plt.suptitle(
    'Held-Out Test Set Multi-Class Confusion Matrices Across Benchmarked Object Detectors (N = 215 Slides, 460 Ova)\n'
    'Normalized by True Class Target Count (Row Sum = 100%). Diagonal Indicates Correct Species Identification.',
    fontsize=14,
    fontweight='bold',
    y=0.99,
    color='#0F172A'
)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

out_path = Path('experiments/metrics/figure4_confusion_matrices.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Successfully generated {out_path}')

# Copy to artifact directory
art_dir = Path(r'C:\Users\Amit_Addi\.gemini\antigravity-ide\brain\4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba')
if art_dir.exists():
    shutil.copy2(out_path, art_dir / 'figure4_confusion_matrices.png')
    print(f'Copied to artifact directory: {art_dir / "figure4_confusion_matrices.png"}')
