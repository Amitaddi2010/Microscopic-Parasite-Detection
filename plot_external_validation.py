import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw
import shutil
import torch
from ultralytics import YOLO, RTDETR

# Set publication style
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#2D3748'
plt.rcParams['axes.linewidth'] = 0.8

with open('experiments/metrics/external_validation_results.json', 'r') as f:
    results = json.load(f)

# Models and styling
ckpts = {
    'RT-DETR-L': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model1_rtdetr/weights/best.pt', RTDETR, '#1E3A8A', '-'),
    'YOLO11s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model2_mamba_yolo/weights/best.pt', YOLO, '#059669', '--'),
    'YOLOv8s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/baseline_yolov8s/weights/best.pt', YOLO, '#D97706', '-.'),
    'YOLO11n': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model4_yac_net/weights/best.pt', YOLO, '#DC2626', ':')
}

base_dir = Path('E:/SpainWork/ConferencePaper/External vaildation')
all_images = sorted(list(base_dir.glob('*/*.jpg')))

# Granular evaluation across all 1,000 external images using batch_size=32
eval_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
curve_data = {m: [] for m in ckpts}

print(f"Evaluating curve on all {len(all_images)} external images...")

for m_name, (ckpt_p, m_cls, color, ls) in ckpts.items():
    model = m_cls(ckpt_p)
    all_res_boxes = []
    
    # Process in chunks of 32
    for i in range(0, len(all_images), 32):
        batch = [str(p) for p in all_images[i:i+32]]
        with torch.no_grad():
            res = model.predict(batch, conf=0.05, imgsz=640, device=0, verbose=False)
        for r in res:
            b = r.boxes
            if len(b) > 0:
                confs = b.conf.cpu().numpy()
                clss = b.cls.cpu().numpy().astype(int)
                all_res_boxes.append((clss, confs))
            else:
                all_res_boxes.append((np.array([]), np.array([])))
                
    # Calculate sensitivity across thresholds on all 1,000 images
    for th in eval_thresholds:
        hit = sum(1 for clss, confs in all_res_boxes if any((c == 1 and s >= th) for c, s in zip(clss, confs)))
        rate = (hit / len(all_images)) * 100.0
        curve_data[m_name].append(rate)
        if th in [0.15, 0.25, 0.35, 0.50]:
            print(f"  {m_name} @ {th:.2f}: {rate:.2f}%")

# Create publication figure
fig = plt.figure(figsize=(18, 12), dpi=300)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.3], hspace=0.32, wspace=0.22)

# PANEL A: Sensitivity vs Operating Threshold
ax_a = fig.add_subplot(gs[0, 0])
for m_name, (ckpt_p, m_cls, color, ls) in ckpts.items():
    ax_a.plot(eval_thresholds, curve_data[m_name], label=m_name, color=color, linestyle=ls, linewidth=2.8, marker='o', markersize=5)

ax_a.axvline(0.25, color='#64748B', linestyle='--', alpha=0.8, label='Standard Operating Threshold (0.25)')
ax_a.set_title('(A) Cross-Domain Diagnostic Sensitivity vs. Operating Confidence Threshold', fontsize=12, fontweight='bold', color='#1A365D', pad=10)
ax_a.set_xlabel('Confidence Operating Threshold', fontsize=10.5, fontweight='bold', color='#2D3748')
ax_a.set_ylabel('Slide-Level Ascaris Sensitivity (%)', fontsize=10.5, fontweight='bold', color='#2D3748')
ax_a.set_xlim(0.08, 0.62)
ax_a.set_ylim(20, 102)
ax_a.grid(True, linestyle=':', alpha=0.6)
ax_a.legend(loc='lower left', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# PANEL B: Full 1,000 Images Breakdown at Threshold 0.25
ax_b = fig.add_subplot(gs[0, 1])
model_names_list = ['RT-DETR-L', 'YOLO11s', 'YOLOv8s', 'YOLO11n']
x = np.arange(len(model_names_list))
width = 0.24

ascaris_rates = [results[m]['threshold_evaluations']['conf_0.25']['ascaris_sensitivity_pct'] for m in model_names_list]
top1_rates = [results[m]['threshold_evaluations']['conf_0.25']['top1_ascaris_pct'] for m in model_names_list]
any_rates = [results[m]['threshold_evaluations']['conf_0.25']['any_helminth_detection_pct'] for m in model_names_list]

rects1 = ax_b.bar(x - width, ascaris_rates, width, label='Ascaris Detected (%)', color='#1E3A8A', edgecolor='#0F172A')
rects2 = ax_b.bar(x, top1_rates, width, label='Top-1 Diagnostic Match (%)', color='#0284C7', edgecolor='#0F172A')
rects3 = ax_b.bar(x + width, any_rates, width, label='Any Helminth Flagged (%)', color='#64748B', edgecolor='#0F172A')

ax_b.set_title('(B) Diagnostic Screening Yield on External Cohort (N = 1,000 Slides, Conf = 0.25)', fontsize=12, fontweight='bold', color='#1A365D', pad=10)
ax_b.set_ylabel('Percentage of External Slides (%)', fontsize=10.5, fontweight='bold', color='#2D3748')
ax_b.set_xticks(x)
ax_b.set_xticklabels(model_names_list, fontsize=10, fontweight='bold')
ax_b.set_ylim(0, 115)
ax_b.grid(True, axis='y', linestyle=':', alpha=0.6)
ax_b.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)

# Annotate values on bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax_b.annotate(f'{h:.1f}%',
                      xy=(rect.get_x() + rect.get_width() / 2, h),
                      xytext=(0, 3), textcoords="offset points",
                      ha='center', va='bottom', fontsize=8, fontweight='bold')

# PANEL C: Qualitative Visual Detections on External Microscopy Slides
ax_c = fig.add_subplot(gs[1, :])
ax_c.axis('off')

# Select 2 external slides
sample_external_1 = all_images[15]
sample_external_2 = all_images[100]

CLASS_NAMES = [
    'Ancylostoma', 'Ascaris', 'Enterobius',
    'Fasciola', 'Hymenolepis', 'Schistosoma',
    'Taenia', 'Trichuris'
]

def render_detection_row(img_path):
    base_img = Image.open(img_path).convert('RGB')
    sub_imgs = []
    
    # Ground Truth Raw Slide
    gt_img = base_img.copy()
    d = ImageDraw.Draw(gt_img)
    d.rectangle([0, 0, 640, 35], fill=(30, 41, 59))
    d.text((10, 8), f"External #{img_path.stem[:12]} | GT: Ascaris", fill='white')
    sub_imgs.append(gt_img)
    
    # Models
    for m_name in model_names_list:
        ckpt_p, m_cls, col, _ = ckpts[m_name]
        m = m_cls(ckpt_p)
        with torch.no_grad():
            res = m.predict(str(img_path), conf=0.25, imgsz=640, device=0, verbose=False)[0]
        
        m_img = base_img.copy()
        draw = ImageDraw.Draw(m_img)
        draw.rectangle([0, 0, 640, 35], fill=(15, 23, 42))
        draw.text((10, 8), f"{m_name} Predictions", fill='white')
        
        boxes = res.boxes
        if len(boxes) > 0:
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy()[0]
                conf = float(b.conf.cpu().numpy()[0])
                c = int(b.cls.cpu().numpy()[0])
                c_name = CLASS_NAMES[c]
                box_col = '#10B981' if c == 1 else '#EF4444'
                draw.rectangle([xyxy[0], xyxy[1], xyxy[2], xyxy[3]], outline=box_col, width=3)
                label_txt = f"{c_name}: {conf:.2f}"
                lbl_y0 = max(36.0, xyxy[1] - 20.0)
                lbl_y1 = max(lbl_y0 + 16.0, xyxy[1])
                draw.rectangle([xyxy[0], lbl_y0, xyxy[0] + len(label_txt)*8 + 6, lbl_y1], fill=box_col)
                draw.text((xyxy[0]+3, lbl_y0 + 2), label_txt, fill='white')
        else:
            draw.text((200, 300), "No Detection (Missed)", fill='#DC2626')
            
        sub_imgs.append(m_img)
    return sub_imgs

row1 = render_detection_row(sample_external_1)
row2 = render_detection_row(sample_external_2)

# Stitch 2x5 grid into Panel C
grid_w, grid_h = 640 * 5, 640 * 2
canvas = Image.new('RGB', (grid_w, grid_h), color='#FFFFFF')
for j in range(5):
    canvas.paste(row1[j], (j * 640, 0))
    canvas.paste(row2[j], (j * 640, 640))

ax_c.imshow(canvas)
ax_c.set_title('(C) Qualitative Cross-Domain Detection Performance on Independent External Microscopy Fields', fontsize=12, fontweight='bold', color='#1A365D', pad=12)

plt.suptitle(
    'External Clinical Generalizability Benchmark on Independent Microscopic Cohort (N = 1,000 Out-of-Distribution Slides)\n'
    'Demonstrating Detection Transformer Robustness to Optical and Preparation Shifts in Coprological Wet Mounts',
    fontsize=14,
    fontweight='bold',
    y=0.99,
    color='#0F172A'
)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

out_fig = Path('experiments/metrics/figure5_external_validation.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Successfully generated {out_fig}")

# Copy to brain artifact directory
art_dir = Path(r'C:\Users\Amit_Addi\.gemini\antigravity-ide\brain\4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba')
if art_dir.exists():
    shutil.copy2(out_fig, art_dir / 'figure5_external_validation.png')
    print(f"Copied figure5 to artifact directory: {art_dir / 'figure5_external_validation.png'}")
