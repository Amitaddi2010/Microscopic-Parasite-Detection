import os
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO, RTDETR

CLASS_NAMES = [
    'Ancylostoma Spp', 'Ascaris Lumbricoides', 'Enterobius Vermicularis',
    'Fasciola Hepatica', 'Hymenolepis', 'Schistosoma',
    'Taenia Sp', 'Trichuris Trichiura'
]

CLASS_COLORS = {
    0: '#E6194B', # Ancylostoma - Red
    1: '#3CBL00', # Ascaris - Green
    2: '#4363D8', # Enterobius - Blue
    3: '#F58231', # Fasciola - Orange
    4: '#911EB4', # Hymenolepis - Purple
    5: '#42D4F4', # Schistosoma - Cyan
    6: '#F032E6', # Taenia - Magenta
    7: '#BFEF45', # Trichuris - Lime
}

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def generate_figure1():
    print("Generating Figure 1: Benchmark Summary...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    
    models = ['RT-DETR-L', 'YOLO11s', 'YOLOv8s', 'YOLO11n']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    
    # 1. Accuracy
    map50 = [91.80, 88.33, 87.74, 86.82]
    map50_low = [88.06, 83.77, 83.52, 82.08]
    map50_high = [94.96, 92.35, 91.81, 92.40]
    yerr = [
        [map50[i] - map50_low[i] for i in range(4)],
        [map50_high[i] - map50[i] for i in range(4)]
    ]
    
    bars1 = axes[0].bar(models, map50, yerr=yerr, capsize=5, color=colors, alpha=0.88, edgecolor='black', linewidth=1.2)
    axes[0].set_ylim(75, 100)
    axes[0].set_ylabel('mAP@0.5 (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('(A) Detection Accuracy (with 95% Bootstrap CI)', fontsize=11, fontweight='bold', pad=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    # 2. FPS & Latency
    fps = [47.1, 141.6, 201.5, 148.0]
    bars2 = axes[1].bar(models, fps, color=colors, alpha=0.88, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Frames Per Second (FPS)', fontsize=11, fontweight='bold')
    axes[1].set_title('(B) Inference Throughput on RTX 5060 Ti', fontsize=11, fontweight='bold', pad=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 3.0, f"{yval:.1f}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    # 3. Parameters vs Accuracy
    params = [32.0, 9.4, 11.1, 2.58]
    bars3 = axes[2].bar(models, params, color=colors, alpha=0.88, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    axes[2].set_title('(C) Computational Footprint (Params)', fontsize=11, fontweight='bold', pad=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}M", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    plt.tight_layout()
    out_path = Path("experiments/metrics/figure1_benchmark_summary.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def generate_figure2():
    print("Generating Figure 2: Qualitative Model Comparisons...")
    ckpts = {
        'RT-DETR-L': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model1_rtdetr/weights/best.pt', RTDETR),
        'YOLO11s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model2_mamba_yolo/weights/best.pt', YOLO),
        'YOLOv8s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/baseline_yolov8s/weights/best.pt', YOLO),
        'YOLO11n': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model4_yac_net/weights/best.pt', YOLO)
    }
    
    models = {}
    for name, (path, cls) in ckpts.items():
        models[name] = cls(path)
        
    with open('Dataset/test/_annotations.coco.json', 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    img_map = {img['id']: img for img in coco['images']}
    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)
        
    # Select 4 representative images from test split
    # 1. Multi-egg Enterobius (Pinworm)
    # 2. Thin-Shelled Hookworm (Ancylostoma Spp, img_id 4)
    # 3. Mammillated Roundworm (Ascaris lumbricoides, img_id 22 - clean, unwatermarked)
    # 4. Large Operculated Fluke (Fasciola hepatica, img_id 6)
    selected = [
        (0, "Multi-Egg Field (Enterobius vermicularis)"),
        (4, "Thin-Shelled Hookworm (Ancylostoma Spp)"),
        (22, "Mammillated Ovum (Ascaris lumbricoides)"),
        (6, "Large Fluke Ovum (Fasciola hepatica)")
    ]
    
    fig, axes = plt.subplots(4, 5, figsize=(18, 14.5), dpi=300)
    col_titles = ['Ground Truth Annotation', 'RT-DETR-L (Transformer)', 'YOLO11s (Compact CNN)', 'YOLOv8s Baseline', 'YOLO11n (Edge CNN)']
    
    for row_idx, (img_id, row_desc) in enumerate(selected):
        img_info = img_map[img_id]
        img_path = Path('Dataset/test') / img_info['file_name']
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        
        # 1. Ground Truth
        gt_img = img_rgb.copy()
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id'] - 1
            color = (0, 255, 0) # Green for GT
            cv2.rectangle(gt_img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            lbl = f"GT: {CLASS_NAMES[cat_id][:12]}"
            cv2.putText(gt_img, lbl, (int(x), max(18, int(y)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            
        axes[row_idx, 0].imshow(gt_img)
        axes[row_idx, 0].set_ylabel(f"Slide {row_idx+1}\n{row_desc}", fontsize=9.5, fontweight='bold', rotation=90, labelpad=8)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=11, fontweight='bold', pad=8)
            
        # Predictions across 4 models
        for col_idx, (m_name, m_obj) in enumerate(models.items(), start=1):
            pred_res = m_obj.predict(img_path, conf=0.25, imgsz=640, verbose=False)[0]
            pred_img = img_rgb.copy()
            
            boxes = pred_res.boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                color = (0, 120, 255) if col_idx == 1 else (255, 100, 0)
                cv2.rectangle(pred_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                lbl = f"{CLASS_NAMES[cls_id][:10]} {conf:.2f}"
                cv2.putText(pred_img, lbl, (xyxy[0], max(18, xyxy[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)
                
            axes[row_idx, col_idx].imshow(pred_img)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight='bold', pad=8)
                
    plt.tight_layout()
    out_path = Path("experiments/metrics/figure2_qualitative_comparisons.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def generate_figure3():
    print("Generating Figure 3: Precision-Recall & F1 Operating Curves...")
    with open('experiments/metrics/recorded_stats.pkl', 'rb') as f:
        import pickle
        stats = pickle.load(f)
        
    from ultralytics.utils.metrics import ap_per_class
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    models = ['RT-DETR-L', 'YOLO11s', 'YOLOv8s', 'YOLO11n']
    colors = {'RT-DETR-L': '#1f77b4', 'YOLO11s': '#2ca02c', 'YOLOv8s': '#d62728', 'YOLO11n': '#ff7f0e'}
    
    for m in models:
        recs = stats[m]['records']
        tp = np.concatenate([r['tp'] for r in recs if len(r['tp']) > 0], 0)
        conf = np.concatenate([r['conf'] for r in recs if len(r['conf']) > 0], 0)
        pred_cls = np.concatenate([r['pred_cls'] for r in recs if len(r['pred_cls']) > 0], 0)
        target_cls = np.concatenate([r['target_cls'] for r in recs if len(r['target_cls']) > 0], 0)
        
        _, _, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
            tp, conf, pred_cls, target_cls
        )
        
        # 1. Macro PR curve
        # prec_values is shape (nc, 1000)
        mean_prec = prec_values.mean(axis=0)
        # x is recall values from 0 to 1
        map50 = ap[:, 0].mean() * 100
        axes[0].plot(x, mean_prec, color=colors[m], label=f"{m} (mAP@0.5: {map50:.1f}%)", linewidth=2.2)
        
        # 2. Macro F1-score vs confidence threshold
        mean_f1 = f1_curve.mean(axis=0)
        best_idx = np.argmax(mean_f1)
        axes[1].plot(x, mean_f1, color=colors[m], label=f"{m} (Peak F1: {mean_f1[best_idx]*100:.1f}% @ {x[best_idx]:.2f})", linewidth=2.2)
        
    axes[0].set_xlabel('Recall', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0].set_title('(A) Macro Precision-Recall Curves on Test Set', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower left', fontsize=9.5, framealpha=0.9)
    
    axes[1].set_xlabel('Confidence Operating Threshold', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1].set_title('(B) F1-Score vs. Confidence Operating Curves', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='lower center', fontsize=9.5, framealpha=0.9)
    
    plt.tight_layout()
    out_path = Path("experiments/metrics/figure3_pr_f1_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    generate_figure3()
    print("All publication figures successfully created!")
