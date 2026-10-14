import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO, RTDETR

CLASS_NAMES = [
    'Ancylostoma Spp', 'Ascaris Lumbricoides', 'Enterobius Vermicularis',
    'Fasciola Hepatica', 'Hymenolepis', 'Schistosoma',
    'Taenia Sp', 'Trichuris Trichiura'
]
TARGET_CLASS_IDX = 1 # Ascaris Lumbricoides

base_dir = Path('E:/SpainWork/ConferencePaper/External vaildation')
all_images = sorted(list(base_dir.glob('*/*.jpg')))
print(f"Found {len(all_images)} external validation images.")

ckpts = {
    'RT-DETR-L': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model1_rtdetr/weights/best.pt', RTDETR),
    'YOLO11s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model2_mamba_yolo/weights/best.pt', YOLO),
    'YOLOv8s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/baseline_yolov8s/weights/best.pt', YOLO),
    'YOLO11n': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model4_yac_net/weights/best.pt', YOLO)
}

conf_thresholds = [0.15, 0.25, 0.35, 0.50]
results_summary = {}

for model_name, (ckpt_path, model_cls) in ckpts.items():
    print(f"\n==========================================")
    print(f"Evaluating {model_name} on External Dataset...")
    print(f"==========================================")
    
    model = model_cls(ckpt_path)
    
    # Run batch inference
    batch_size = 32
    all_preds = []
    
    t0 = time.time()
    for i in range(0, len(all_images), batch_size):
        batch_paths = [str(p) for p in all_images[i:i+batch_size]]
        res = model.predict(batch_paths, conf=0.10, imgsz=640, device=0, verbose=False)
        for r, p in zip(res, batch_paths):
            boxes = r.boxes
            img_preds = []
            if len(boxes) > 0:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)
                xywhn = boxes.xywhn.cpu().numpy().astype(float)
                for c, score, box in zip(cls_ids, confs, xywhn):
                    img_preds.append({
                        'cls': int(c),
                        'class_name': CLASS_NAMES[c],
                        'conf': float(score),
                        'box': box.tolist()
                    })
            all_preds.append({
                'image': Path(p).name,
                'predictions': img_preds
            })
    eval_time = time.time() - t0
    fps = len(all_images) / eval_time
    print(f"Inference complete in {eval_time:.2f}s ({fps:.1f} FPS)")
    
    # Analyze sensitivity across confidence thresholds
    thresh_stats = {}
    for th in conf_thresholds:
        images_with_any_detection = 0
        images_with_ascaris = 0
        images_top1_ascaris = 0
        ascaris_confidences = []
        species_counts = {c: 0 for c in CLASS_NAMES}
        
        for item in all_preds:
            preds_above_th = [p for p in item['predictions'] if p['conf'] >= th]
            if len(preds_above_th) > 0:
                images_with_any_detection += 1
                has_ascaris = any(p['cls'] == TARGET_CLASS_IDX for p in preds_above_th)
                if has_ascaris:
                    images_with_ascaris += 1
                    ascaris_preds = [p['conf'] for p in preds_above_th if p['cls'] == TARGET_CLASS_IDX]
                    ascaris_confidences.append(max(ascaris_preds))
                
                # Check top-1 confidence prediction
                top1 = max(preds_above_th, key=lambda x: x['conf'])
                if top1['cls'] == TARGET_CLASS_IDX:
                    images_top1_ascaris += 1
                
                for p in preds_above_th:
                    species_counts[p['class_name']] += 1
                    
        total_n = len(all_images)
        ascaris_sens = (images_with_ascaris / total_n) * 100.0
        top1_sens = (images_top1_ascaris / total_n) * 100.0
        any_det_rate = (images_with_any_detection / total_n) * 100.0
        mean_conf = float(np.mean(ascaris_confidences)) if len(ascaris_confidences) > 0 else 0.0
        median_conf = float(np.median(ascaris_confidences)) if len(ascaris_confidences) > 0 else 0.0
        
        thresh_stats[f"conf_{th}"] = {
            'threshold': th,
            'images_with_ascaris': images_with_ascaris,
            'ascaris_sensitivity_pct': round(ascaris_sens, 2),
            'top1_ascaris_pct': round(top1_sens, 2),
            'any_helminth_detection_pct': round(any_det_rate, 2),
            'mean_ascaris_confidence': round(mean_conf, 4),
            'median_ascaris_confidence': round(median_conf, 4),
            'species_distribution': species_counts
        }
        print(f"  Threshold {th}: Ascaris Sensitivity = {ascaris_sens:.2f}% ({images_with_ascaris}/{total_n}), Top-1 = {top1_sens:.2f}%, Mean Conf = {mean_conf:.3f}")
        
    results_summary[model_name] = {
        'total_images': len(all_images),
        'fps': round(fps, 1),
        'threshold_evaluations': thresh_stats,
        'sample_predictions': all_preds[:20]
    }

# Save complete results
out_json = Path('experiments/metrics/external_validation_results.json')
with open(out_json, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved external validation results to {out_json}")
