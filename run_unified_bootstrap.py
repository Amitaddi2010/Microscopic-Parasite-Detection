import os
import sys
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO, RTDETR
from ultralytics.utils.metrics import DetMetrics, ap_per_class

CLASS_NAMES = [
    'Ancylostoma Spp', 'Ascaris Lumbricoides', 'Enterobius Vermicularis',
    'Fasciola Hepatica', 'Hymenolepis', 'Schistosoma',
    'Taenia Sp', 'Trichuris Trichiura'
]

NAMES_DICT = {i: name for i, name in enumerate(CLASS_NAMES)}

def collect_per_image_stats():
    ckpts = {
        'RT-DETR-L': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model1_rtdetr/weights/best.pt', RTDETR),
        'YOLO11s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model2_mamba_yolo/weights/best.pt', YOLO),
        'YOLOv8s': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/baseline_yolov8s/weights/best.pt', YOLO),
        'YOLO11n': ('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model4_yac_net/weights/best.pt', YOLO)
    }

    pkl_cache = Path("experiments/metrics/recorded_stats.pkl")
    if pkl_cache.exists():
        import pickle
        print(f"Loading cached per-image stats from {pkl_cache}...")
        with open(pkl_cache, "rb") as f:
            return pickle.load(f)

    model_stats = {}
    orig_update = DetMetrics.update_stats

    for model_name, (ckpt_path, model_cls) in ckpts.items():
        print(f"Collecting stats for {model_name}...")
        recorded = []

        def make_hook(rec_list):
            def hooked_update(self, stats):
                rec_list.append({
                    'tp': stats['tp'].copy() if hasattr(stats['tp'], 'copy') else np.array(stats['tp']),
                    'conf': stats['conf'].copy() if hasattr(stats['conf'], 'copy') else np.array(stats['conf']),
                    'pred_cls': stats['pred_cls'].copy() if hasattr(stats['pred_cls'], 'copy') else np.array(stats['pred_cls']),
                    'target_cls': stats['target_cls'].copy() if hasattr(stats['target_cls'], 'copy') else np.array(stats['target_cls']),
                    'im_name': stats['im_name']
                })
                return orig_update(self, stats)
            return hooked_update

        DetMetrics.update_stats = make_hook(recorded)
        model = model_cls(ckpt_path)
        val_res = model.val(data='dataset.yaml', split='test', imgsz=640, device=0, verbose=False)
        DetMetrics.update_stats = orig_update

        # Sort recorded by im_name to ensure identical image ordering across all models
        recorded.sort(key=lambda x: x['im_name'])
        model_stats[model_name] = {
            'records': recorded,
            'official_map50': float(val_res.box.map50),
            'official_map50_95': float(val_res.box.map),
            'official_mp': float(val_res.box.mp),
            'official_mr': float(val_res.box.mr),
            'names': val_res.names
        }
        print(f"  {model_name}: {len(recorded)} images, official mAP50 = {val_res.box.map50*100:.2f}%, mAP50-95 = {val_res.box.map*100:.2f}%")

    # Verify identical im_names across models
    base_names = [r['im_name'] for r in model_stats['RT-DETR-L']['records']]
    for m in ['YOLO11s', 'YOLOv8s', 'YOLO11n']:
        m_names = [r['im_name'] for r in model_stats[m]['records']]
        assert base_names == m_names, f"Image order mismatch in {m}!"

    import pickle
    with open(pkl_cache, "wb") as f:
        pickle.dump(model_stats, f)
    print(f"Cached per-image stats to {pkl_cache}")

    return model_stats

def eval_subset(records, names_dict=NAMES_DICT):
    if len(records) == 0:
        return 0.0, 0.0, np.zeros(8)
    
    tp_list = [r['tp'] for r in records if len(r['tp']) > 0]
    conf_list = [r['conf'] for r in records if len(r['conf']) > 0]
    pred_cls_list = [r['pred_cls'] for r in records if len(r['pred_cls']) > 0]
    target_cls_list = [r['target_cls'] for r in records if len(r['target_cls']) > 0]

    if len(tp_list) == 0 or len(target_cls_list) == 0:
        return 0.0, 0.0, np.zeros(8)

    tp = np.concatenate(tp_list, 0)
    conf = np.concatenate(conf_list, 0)
    pred_cls = np.concatenate(pred_cls_list, 0)
    target_cls = np.concatenate(target_cls_list, 0)

    # ap_per_class
    tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp, conf, pred_cls, target_cls, names=names_dict
    )
    
    # Map back to 8 classes
    class_ap50 = np.zeros(8)
    class_ap50_95 = np.zeros(8)
    for idx, c in enumerate(unique_classes):
        if c < 8:
            class_ap50[c] = ap[idx, 0]
            class_ap50_95[c] = ap[idx].mean()

    # Macro mean across classes present in target_cls
    # Note: Ultralytics computes macro mean across classes present in evaluation targets
    map50 = class_ap50[unique_classes].mean() if len(unique_classes) > 0 else 0.0
    map50_95 = class_ap50_95[unique_classes].mean() if len(unique_classes) > 0 else 0.0

    return map50, map50_95, class_ap50

def run_bootstrap_analysis(model_stats, B=5000, seed=42):
    np.random.seed(seed)
    N = len(model_stats['RT-DETR-L']['records'])
    print(f"\nRunning B={B} Paired Bootstrap Resamples on N={N} test images...")

    # Identify single-egg vs multi-egg indices based on ground truth count
    gt_counts = [len(r['target_cls']) for r in model_stats['RT-DETR-L']['records']]
    single_indices = np.array([i for i, c in enumerate(gt_counts) if c == 1])
    multi_indices = np.array([i for i, c in enumerate(gt_counts) if c > 1])

    print(f"Stratification: {len(single_indices)} single-egg images (135), {len(multi_indices)} multi-egg images (80). Total = {N}")

    # Compute observed point estimates
    models = ['RT-DETR-L', 'YOLO11s', 'YOLOv8s', 'YOLO11n']
    observed = {}
    for m in models:
        recs = model_stats[m]['records']
        map50, map50_95, cls_ap50 = eval_subset(recs)
        single_map50, single_map50_95, _ = eval_subset([recs[i] for i in single_indices])
        multi_map50, multi_map50_95, _ = eval_subset([recs[i] for i in multi_indices])

        observed[m] = {
            'overall_map50': float(map50 * 100),
            'overall_map50_95': float(map50_95 * 100),
            'single_map50': float(single_map50 * 100),
            'multi_map50': float(multi_map50 * 100),
            'cls_ap50': (cls_ap50 * 100).tolist()
        }
        print(f"Observed {m:10s}: Overall = {map50*100:.2f}%, Single = {single_map50*100:.2f}%, Multi = {multi_map50*100:.2f}%")

    # Arrays to store bootstrap draws
    boot_overall = {m: np.zeros(B) for m in models}
    boot_single = {m: np.zeros(B) for m in models}
    boot_multi = {m: np.zeros(B) for m in models}
    boot_cls_ap50 = {m: np.zeros((B, 8)) for m in models}

    # Paired draws
    for b in range(B):
        if (b + 1) % 1000 == 0:
            print(f"  Bootstrap progress: {b+1}/{B} iterations...")

        # 1. Overall test resampling (215 images)
        boot_idx = np.random.choice(N, size=N, replace=True)
        for m in models:
            recs = [model_stats[m]['records'][i] for i in boot_idx]
            map50, _, cls_ap50 = eval_subset(recs)
            boot_overall[m][b] = map50 * 100
            boot_cls_ap50[m][b] = cls_ap50 * 100

        # 2. Stratified single-egg resampling (135 images)
        boot_single_idx = np.random.choice(single_indices, size=len(single_indices), replace=True)
        for m in models:
            recs = [model_stats[m]['records'][i] for i in boot_single_idx]
            map50, _, _ = eval_subset(recs)
            boot_single[m][b] = map50 * 100

        # 3. Stratified multi-egg resampling (80 images)
        boot_multi_idx = np.random.choice(multi_indices, size=len(multi_indices), replace=True)
        for m in models:
            recs = [model_stats[m]['records'][i] for i in boot_multi_idx]
            map50, _, _ = eval_subset(recs)
            boot_multi[m][b] = map50 * 100

    # Summary calculations
    results = {'observed': observed, 'B': B}
    
    print("\n" + "="*70)
    print("UNIFIED BOOTSTRAP POINT ESTIMATES, MEANS, BIAS, AND 95% CIs")
    print("="*70)
    results['marginal'] = {}
    for m in models:
        obs = observed[m]['overall_map50']
        b_mean = float(boot_overall[m].mean())
        bias = float(b_mean - obs)
        ci_low = float(np.percentile(boot_overall[m], 2.5))
        ci_high = float(np.percentile(boot_overall[m], 97.5))
        results['marginal'][m] = {
            'point': obs,
            'boot_mean': b_mean,
            'bias': bias,
            'ci95': [ci_low, ci_high]
        }
        print(f"{m:10s}: Obs = {obs:.2f}%, Boot Mean = {b_mean:.2f}%, Bias = {bias:+.2f}%, 95% CI = [{ci_low:.2f}%, {ci_high:.2f}%]")

    # Overall Paired Differences
    print("\n" + "="*70)
    print("OVERALL PAIRED DIFFERENCES (Model A - Model B) [%]")
    print("="*70)
    results['paired_overall'] = {}
    pairs = [
        ('RT-DETR-L', 'YOLOv8s'),
        ('RT-DETR-L', 'YOLO11s'),
        ('YOLO11s', 'YOLOv8s'),
        ('YOLO11n', 'YOLOv8s')
    ]
    for ma, mb in pairs:
        delta_obs = observed[ma]['overall_map50'] - observed[mb]['overall_map50']
        boot_delta = boot_overall[ma] - boot_overall[mb]
        d_mean = float(boot_delta.mean())
        d_bias = float(d_mean - delta_obs)
        ci_low = float(np.percentile(boot_delta, 2.5))
        ci_high = float(np.percentile(boot_delta, 97.5))
        p_val = float((boot_delta <= 0).mean()) if delta_obs > 0 else float((boot_delta >= 0).mean())
        results['paired_overall'][f"{ma}_vs_{mb}"] = {
            'delta_obs': delta_obs,
            'boot_mean': d_mean,
            'bias': d_bias,
            'ci95': [ci_low, ci_high],
            'p_value': p_val
        }
        print(f"{ma} vs {mb}: Obs Δ = {delta_obs:+.2f} pp, Boot Mean = {d_mean:+.2f} pp, 95% CI = [{ci_low:+.2f}, {ci_high:+.2f}] pp (p={p_val:.4f})")

    # Stratified Paired Differences
    print("\n" + "="*70)
    print("STRATIFIED PAIRED DIFFERENCES (Single-Egg vs Multi-Egg)")
    print("="*70)
    results['stratified_paired'] = {}
    for ma, mb in [('RT-DETR-L', 'YOLOv8s'), ('RT-DETR-L', 'YOLO11s'), ('YOLO11s', 'YOLOv8s')]:
        # Single
        d_s_obs = observed[ma]['single_map50'] - observed[mb]['single_map50']
        boot_d_s = boot_single[ma] - boot_single[mb]
        ci_s_low = float(np.percentile(boot_d_s, 2.5))
        ci_s_high = float(np.percentile(boot_d_s, 97.5))

        # Multi
        d_m_obs = observed[ma]['multi_map50'] - observed[mb]['multi_map50']
        boot_d_m = boot_multi[ma] - boot_multi[mb]
        ci_m_low = float(np.percentile(boot_d_m, 2.5))
        ci_m_high = float(np.percentile(boot_d_m, 97.5))

        # Direct Interaction: (Delta_multi - Delta_single)
        inter_obs = d_m_obs - d_s_obs
        boot_inter = boot_d_m - boot_d_s
        ci_inter_low = float(np.percentile(boot_inter, 2.5))
        ci_inter_high = float(np.percentile(boot_inter, 97.5))
        p_inter = float((boot_inter <= 0).mean()) if inter_obs > 0 else float((boot_inter >= 0).mean())

        results['stratified_paired'][f"{ma}_vs_{mb}"] = {
            'single_delta_obs': d_s_obs,
            'single_ci95': [ci_s_low, ci_s_high],
            'multi_delta_obs': d_m_obs,
            'multi_ci95': [ci_m_low, ci_m_high],
            'interaction_obs': inter_obs,
            'interaction_boot_mean': float(boot_inter.mean()),
            'interaction_ci95': [ci_inter_low, ci_inter_high],
            'interaction_p_value': p_inter
        }
        print(f"\n{ma} vs {mb}:")
        print(f"  Single-egg: Obs Δ = {d_s_obs:+.2f} pp, 95% CI = [{ci_s_low:+.2f}, {ci_s_high:+.2f}] pp")
        print(f"  Multi-egg:  Obs Δ = {d_m_obs:+.2f} pp, 95% CI = [{ci_m_low:+.2f}, {ci_m_high:+.2f}] pp")
        print(f"  Interaction (Δ_multi - Δ_single): Obs = {inter_obs:+.2f} pp, 95% CI = [{ci_inter_low:+.2f}, {ci_inter_high:+.2f}] pp (p={p_inter:.4f})")

    # Per-Class AP50 CIs
    print("\n" + "="*70)
    print("PER-CLASS AP@0.5 POINT ESTIMATES & 95% CIs")
    print("="*70)
    results['per_class'] = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        results['per_class'][c_name] = {}
        row_str = f"{c_name:24s} | "
        for m in models:
            obs = observed[m]['cls_ap50'][c_idx]
            ci_low = float(np.percentile(boot_cls_ap50[m][:, c_idx], 2.5))
            ci_high = float(np.percentile(boot_cls_ap50[m][:, c_idx], 97.5))
            results['per_class'][c_name][m] = {
                'point': obs,
                'ci95': [ci_low, ci_high]
            }
            row_str += f"{m}: {obs:.1f} [{ci_low:.1f}, {ci_high:.1f}] | "
        print(row_str)

    # Save to JSON
    out_path = Path("experiments/metrics/unified_bootstrap_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved comprehensive unified bootstrap results to {out_path}")

    return results

if __name__ == "__main__":
    stats = collect_per_image_stats()
    run_bootstrap_analysis(stats, B=5000, seed=42)
