import time
import torch
import numpy as np
from ultralytics import YOLO, RTDETR

device = 'cuda:0'
dummy = torch.rand(1, 3, 640, 640, device=device) # normalized 0-1

models = {
    'RT-DETR-L': RTDETR('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model1_rtdetr/weights/best.pt'),
    'YOLO11s': YOLO('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model2_mamba_yolo/weights/best.pt'),
    'YOLOv8s': YOLO('E:/Amit Data/BT/runs/detect/experiments/checkpoints/baseline_yolov8s/weights/best.pt'),
    'YOLO11n': YOLO('E:/Amit Data/BT/runs/detect/experiments/checkpoints/model4_yac_net/weights/best.pt')
}

res = {}
for name, m in models.items():
    net = m.model.to(device).eval()
    for _ in range(25):
        with torch.no_grad():
            _ = net(dummy)
    torch.cuda.synchronize()
    
    t_fwd = []
    for _ in range(150):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = net(dummy)
        torch.cuda.synchronize()
        t_fwd.append((time.perf_counter() - t0) * 1000)
        
    t_e2e = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = m.predict(dummy, verbose=False)
        torch.cuda.synchronize()
        t_e2e.append((time.perf_counter() - t0) * 1000)
        
    fwd_m = np.mean(t_fwd)
    fwd_s = np.std(t_fwd)
    e2e_m = np.mean(t_e2e)
    e2e_s = np.std(t_e2e)
    fwd_fps = 1000.0 / fwd_m
    e2e_fps = 1000.0 / e2e_m
    
    res[name] = (fwd_m, fwd_s, fwd_fps, e2e_m, e2e_s, e2e_fps)

print("\n" + "="*80)
print(f"{'Model':10} | {'Pure Fwd (ms)':16} | {'Fwd FPS':8} | {'End-to-End (ms)':18} | {'E2E FPS':8}")
print("="*80)
for k, (fm, fs, ffps, em, es, efps) in res.items():
    print(f"{k:10} | {fm:5.2f} +/- {fs:4.2f} ms | {ffps:6.1f}   | {em:5.2f} +/- {es:4.2f} ms   | {efps:6.1f}")
print("="*80)
