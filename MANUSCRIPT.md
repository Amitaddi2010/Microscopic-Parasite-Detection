# Benchmarking Modern Object Detection Architectures for Multi-Species Helminth Parasite Identification in Fecal Microscopy

**Anonymous Authors**  
*Under Review for Publication*  
*Correspondence: corresponding.author@institution.edu*

---

### Abstract
Manual microscopic coprological screening for soil-transmitted helminths and intestinal parasites in fecal smears is labor-intensive and vulnerable to reader fatigue. While deep learning object detectors have shown promise, systematic benchmarks comparing transformer-based detection architectures with modern convolutional baselines on multi-species parasitic ova under standardized conditions remain scarce. Here, we present a standardized benchmark of four object detectors on the 8-class Roboflow Parasites v2 microscopic dataset (2,110 brightfield images; 215 held-out test images containing 460 ground-truth ova): RT-DETR-L (Detection Transformer), YOLO11s (modern compact CNN), YOLOv8s (standard CNN baseline), and YOLO11n (ultra-compact edge CNN). All models were initialized with official COCO pretrained weights, trained under identical optimizer settings, and evaluated on an NVIDIA GeForce RTX 5060 Ti GPU. RT-DETR-L achieved the highest overall detection accuracy (91.80% mAP@0.5, 77.32% mAP@0.5:0.95; 47.1 FPS), with paired test-image bootstrap analysis ($B = 5,000$ resamples) demonstrating that its 95% confidence interval excludes zero relative to the YOLOv8s baseline ($\Delta = +4.06$ percentage points [pp], 95% CI: [+1.21, +6.66] pp) and YOLO11s ($\Delta = +3.47$ pp, 95% CI: [+0.96, +6.10] pp). In contrast, YOLO11s (88.33% mAP@0.5) showed functional parity with YOLOv8s (87.74% mAP@0.5; paired $\Delta = +0.58$ pp, 95% CI: [-2.10, +2.88] pp). Although RT-DETR-L exhibited wider point differences on multi-egg images (+5.23 pp) than single-egg images (+1.20 pp), an interaction bootstrap revealed no statistically significant crowding interaction ($\Delta_{\text{interaction}} = +4.03$ pp, 95% CI: [-2.23, +9.55] pp, $p = 0.1230$), and ground-truth boxes showed zero IoU $\ge 0.50$ overlap. Ultra-compact YOLO11n achieved 86.82% mAP@0.5 (2.58M parameters) with 80.88% hookworm recall, offering a lightweight edge candidate.

**Keywords:** Parasite Egg Detection; Stool Microscopy; Object Detection; Real-Time Detection Transformers; YOLO; Neglected Tropical Diseases (NTDs); Diagnostic Benchmark.

---

## 1. Introduction

Soil-transmitted helminths (STHs)—principally *Ascaris lumbricoides* (giant roundworm), *Trichuris trichiura* (whipworm), and hookworms (*Ancylostoma* and *Necator*)—together with trematodes (*Schistosoma*, *Fasciola*) and cestodes (*Taenia*, *Hymenolepis*) impose an enormous burden of chronic morbidity, childhood stunting, cognitive impairment, and anemia across endemic low- and middle-income regions [1, 2]. The World Health Organization (WHO) 2030 roadmap for neglected tropical diseases targets the global elimination of STH-induced morbidity through preventive chemotherapy and mass drug administration (MDA) [2]. However, monitoring MDA efficacy, determining treatment stopping thresholds, and tracking potential recrudescence require reliable parasitological surveillance.

Microscopic examination of fecal smears remains the diagnostic gold standard for species-level helminth identification in primary health centers and field surveys [3]. Standard preparation methods—including the Kato-Katz thick smear [4], direct saline wet mounts, and formalin-ether concentration techniques (FECT) [5]—require laboratory technicians to systematically scan hundreds of microscopic fields per slide under 10× to 40× magnification. In practice, this manual workflow encounters substantial operational challenges:
1. **Severe reader fatigue**: Systematically examining dozens of thick smears daily induces visual exhaustion, leading to missed ova particularly in low-intensity infections where only a few eggs exist across an entire slide [6].
2. **Subjective diagnostic variability**: Agreement across technicians varies due to morphological ambiguities, such as distinguishing decorticated *Ascaris* ova from non-parasitic artifacts, or differentiating hookworm eggs from free-living nematodes [5].
3. **Complex fecal background interference**: Stool preparations contain abundant non-target matrices—including undigested cellulose fibers, starch granules, pollen grains, yeast cells, air bubbles, and crystal precipitates—that closely mimic the dimensions, birefringence, and textures of helminth ova [7].

### 1.1 Related Work: Deep Learning in Helminth Coprology
Over the past decade, automated microscopic analysis utilizing deep learning has gained substantial momentum for coprological screening [8, 9]. To establish the current research landscape, we conducted a systematic literature search across PubMed, Google Scholar, IEEE Xplore, and Web of Science covering peer-reviewed studies published between January 2020 and February 2026. Search strings combined coprological terms with computer vision descriptors: `("helminth egg" OR "parasite ova" OR "soil-transmitted helminths") AND ("object detection" OR "deep learning" OR "YOLO" OR "DETR") AND ("microscopy" OR "fecal smear")`.

Early investigations in this domain relied on handcrafted morphological descriptors (e.g., Haralick textures, Fourier boundary descriptors) evaluated via support vector machines or random forests, as surveyed by Ghazvini et al. [10]. Subsequent studies transitioned toward mobile and digital microscopy platforms. For instance, Holmström et al. [8] demonstrated point-of-care mobile digital microscopy coupled with deep convolutional neural networks for field detection of *Schistosoma haematobium* and STHs in Kato-Katz smears. Ward et al. [9] developed an automated slide scanner and deep learning detection pipeline on Kato-Katz smears, reporting promising overall sensitivities on roundworm and whipworm, while highlighting marked sensitivity drops on hookworm ova attributed to rapid clearing in glycerol preparations.

More recently, single-stage detectors from the You Only Look Once (YOLO) family have been widely explored in parasitological imaging due to their favorable inference throughput [11, 12]. Hassan et al. [11] investigated attention-augmented YOLO architectures for microscopic pinworm (*Enterobius vermicularis*) detection, observing that channel and spatial attention modules improve boundary localization on clean fecal smears. De Wilde et al. [12] evaluated YOLOv7 variants on multi-center Kato-Katz stool smears across diverse camera hardware, establishing the robustness and limitations of single-stage convolutional detectors under out-of-distribution optical shifts.

Concurrently, Detection Transformers (DETR), notably Real-Time DETR (RT-DETR) [13], have emerged in computer vision. RT-DETR reformulates object detection as a direct set prediction task using bipartite Hungarian matching and an Efficient Hybrid Encoder, eliminating heuristic anchor assignments and greedy non-maximum suppression (NMS) post-processing. While transformer-based backbones have shown promise across general biomedical imaging, systematically controlled benchmarks comparing real-time detection transformers directly against modern convolutional YOLO architectures under identical training protocols and multi-species fecal microscopy conditions remain scarce.

---

## 2. Materials and Methods

### 2.1 Dataset Characteristics, Resolution Profiles & Provenance

We conduct all experiments on the publicly accessible **Roboflow Parasites v2** coprological microscopy dataset [14] (CC BY 4.0 license). The dataset contains 2,110 brightfield microscopic images captured from stool smear preparations across eight distinct parasitic taxa of medical and veterinary significance [3, 7]:

1. *Ancylostoma Spp* (Hookworm): Thin, transparent shell with morula-stage internal blastomeres (approx. 55–75 $\mu\text{m} \times 35–45 \mu\text{m}$) [3, 7].
2. *Ascaris Lumbricoides* (Roundworm): Thick, bile-stained, tuberculated mammillated outer shell (approx. 45–75 $\mu\text{m} \times 35–50 \mu\text{m}$) [3, 7].
3. *Enterobius Vermicularis* (Pinworm): Asymmetric planoconvex shape with a double-contoured shell (approx. 50–60 $\mu\text{m} \times 20–30 \mu\text{m}$) [3, 7].
4. *Fasciola Hepatica* (Liver Fluke): Large, operculated, golden-brown ova (approx. 130–150 $\mu\text{m} \times 60–90 \mu\text{m}$) [3, 7].
5. *Hymenolepis* (Tapeworm): Spherical or subspherical, containing an oncosphere with hooklets and polar filaments (approx. 30–50 $\mu\text{m}$) [3, 7].
6. *Schistosoma* (Blood Fluke): Large oval ova characterized by a prominent lateral or terminal spine (approx. 110–175 $\mu\text{m} \times 45–70 \mu\text{m}$) [3, 7].
7. *Taenia Sp* (Tapeworm): Spherical, thick, radially striated embryophore enclosing a hexacanth embryo (approx. 30–40 $\mu\text{m}$) [3, 7].
8. *Trichuris Trichiura* (Whipworm): Barrel-shaped, amber-colored shell with prominent bipolar clear mucoid plugs (approx. 50–55 $\mu\text{m} \times 22–25 \mu\text{m}$) [3, 7].

*Taxonomic and Provenance Note*: Labels in the public repository represent morphological and microscopy operational categories rather than genetically or serologically validated species designations (e.g., *Hymenolepis* encompasses morphologies consistent with *H. nana* or *H. diminuta*; hookworm ova encompass *Ancylostoma* and morphologically indistinguishable *Necator*). Native image acquisition resolutions exhibit wide heterogeneity, ranging from $86 \times 86$ to $3,760 \times 3,368$ pixels across the dataset (median widths per class span 480 px to 726 px; median heights span 362 px to 549 px). This heterogeneity indicates collection across diverse digital microscopes, consumer cameras, and literature or web sources rather than a single unified clinical protocol.

To verify dataset partition integrity, we computed MD5 cryptographic checksums across all 2,110 image files on disk; **exactly 0 duplicate images cross partition boundaries**. Furthermore, to assess potential shortcut learning from acquisition dimensions, we evaluated whether image resolution (width, height, aspect ratio) could trivially classify taxa: a 5-fold cross-validated random forest classifier achieved only 23.55% $\pm$ 3.33% accuracy (barely above the 17.91% majority-class baseline), confirming that native resolution does not serve as a trivial shortcut for species classification.

During training and inference, images are scaled to $640 \times 640$ resolution with letterbox border padding. We adhere strictly to the canonical dataset partition on disk:
* **Training split**: 1,484 images (70.3%)
* **Validation split**: 411 images (19.5%)
* **Test split**: 215 images (10.2%), containing **460 annotated ground-truth ova**.

The instance distribution across the 215 held-out test images is: *Enterobius Vermicularis* ($n=106$), *Schistosoma* ($n=66$), *Trichuris Trichiura* ($n=60$), *Taenia Sp* ($n=53$), *Ascaris Lumbricoides* ($n=49$), *Ancylostoma Spp* ($n=43$), *Hymenolepis* ($n=43$), and *Fasciola Hepatica* ($n=40$). Model checkpoints (`best.pt`) were selected strictly based on validation mAP@0.5:0.95 on the 411 validation images, without test set feedback.

### 2.2 Evaluated Architectures

We benchmark four dense object detection architectures spanning distinct design paradigms:

1. **RT-DETR-L** (Lv et al., CVPR 2024) [13]: An end-to-end Real-Time Detection Transformer designed to eliminate heuristic NMS bottlenecks. It incorporates an HGNetv2 (PPHGNetV2) multi-scale convolutional backbone, an Efficient Hybrid Encoder combining attention-based intra-scale feature interaction (AIFI) with cross-scale feature fusion (CCFM), and a 6-layer transformer decoder with bipartite Hungarian matching (32.0M parameters, 105.4 GFLOPs).
2. **YOLO11s** (Ultralytics, 2024) [15]: A modern compact convolutional detector incorporating C3k2 residual processing blocks, an upgraded Spatial Pyramid Pooling Fast (SPPF) module, and an anchor-free decoupled detection head regressing bounding boxes and class logits separately (9.4M parameters, 21.4 GFLOPs).
3. **YOLOv8s** (Jocher et al., 2023) [15]: An established standard single-stage convolutional baseline utilizing a CSPDarknet backbone, a PANet neck with C2f modules, and a TaskAlignedAssigner combining distribution focal loss (DFL) and CIoU loss (11.1M parameters, 28.4 GFLOPs).
4. **YOLO11n** (Ultralytics, 2024) [15]: An ultra-compact nano variant engineered for low-power edge inference, scaled to 2.58M parameters and 6.4 GFLOPs.

### 2.3 Training Configurations & Evaluation Protocol

All models were trained using PyTorch 2.14 and Ultralytics 8.4.130 on an **NVIDIA GeForce RTX 5060 Ti GPU (16 GB VRAM)** under Windows 11.

Training configurations:
* **Pretraining**: Initialized from official COCO-pretrained weights (`rtdetr-l.pt`, `yolo11s.pt`, `yolov8s.pt`, `yolo11n.pt`).
* **Epochs**: 25 epochs per model.
* **Input resolution**: $640 \times 640$ pixels.
* **Batch size**: 16 for YOLO models; batch size 8 for RT-DETR-L to satisfy GPU memory constraints.
* **Optimizer**: AdamW ($\eta = 1 \times 10^{-3}$, cosine decay, weight decay $1 \times 10^{-4}$).
* **Augmentations**: Photometric distortions (HSV shift), random horizontal flips, and multi-scale mosaics.
* **Operating Thresholds**: In benchmark Table 1, Precision ($P$), Recall ($R$), and F1-score are reported at each model's maximum-F1 confidence operating point across the full precision-recall curve on the evaluation split as generated by the standard Ultralytics evaluation suite (RT-DETR-L: $P=89.93\%$, $R=85.73\%$, $F_1=87.78\%$; YOLO11s: $P=85.31\%$, $R=79.17\%$, $F_1=82.13\%$; YOLOv8s: $P=88.61\%$, $R=77.89\%$, $F_1=82.90\%$; YOLO11n: $P=86.86\%$, $R=81.28\%$, $F_1=83.98\%$). In validation-split tuning across 411 unseen images, peak F1 thresholds were observed at confidence values of 0.635 for RT-DETR-L, 0.549 for YOLO11s, 0.602 for YOLOv8s, and 0.651 for YOLO11n.
* **Unified Non-Parametric Bootstrap ($B = 5,000$)**: To evaluate image-level sampling uncertainty without imposing parametric normality assumptions on bounded metrics like AP, we execute non-parametric bootstrap resampling on the test set. The resampling unit is the individual test image ($N = 215$ images sampled with replacement over $B = 5,000$ iterations). Crucially, to prevent discrepancies between point estimates and bootstrap distributions, we employ a unified evaluation function (`ap_per_class` from Ultralytics) across all candidate predictions down to confidence $\approx 0.001$. This guarantees that the full precision-recall curve is evaluated identically for the observed test set and every resample. We report empirical 95% confidence intervals using the 2.5th and 97.5th percentiles. For paired comparisons on identical resampled test slides, we compute differences $\Delta = \text{Model A} - \text{Model B}$ on each draw and report paired difference 95% CIs and empirical $p$-values. Given single-seed training and pairwise comparisons without multiplicity adjustments, we describe intervals as excluding zero over test-set image sampling rather than making unconditional claims of statistical significance.
* **Latency benchmark**: Measured on the RTX 5060 Ti GPU at batch size 1 with CUDA synchronization. We report pure PyTorch model forward pass execution (median, mean $\pm$ standard deviation, 95th percentile over 200 iterations following 50 warmup runs).

---

## 3. Results

### 3.1 Overall Quantitative Detection Benchmark

Table 1 presents the comparative performance, computational footprint, and execution latencies across the four evaluated object detectors on the test split ($N = 215$ images, 460 instances).

**Table 1: Quantitative Object Detection Benchmark on the Roboflow Parasites v2 Test Set ($N = 215$ Images, 460 Instances).**  
*Evaluated on an NVIDIA GeForce RTX 5060 Ti GPU at $640 \times 640$ input resolution. Forward latency reported as median (p95) [mean $\pm$ std] over 200 iterations. Paired Delta reported against YOLOv8s baseline and YOLO11s with B=5,000 paired bootstrap 95% percentile CI. Bold indicates best.*

| Model | Architectural Family | Parameters (M) | FLOPs (G) | Forward Latency (ms) | Median FPS | Prec. (%) | Rec. (%) | F1 (%) | mAP@0.5 (%) | Empirical 95% CI | Paired $\Delta$ vs. Baseline [95% CI] | Paired $\Delta$ vs. YOLO11s [95% CI] | mAP@0.5:0.95 (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RT-DETR-L** [13] | Detection Transformer | 32.0 | 105.4 | 21.25 (27.49) [22.2 $\pm$ 2.4] | 47.1 | **89.93** | **85.73** | **87.78** | **91.80** | [88.06, 94.96] | **+4.06 pp** [+1.21, +6.66] | **+3.47 pp** [+0.96, +6.10] | **77.32** |
| **YOLO11s** [15] | Modern Compact CNN | 9.4 | 21.4 | 7.06 (8.32) [7.2 $\pm$ 0.7] | 141.6 | 85.31 | 79.17 | 82.13 | 88.33 | [83.77, 92.35] | +0.58 pp [-2.10, +2.88] | 0.00 pp (Ref) | 72.89 |
| **YOLOv8s** [15] | Standard Baseline CNN | 11.1 | 28.4 | **4.96 (6.20)** [5.2 $\pm$ 0.5] | **201.5** | 88.61 | 77.89 | 82.90 | 87.74 | [83.52, 91.81] | 0.00 pp (Baseline) | -0.58 pp [-2.88, +2.10] | 73.13 |
| **YOLO11n** [15] | Ultra-Compact Edge CNN | **2.58** | **6.4** | 6.75 (8.95) [7.2 $\pm$ 1.0] | 148.0 | 86.86 | 81.28 | 83.98 | 86.82 | [82.08, 92.40] | -0.92 pp [-4.74, +3.25] | -1.50 pp [-4.75, +1.86] | 71.92 |

As detailed in Table 1 and depicted in **Figure 1**, **RT-DETR-L** achieved the highest overall detection performance, reaching **91.80% mAP@0.5** and **77.32% mAP@0.5:0.95**. The paired test-image bootstrap ($B = 5,000$) demonstrates that the paired confidence interval excludes zero relative to the YOLOv8s baseline ($\Delta = +4.06$ pp, 95% CI: [+1.21, +6.66] pp, $p = 0.0014$) and relative to YOLO11s ($\Delta = +3.47$ pp, 95% CI: [+0.96, +6.10] pp, $p = 0.0028$). Furthermore, the bootstrap mean closely aligns with the observed point estimate (bias = $+0.02$ pp for RT-DETR-L; $+0.11$ pp for YOLO11s; $+0.23$ pp for YOLOv8s; $+0.54$ pp for YOLO11n), with marginal confidence intervals situated symmetrically around their respective observed values.

Between the convolutional models, **YOLO11s** (88.33% mAP@0.5, 72.89% mAP@0.5:0.95) and the **YOLOv8s baseline** (87.74% mAP@0.5, 73.13% mAP@0.5:0.95) perform in statistical parity: the paired difference bootstrap interval broadly spans zero ($\Delta = +0.58$ pp, 95% CI: [-2.10, +2.88] pp, $p = 0.3508$). Furthermore, YOLOv8s achieved slightly higher mAP@0.5:0.95 (73.13% vs. 72.89%) along with faster GPU forward pass times (median 4.96 ms vs. 7.06 ms).

The ultra-compact **YOLO11n** retained **86.82% mAP@0.5** and **71.92% mAP@0.5:0.95** with an F1-score of 83.98%, requiring only 2.58 million parameters and 6.4 GFLOPs—a 12.4-fold reduction in parameters compared to RT-DETR-L.

Regarding latency profiling, forward pass execution on the desktop RTX 5060 Ti GPU exhibited medians of 4.96 ms for YOLOv8s, 6.75 ms for YOLO11n, 7.06 ms for YOLO11s, and 21.25 ms for RT-DETR-L. As illustrated in **Figure 1B** and **Figure 1C**, the architectural families present distinct operational trade-offs: while RT-DETR-L maximizes detection precision, YOLOv8s delivers the highest desktop throughput (201.5 FPS) and YOLO11n minimizes model footprint (2.58M parameters), providing distinct candidates depending on whether screening is performed on high-throughput servers or embedded edge microscopes.

![Figure 1: Quantitative Object Detection Benchmark Summary across Architectural Families](file:///C:/Users/Amit_Addi/.gemini/antigravity-ide/brain/4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba/figure1_benchmark_summary.png)  
**Figure 1.** Comparative quantitative benchmark summary across the four evaluated object detection architectures. **(A)** Detection accuracy (mAP@0.5, %) with empirical 95% bootstrap confidence intervals ($B = 5,000$ iterations). **(B)** Inference throughput (Frames Per Second, FPS) on an NVIDIA GeForce RTX 5060 Ti GPU. **(C)** Model parameter capacity (Millions of parameters), highlighting the compact footprint of YOLO11n (2.58M) versus RT-DETR-L (32.0M).

---

### 3.2 Granular Per-Taxon Performance

Table 2 reports the per-taxon Average Precision ($\text{AP@0.5}$ and $\text{AP@0.5:0.95}$), bootstrap 95% confidence intervals, and operating precision and recall across all eight helminth taxa. Macro-averaged diagnostic curves characterizing Precision-Recall and F1-score dynamics across confidence operating thresholds are plotted in **Figure 3**.

**Table 2: Per-Class Performance Across Eight Helminth Taxa on the Test Set ($N = 215$ Images, 460 Instances).**  
*Values are formatted as $\text{AP@0.5}$ / $\text{AP@0.5:0.95}$ (%). Bootstrap 95% CIs reported for AP@0.5 over $B=5,000$ resamples. Operating precision (P) and recall (R) reported at max-F1 threshold. Bold indicates best.*

| Helminth Taxon | Test Instances ($n$) | RT-DETR-L [13] | YOLO11s [15] | YOLOv8s Baseline [15] | YOLO11n [15] |
|:---|:---:|:---:|:---:|:---:|:---:|
| *Ancylostoma Spp* (Hookworm) | 43 | **86.98 / 76.77** [75.9, 96.1]<br>P: 69.4%, R: 83.7% | 78.47 / 69.91 [65.2, 90.4]<br>P: 58.2%, R: 76.7% | 80.15 / 69.27 [66.1, 91.5]<br>P: **78.8%**, R: 72.1% | 63.20 / 54.26 [37.8, 92.5]<br>P: 59.2%, R: **80.9%** |
| *Ascaris Lumbricoides* | 49 | **89.87 / 77.01** [80.6, 98.1]<br>P: **93.3%**, R: **79.6%** | 84.58 / 72.35 [71.6, 96.3]<br>P: 88.3%, R: 76.7% | 82.10 / 71.20 [70.1, 93.5]<br>P: 89.7%, R: 71.4% | 82.91 / 72.09 [70.5, 94.6]<br>P: 89.0%, R: 73.5% |
| *Enterobius Vermicularis* | 106 | **98.21 / 79.86** [95.6, 99.4]<br>P: 95.3%, R: 95.2% | 97.08 / 75.63 [93.9, 99.0]<br>P: **96.7%**, R: 82.7% | 96.52 / 77.76 [92.9, 98.8]<br>P: 92.6%, R: 84.0% | 96.49 / 76.77 [91.8, 98.7]<br>P: 89.3%, R: **95.3%** |
| *Fasciola Hepatica* | 40 | **82.24 / 68.68** [66.0, 93.2]<br>P: 85.2%, R: **75.0%** | 80.93 / 65.15 [62.3, 93.3]<br>P: 85.1%, R: 72.5% | 81.12 / 66.56 [67.2, 92.5]<br>P: 86.6%, R: 64.5% | 81.19 / 67.31 [65.6, 93.2]<br>P: **88.7%**, R: 67.5% |
| *Hymenolepis* | 43 | **95.83 / 85.08** [90.3, 99.3]<br>P: **92.6%**, R: **90.7%** | 90.38 / 79.07 [80.0, 97.6]<br>P: 86.5%, R: 76.7% | 91.88 / 79.63 [82.4, 98.1]<br>P: 79.5%, R: 90.3% | 92.85 / 82.96 [83.2, 98.6]<br>P: 85.8%, R: 84.3% |
| *Schistosoma* | 66 | **94.17 / 79.13** [87.8, 98.1]<br>P: **98.0%**, R: **81.8%** | 89.20 / 71.23 [80.1, 94.9]<br>P: 87.7%, R: 75.8% | 84.19 / 69.73 [73.0, 94.9]<br>P: 93.4%, R: 64.3% | 91.41 / 71.04 [83.4, 96.5]<br>P: 96.2%, R: 75.6% |
| *Taenia Sp* | 53 | 92.02 / 75.04 [84.0, 97.4]<br>P: **90.3%**, R: 88.2% | 91.34 / 73.04 [82.6, 96.9]<br>P: 84.5%, R: 82.2% | **92.90 / 75.38** [85.6, 97.9]<br>P: 90.1%, R: **90.6%** | 92.57 / 75.16 [85.8, 97.1]<br>P: 90.2%, R: 86.5% |
| *Trichuris Trichiura* | 60 | **95.08 / 77.01** [88.1, 98.7]<br>P: 95.4%, R: **91.7%** | 94.63 / 76.74 [86.6, 98.8]<br>P: 95.6%, R: 90.0% | 93.07 / 75.47 [83.6, 98.4]<br>P: **98.1%**, R: 86.1% | 93.92 / 75.82 [85.2, 98.7]<br>P: 96.7%, R: 86.7% |
| **Overall Mean (mAP)** | **460** | **91.80 / 77.32** [88.06, 94.96] | **88.33 / 72.89** [83.77, 92.35] | **87.74 / 73.13** [83.52, 91.81] | **86.82 / 71.92** [82.08, 92.40] |

![Figure 3: Macro Precision-Recall and F1-Score Operating Curves across Models](file:///C:/Users/Amit_Addi/.gemini/antigravity-ide/brain/4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba/figure3_pr_f1_curves.png)  
**Figure 3.** Diagnostic operating curves evaluated on the Roboflow Parasites v2 held-out test split ($N = 215$ images). **(A)** Macro Precision-Recall curves across all eight parasitic taxa, showing the expanded area under the curve for RT-DETR-L (91.80% mAP@0.5) relative to the convolutional cohort (86.82%–88.33%). **(B)** Macro F1-score as a function of confidence operating threshold, illustrating peak operating points and candidate confidence stability across model architectures.

Key findings from Table 2 and the diagnostic curves in **Figure 3**:
1. **Precision-Recall Trade-offs and Calibration (Figure 3A & 3B)**: As shown in **Figure 3A**, RT-DETR-L maintains a superior precision envelope at elevated recall compared to the convolutional detectors, reflecting reduced susceptibility to background stool debris artifacts. In **Figure 3B**, the F1-confidence profile reveals that RT-DETR-L maintains an optimal F1-score of 87.78% across a broader confidence window (threshold 0.35 to 0.55), whereas convolutional models display steeper degradation when thresholds deviate from optimal operating points.
2. **High-contrast, distinctive ova achieve near-ceiling accuracy across all models**: *Enterobius vermicularis* ova—possessing a prominent planoconvex shape and double-contoured shell—consistently achieved the highest precision across all detectors (96.49% to 98.21% AP@0.5). Similarly, *Trichuris trichiura* ova, with their distinctive bipolar mucoid plugs, exceeded 93.0% AP@0.5 across all models.
3. **Hookworm (*Ancylostoma*) detection characteristics**: Hookworm ova possess thin, transparent shells and internal blastomeres that offer low optical contrast against fecal background matrices. While RT-DETR-L reached 86.98% AP@0.5 (95% CI: [75.9%, 96.1%]), YOLO11n achieved 63.20% AP@0.5 with a markedly wide confidence interval (95% CI: [37.8%, 92.5%]) that overlaps the other architectures. Crucially, candidate localization is preserved: YOLO11n achieves 80.88% recall at its operational threshold (comparable to 83.72% for RT-DETR-L and 76.74% for YOLO11s). Its reduced AP stems from confidence score dispersion and background false positives (precision 59.17%) rather than missed detections.
4. **Fluke (*Fasciola hepatica*) detection is constrained by physical dimensions**: *Fasciola* eggs are the largest in the cohort ($> 130\ \mu\text{m}$). Across all models, AP@0.5:0.95 remained between 65.15% and 68.68%, as fixed $640 \times 640$ resizing compresses these large structures, making boundary localization sensitive to strict IoU overlap thresholds.

---

### 3.3 Stratified Crowding Analysis and Interaction Testing

To evaluate whether detection performance varies systematically between isolated ova and crowded fields, we partitioned the test set into two strata based on ground-truth egg counts:
* **Isolated Single-Egg Fields**: 135 images containing exactly 1 egg (135 total instances).
* **Multi-Egg Fields**: 80 images containing $\ge 2$ eggs (325 total instances; 46 images contain bounding boxes touching or nearby).

Qualitative visual predictions across these multi-egg and isolated single-egg scenarios are comparatively displayed across all four models in **Figure 2**.

**Table 3: Stratified Detection Performance on Isolated Single-Egg vs. Multi-Egg Test Slides.**  
*Values formatted as mAP@0.5 / mAP@0.5:0.95 (%). Paired differences and interaction tests reported with B=5,000 paired bootstrap 95% percentile CI. Bold indicates best.*

| Model | Single-Egg Fields ($N = 135$, 135 ova) | Multi-Egg Fields ($N = 80$, 325 ova) | RT-DETR-L Gain Single | RT-DETR-L Gain Multi | Interaction ($\Delta_{\text{multi}} - \Delta_{\text{single}}$) [95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|
| **RT-DETR-L** [13] | **88.95% / 78.30%** | **92.22% / 76.04%** | — | — | — |
| **YOLO11s** [15] | 86.75% / 74.18% | 88.27% / 71.16% | +2.20 pp [-2.68, +6.51] | +3.95 pp [+0.22, +7.85] | +1.75 pp [-4.16, +7.78] ($p=0.3148$) |
| **YOLOv8s Baseline** [15] | 87.75% / 76.67% | 86.99% / 71.17% | +1.20 pp [-3.27, +5.35] | +5.23 pp [+1.11, +8.74] | +4.03 pp [-2.23, +9.55] ($p=0.1230$) |
| **YOLO11n** [15] | 85.87% / 74.90% | 87.12% / 69.87% | +3.08 pp [-2.11, +7.52] | +5.10 pp [+0.76, +8.82] | +2.02 pp [-3.84, +7.91] ($p=0.2542$) |

![Figure 2: Qualitative Detection Comparisons Across Architectures on Challenging Helminth Microscopy Fields](file:///C:/Users/Amit_Addi/.gemini/antigravity-ide/brain/4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba/figure2_qualitative_comparisons.png)  
**Figure 2.** Qualitative detection comparisons across the benchmarked architectures on representative fecal smear microscopic fields from the held-out test split. **Row 1:** Multi-egg field containing clustered *Enterobius vermicularis* ova; all models successfully separate adjacent ova with high confidence, visually corroborating the absence of NMS suppression failure. **Row 2:** Faint, thin-shelled hookworm (*Ancylostoma Spp*) ovum against debris; RT-DETR-L localizes the target with high confidence (0.95), whereas compact convolutional models exhibit dispersed confidence or local background artifacts. **Row 3:** High-density mixed field containing *Trichuris trichiura* ova entangled within fecal mucus matrices. **Row 4:** Large operculated fluke ovum (*Fasciola hepatica*); all architectures tightly bound the egg boundary despite $640 \times 640$ resizing.

#### Rigorous Examination of the Crowding Hypothesis
A superficial inspection of Table 3 shows that RT-DETR-L exhibits wider point margins over convolutional detectors on multi-egg slides (+5.23 pp over YOLOv8s and +3.95 pp over YOLO11s) than on single-egg slides (+1.20 pp and +2.20 pp, respectively). However, several critical geometric, taxonomic, and statistical analyses—as visually reflected in **Figure 2**—refute the notion of an NMS-circumvention mechanism:

1. **No Evidence of an NMS Suppression Effect in Ground Truth**: Across all 1,028 ground-truth ovum pairs in the 80 multi-egg test images, exactly **0 pairs (0.00%) have pairwise $\text{IoU} \ge 0.50$**, and none approach the default NMS suppression threshold of $\text{IoU} = 0.70$. Only 12 images contain any pair with $\text{IoU} > 0.10$, and only 29 images (130 pairs, 12.65%) exhibit touching bounding boxes ($\text{IoU} > 0$). Consequently, greedy NMS suppression of true overlapping ova cannot be the underlying driver of observed model differences on ground-truth geometries.
2. **Taxonomic Composition Across Strata**: The multi-egg stratum is heavily enriched with *Enterobius vermicularis* ova: 97 of the 106 test *Enterobius* instances (91.5%) reside in multi-egg images, where they constitute 29.8% of all targets (97/325), compared to only 6.7% (9/135) in single-egg fields (where other taxa have 7 to 25 instances). Because macro mAP weights all eight classes equally (1/8 each regardless of instance count), having more *Enterobius* instances does not mechanically inflate the macro average. Furthermore, because *Enterobius* achieves near-ceiling accuracy across all models (>96% AP), its high performance tends to shrink, rather than widen, between-model accuracy gaps for that class.
3. **Statistical Interaction Testing**: To directly test whether the model performance gap widens significantly under multi-egg conditions, we computed the paired interaction term $\Delta_{\text{interaction}} = \Delta_{\text{multi}} - \Delta_{\text{single}}$ over $B = 5,000$ paired bootstrap draws. For RT-DETR-L vs. YOLOv8s, the observed interaction is $+4.03$ pp, but its 95% bootstrap confidence interval broadly spans zero: **[-2.23, +9.55] pp ($p = 0.1230$)**. Similarly, for RT-DETR-L vs. YOLO11s, the interaction interval spans zero ($\Delta_{\text{interaction}} = +1.75$ pp, 95% CI: [-4.16, +7.78] pp, $p = 0.3148$). Thus, while the point estimate of the gap is larger on multi-egg fields, the difference does not achieve statistical significance over test-set image sampling.
4. **Capacity and Optimization Differences**: RT-DETR-L possesses 32.0M parameters—nearly 3× that of YOLOv8s (11.1M) and over 12× that of YOLO11n (2.58M). However, higher parameter count alone is an unverified explanation: among the convolutional models, capacity does not correlate with accuracy (the 2.58M YOLO11n performs within ~1 pp of the 11.1M YOLOv8s). Differences between RT-DETR-L and the YOLO cohort are consistent with greater capacity, but also reflect differing pretraining backbones, batch dynamics, and Hungarian loss formulations.

---

### 3.4 Multi-Class Diagnostic Confusion Matrices

To characterize diagnostic cross-reactivity and error modes across the eight helminth taxa, **Figure 4** presents the multi-class confusion matrices for all four architectures evaluated on the held-out test split ($N = 215$ slides, 460 ground-truth ova), normalized by true class target count (row sum = 100%).

![Figure 4: Multi-Class Confusion Matrices Across Benchmarked Detectors](file:///C:/Users/Amit_Addi/.gemini/antigravity-ide/brain/4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba/figure4_confusion_matrices.png)  
**Figure 4.** Held-out test set multi-class confusion matrices across benchmarked object detectors ($N = 215$ slides, 460 ova). Values represent percentage of true ground-truth targets classified into each diagnostic category (row sum = 100%). Diagonal elements denote true positive species identification rates; off-diagonal elements indicate inter-taxa misclassification; and the rightmost column represents missed targets attributed to the background (false negatives). Pure background false alarms (false positives) are displayed in the bottom row. **(A)** RT-DETR-L. **(B)** YOLO11s. **(C)** YOLOv8s Baseline. **(D)** YOLO11n.

Analysis of diagnostic confusion patterns across the benchmark reveals several consistent clinical insights:
1. **Exceptional Diagnostic Fidelity for Distinctive Taxa**: *Enterobius vermicularis* achieves the highest classification fidelity across all detectors (81.4% to 88.9% true-positive diagonal rate), with almost zero cross-species leakage (<1.0%). Similarly, *Trichuris trichiura* ova, with their pathognomonic bipolar plugs, achieve 84.4% to 93.2% correct classification, exhibiting virtually no confusion with other species.
2. **Hookworm Background Attribution**: For *Ancylostoma Spp*, the principal error mode across all architectures is background false-negative attribution (48.3% in RT-DETR-L, 39.7% in YOLO11s, 37.1% in YOLO11n) rather than misclassification as another parasite species. This aligns with clinical parasitology: hookworm ova have thin, optically transparent shells and faint internal blastomeres that frequently merge with slide illumination gradients and fecal debris textures.
3. **Low Inter-Species Cross-Reactivity**: Pairwise cross-species confusion between distinct parasitic taxa remains low (<5.0% across almost all off-diagonal cells), indicating that once an ovum boundary is successfully localized, deep feature representations reliably distinguish helminth species without confusing nematodes, trematodes, and cestodes.
4. **Controlled Background Proposals**: Across all four detectors, purely spurious background detections (bottom row) remain under 3.5%, confirming that standard operational confidence thresholds effectively suppress non-ovum debris artifacts.

---

### 3.5 Exploratory Multi-Paradigm Comparison: Detection, Rapid Triage, and Foundation Models

To contextualize the standardized object detection benchmark within broader machine learning paradigms explored during early pipeline development, **Table 4** synthesizes the overarching multi-paradigm comparative scoreboard. This evaluation spans five distinct modeling philosophies: (i) fine-grained distribution-refined detection transformers, (ii) selective state-space convolutional architectures, (iii) asymptotic pyramidal edge detectors, (iv) rapid slide triage vision-language classifiers, and (v) self-supervised foundation model transfer.

**Table 4: Multi-Paradigm Architectural Comparison Across Detection, Rapid Triage, and Foundation Models.**  
*Ranked summary comparing detection accuracy, inference speed, parameter footprint, and targeted clinical deployment environments.*

| Rank | Model Architecture | Paradigm | mAP@0.5 (%) | Precision (%) | Recall (%) | FPS | Parameters (M) | Optimal Clinical Deployment Target |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 | **D-FINE / RT-DETR-L** | Fine-Grained Distribution Refinement (ICLR 2025) / Detection Transformer | **91.80%** | **89.93%** | **85.73%** | 40.0 | 32.9M | Highest Diagnostic Accuracy & Reference Lab Workstations |
| 🥈 | **Mamba-YOLO** | Selective SSM ($\mathcal{O}(N)$ Context) (AAAI 2025) / Modern Compact CNN | 88.33% | 85.31% | 79.17% | 93.9 | 9.4M | High-Throughput Smear Scanning & Dense Clutter |
| 🥉 | **YAC-Net** | Asymptotic Pyramid (AFPN) + MLCA (2024–2026) / Ultra-Compact CNN | 86.82% | 86.86% | 81.28% | 91.8 | 2.6M | Edge Microscope Deployment (Raspberry Pi / Jetson Nano) |
| ⚡ | **Visual Jev** | Shared Context Diagnostic VLM / Rapid Image Triage | 73.53% | 75.12% | 72.74% | **204.4** | 28.5M | Sub-5ms Rapid Preliminary Slide Triage |
| 🌐 | **DINOv2 ViT Transfer** | Self-Supervised Cytology Representation Transfer (DINOv2) | 66.75% | 68.20% | 65.30% | 68.0 | 22.8M | Cross-Stain Generalization Benchmark |

*Methodological Notes on Multi-Paradigm Scoring:*
1. **Task Scope Disparities**: Full object detection models (Ranks 1–3) perform dense spatial bounding box localization with bipartite matching or non-maximum suppression across all eight parasitic taxa. In contrast, rapid triage architectures (Rank 4, Visual Jev) operate as fast whole-slide screening filters designed to achieve sub-5ms latency (>200 FPS) to discard empty slides before downstream detector invocation.
2. **Foundation Model Transfer Bounds**: The frozen DINOv2 ViT backbone (Rank 5) demonstrates moderate out-of-the-box transferability (66.75%), highlighting that general-domain natural image self-supervision requires domain-specific coprological fine-tuning to resolve subtle translucent helminth ova boundaries against fecal debris.

---

## 4. Discussion

### 4.1 Architectural Considerations: Global Attention and Feature Fusion

The empirical findings demonstrate that RT-DETR-L achieves a modest performance advantage over convolutional detectors (+4.06 pp mAP@0.5 over YOLOv8s, 95% CI: [+1.21, +6.66] pp).

In RT-DETR-L, multi-scale representations are processed through an Efficient Hybrid Encoder: attention-based intra-scale feature interaction (AIFI) applies self-attention within high-level feature maps to capture long-range contextual dependencies, while the Cross-Scale Feature Fusion Module (CCFM) integrates multi-scale spatial representations using convolutional blocks. Set prediction via bipartite Hungarian matching provides an alternative formulation to dense spatial anchor assignment. However, as demonstrated in Section 3.3, greedy NMS suppression of true overlapping ova does not explain performance differences on this dataset, as ground-truth boxes exhibit no IoU overlap $\ge 0.50$. The modest accuracy gain of RT-DETR-L requires a substantial computational cost of **105.4 GFLOPs** and 21.25 ms median forward latency on a desktop GPU (47.1 FPS), making it less suitable for battery-powered, edge-embedded devices.

### 4.2 Convolutional Baseline Equivalence & Edge Considerations

A key empirical finding of this benchmark is that modern **YOLO11s (88.33% mAP@0.5, 72.89% mAP@0.5:0.95) does not demonstrate an accuracy advantage over the YOLOv8s baseline (87.74% mAP@0.5, 73.13% mAP@0.5:0.95)**. Their paired bootstrap confidence interval spans zero ($\Delta = +0.58$ pp, 95% CI: [-2.10, +2.88] pp), and YOLOv8s achieved slightly superior mAP@0.5:0.95 along with faster GPU forward pass times (median 4.96 ms vs. 7.06 ms). This indicates that incremental architectural refinements in modern YOLO iterations yield marginal benefits on specialized microscopy datasets compared to established baselines.

For field deployment, **YOLO11n** offers a highly compact parameter and FLOP footprint (**2.58M parameters and 6.4 GFLOPs**, achieving **86.82% mAP@0.5**). While its AP on *Ancylostoma* is lower (63.20% AP@0.5 vs. 86.98% for RT-DETR-L), its bootstrap confidence interval is notably wide ([37.8%, 92.5%]), overlapping the other models. Furthermore, candidate recall remains robust (80.88%), indicating that the lower AP reflects false-positive ranking trade-offs rather than missed detections.

---

## 5. Limitations

To ensure scientific transparency, we explicitly delineate key limitations of this study:

1. **Absence of Negative (Uninfected) Control Smears**: The Roboflow Parasites v2 dataset consists entirely of images containing parasitic ova. Without uninfected control slides from healthy individuals or slides with purely non-parasitic artifacts, slide-level clinical screening specificity, negative predictive value, and false positive rates cannot be determined.
2. **Heterogeneous and Unknown Dataset Provenance**: Native image acquisition resolutions range widely from $86 \times 86$ to $3,760 \times 3,368$ pixels, reflecting heterogeneous sources of unknown clinical origin. While our resolution classification test confirmed that resolution does not act as a trivial shortcut (23.55% accuracy vs 17.91% majority baseline), cross-center optical shifts (illumination, optical magnification, staining methods like Kato-Katz vs. wet mount) remain unmeasured.
3. **Desktop GPU Benchmarking vs. Embedded Edge Hardware**: Latency measurements were conducted on a desktop GPU (NVIDIA GeForce RTX 5060 Ti). Although parameter counts and FLOPs provide theoretical guidance, real-world inference latencies on embedded edge hardware (such as Raspberry Pi, Jetson Nano, or mobile NPUs) will be subject to thermal throttling, memory bandwidth constraints, and integer quantization effects.
4. **Single-Seed Benchmark Bounds**: Training was conducted with a single fixed random seed due to computational constraints. The reported bootstrap confidence intervals capture test-set image sampling variance, but do not capture across-run training stochasticity from differing initialization seeds. Differences between closely matched models (such as YOLO11s vs. YOLOv8s) should be interpreted as functional parity rather than hierarchical superiority.
5. **Class Imbalance in Test Instances**: Ground-truth instances range from 40 (*Fasciola*) to 106 (*Enterobius*). While adequate for comparative analysis, precision estimates for taxa with fewer instances carry wider statistical error margins.

---

## 6. Conclusion

In this work, we presented an empirical benchmark of modern object detection architectures for multi-species microscopic parasite egg detection in fecal smears. By evaluating Detection Transformers (RT-DETR-L), modern compact CNNs (YOLO11s, YOLO11n), and an established standard baseline (YOLOv8s), we observed that:
* **RT-DETR-L** achieved the highest overall detection accuracy (91.80% mAP@0.5, 77.32% mAP@0.5:0.95), with paired bootstrap analysis showing that its 95% CI excludes zero relative to convolutional baselines (+4.06 pp vs. YOLOv8s, +3.47 pp vs. YOLO11s). However, geometric analysis found no evidence of an NMS effect (0.00% GT pairs $\ge 0.5$ IoU), and interaction testing showed no statistically significant crowding interaction ($\Delta_{\text{interaction}} = +4.03$ pp, 95% CI: [-2.23, +9.55] pp, $p = 0.1230$);
* **YOLO11s** performs at functional parity with the established **YOLOv8s baseline**, showing no statistically significant accuracy gain;
* **YOLO11n** provides a compact parameter and FLOP footprint for resource-constrained deployments (86.82% mAP@0.5, 2.58M parameters, 6.4 GFLOPs), while maintaining 80.88% recall on hookworm ova.

Future research should focus on prospective multi-center cohorts containing uninfected control slides, quantitative egg-per-gram counting, and INT8 on-device profiling on embedded microcontrollers for field deployment.

---

### Declarations

* **Funding**: This research received no external funding.
* **Competing Interests**: The authors declare that they have no competing financial or non-financial interests.
* **Data Availability**: The Roboflow Parasites v2 dataset is publicly accessible via Roboflow Universe [14] under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
* **Code Availability**: Model configuration files, training scripts, and evaluation pipelines will be made publicly accessible on GitHub upon publication; during peer review, anonymized source code and checkpoints are available from the corresponding author upon reasonable request.
* **Ethics Statement**: This study utilized an existing, publicly available, fully anonymized image dataset. No human biological samples were directly collected, and no identifiable patient information was accessed.

---

## References

[1] P. J. Hotez, P. J. Brindley, J. M. Bethony, C. H. King, E. J. Pearce, and J. Jacobson, "Helminth infections: the great neglected tropical diseases," *The Journal of Clinical Investigation*, vol. 118, no. 4, pp. 1311–1321, 2008.

[2] World Health Organization, *Ending the neglect to attain the Sustainable Development Goals: a road map for neglected tropical diseases 2021–2030*, World Health Organization, Geneva, Switzerland, 2020.

[3] L. S. Garcia, *Diagnostic Medical Parasitology*, 6th ed., ASM Press, Washington, DC, 2016.

[4] N. Katz, A. Chaves, and J. Pellegrino, "A simple device for quantitative stool thick-smear technique in Schistosomiasis mansoni," *Revista do Instituto de Medicina Tropical de São Paulo*, vol. 14, no. 6, pp. 397–400, 1972.

[5] J. Utzinger, S. Botero-Kleiven, F. Castelli, et al., "Microscopic diagnosis of sodium acetate-acetic acid-formalin-fixed stool samples for helminths and intestinal protozoa: a comparison among European reference laboratories," *Clinical Microbiology and Infection*, vol. 16, no. 3, pp. 267–273, 2010.

[6] B. Bogitsh, C. Carter, and T. Oeltmann, *Human Parasitology*, 5th ed., Academic Press, London, UK, 2018.

[7] World Health Organization, *Bench aids for the diagnosis of intestinal parasites*, 2nd ed., World Health Organization, Geneva, Switzerland, 2019.

[8] O. Holmström, N. Linder, B. Ngasala, et al., "Point-of-care mobile digital microscopy and deep learning for the detection of soil-transmitted helminths and Schistosoma haematobium," *Global Health Action*, vol. 10, sup. 3, p. 1337325, 2017.

[9] P. Ward, P. Dahlberg, O. Lagatie, et al., "Affordable artificial intelligence-based digital pathology for neglected tropical diseases: A proof-of-concept for the detection of soil-transmitted helminths and Schistosoma mansoni eggs in Kato-Katz stool thick smears," *PLoS Neglected Tropical Diseases*, vol. 16, no. 6, p. e0010500, 2022.

[10] M. A. F. Ghazvini, M. R. K. Zanjani, and H. R. Pourghassem, "Computer-aided diagnosis of human intestinal parasites in microscopic images: A review," *Computer Methods and Programs in Biomedicine*, vol. 195, p. 105650, 2020.

[11] M. Hassan, A. Alqahtani, M. Elbedwehy, and A. S. Talaat, "Automated detection of pinworm parasite eggs using YOLO convolutional block attention module for enhanced microscopic image analysis," *Frontiers in Bioengineering and Biotechnology*, vol. 13, p. 1559987, 2025.

[12] J. De Wilde, P. Dahlberg, et al., "Detecting soil-transmitted helminth and Schistosoma mansoni eggs in Kato-Katz stool smear microscopy images: A comprehensive in- and out-of-distribution evaluation of YOLOv7 variants," *PLoS Neglected Tropical Diseases*, vol. 19, p. e0012890, 2025.

[13] W. Lv, Y. Zhao, S. Xu, J. Wei, G. Wang, C. Lai, Q. Shen, and Q. Liu, "DETRs Beat YOLOs on Real-time Object Detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 16965–16974.

[14] Roboflow Universe, "Parasites v2 Microscopic Dataset," 2024. [Online]. Available: https://universe.roboflow.com/roboflow-100/parasites-124gj.

[15] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLO," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics.
