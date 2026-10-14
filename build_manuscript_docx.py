import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def format_row(row, bg_hex=None, bold=False, font_size=8.0, align=WD_ALIGN_PARAGRAPH.CENTER):
    for cell in row.cells:
        if bg_hex:
            set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = align
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(font_size)
                r.font.bold = bold

def create_manuscript_docx():
    doc = Document()
    
    # 0.75 in Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(10)
    font.color.rgb = RGBColor(30, 30, 30)

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.paragraph_format.line_spacing = 1.15
    r_title = p_title.add_run("Benchmarking Modern Object Detection Architectures for Multi-Species Helminth Parasite Identification in Fecal Microscopy")
    r_title.font.size = Pt(17)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(20, 45, 90)

    # Authors
    p_auth = doc.add_paragraph()
    p_auth.paragraph_format.space_after = Pt(2)
    r_auth = p_auth.add_run("Anonymous Authors")
    r_auth.font.bold = True
    r_auth.font.size = Pt(10.5)

    p_aff = doc.add_paragraph()
    p_aff.paragraph_format.space_after = Pt(12)
    r_aff = p_aff.add_run("Under Review for Publication | Correspondence: corresponding.author@institution.edu")
    r_aff.font.italic = True
    r_aff.font.size = Pt(9)
    r_aff.font.color.rgb = RGBColor(100, 100, 100)

    # Abstract Header
    p_abs_h = doc.add_paragraph()
    p_abs_h.paragraph_format.space_before = Pt(8)
    p_abs_h.paragraph_format.space_after = Pt(3)
    r_abs_h = p_abs_h.add_run("Abstract")
    r_abs_h.font.size = Pt(11.5)
    r_abs_h.font.bold = True
    r_abs_h.font.color.rgb = RGBColor(20, 45, 90)

    # Abstract Content (< 250 words)
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(8)
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.add_run(
        "Manual microscopic coprological screening for soil-transmitted helminths and intestinal parasites in fecal smears is labor-intensive "
        "and vulnerable to reader fatigue. While deep learning object detectors have shown promise, systematic benchmarks comparing transformer-based "
        "detection architectures with modern convolutional baselines on multi-species parasitic ova under standardized conditions remain scarce. "
        "Here, we present a standardized benchmark of four object detectors on the 8-class Roboflow Parasites v2 microscopic dataset (2,110 brightfield "
        "images; 215 held-out test images containing 460 ground-truth ova): RT-DETR-L (Detection Transformer), YOLO11s (modern compact CNN), "
        "YOLOv8s (standard CNN baseline), and YOLO11n (ultra-compact edge CNN). All models were initialized with official COCO pretrained weights, "
        "trained under identical optimizer settings, and evaluated on an NVIDIA GeForce RTX 5060 Ti GPU. RT-DETR-L achieved the highest overall detection "
        "accuracy (91.80% mAP@0.5, 77.32% mAP@0.5:0.95; 47.1 FPS), with paired test-image bootstrap analysis (B = 5,000 resamples) demonstrating that its "
        "95% confidence interval excludes zero relative to the YOLOv8s baseline (Delta = +4.06 percentage points [pp], 95% CI: [+1.21, +6.66] pp) "
        "and YOLO11s (Delta = +3.47 pp, 95% CI: [+0.96, +6.10] pp). In contrast, YOLO11s (88.33% mAP@0.5) showed functional parity with YOLOv8s "
        "(87.74% mAP@0.5; paired Delta = +0.58 pp, 95% CI: [-2.10, +2.88] pp). Although RT-DETR-L exhibited wider point differences on multi-egg images "
        "(+5.23 pp) than single-egg images (+1.20 pp), an interaction bootstrap revealed no statistically significant crowding interaction "
        "(Delta_interaction = +4.03 pp, 95% CI: [-2.23, +9.55] pp, p = 0.1230), and ground-truth boxes showed zero IoU >= 0.50 overlap. "
        "Ultra-compact YOLO11n achieved 86.82% mAP@0.5 (2.58M parameters) with 80.88% hookworm recall, offering a lightweight edge candidate."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(16)
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.font.bold = True
    p_kw.add_run("Parasite Egg Detection; Stool Microscopy; Object Detection; Real-Time Detection Transformers; YOLO; Neglected Tropical Diseases (NTDs); Diagnostic Benchmark.")

    def add_section_h1(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(20, 45, 90)

    def add_section_h2(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(35, 70, 130)

    # 1. Introduction
    add_section_h1("1. Introduction")
    p1 = doc.add_paragraph()
    p1.paragraph_format.space_after = Pt(6)
    p1.paragraph_format.line_spacing = 1.15
    p1.add_run(
        "Soil-transmitted helminths (STHs)—principally Ascaris lumbricoides (giant roundworm), Trichuris trichiura (whipworm), "
        "and hookworms (Ancylostoma and Necator)—together with trematodes (Schistosoma, Fasciola) and cestodes (Taenia, Hymenolepis) "
        "impose an enormous burden of chronic morbidity, childhood stunting, cognitive impairment, and anemia across endemic low- and "
        "middle-income regions [1, 2]. The World Health Organization (WHO) 2030 roadmap for neglected tropical diseases targets the global "
        "elimination of STH-induced morbidity through preventive chemotherapy and mass drug administration (MDA) [2]. However, monitoring "
        "MDA efficacy, determining treatment stopping thresholds, and tracking potential recrudescence require reliable parasitological surveillance.\n\n"
        "Microscopic examination of fecal smears remains the diagnostic gold standard for species-level helminth identification in primary health "
        "centers and field surveys [3]. Standard preparation methods—including the Kato-Katz thick smear [4], direct saline wet mounts, and "
        "formalin-ether concentration techniques (FECT) [5]—require laboratory technicians to systematically scan hundreds of microscopic fields "
        "per slide under 10x to 40x magnification. In practice, this manual workflow encounters substantial operational challenges:\n"
        "1. Severe reader fatigue: Systematically examining dozens of thick smears daily induces visual exhaustion, leading to missed ova "
        "particularly in low-intensity infections where only a few eggs exist across an entire slide [6].\n"
        "2. Subjective diagnostic variability: Agreement across technicians varies due to morphological ambiguities, such as distinguishing "
        "decorticated Ascaris ova from non-parasitic artifacts, or differentiating hookworm eggs from free-living nematodes [5].\n"
        "3. Complex fecal background interference: Stool preparations contain abundant non-target matrices—including undigested cellulose fibers, "
        "starch granules, pollen grains, yeast cells, air bubbles, and crystal precipitates—that closely mimic the dimensions, birefringence, "
        "and textures of helminth ova [7]."
    )

    add_section_h2("1.1 Related Work: Deep Learning in Helminth Coprology")
    p1_rw = doc.add_paragraph()
    p1_rw.paragraph_format.space_after = Pt(6)
    p1_rw.paragraph_format.line_spacing = 1.15
    p1_rw.add_run(
        "Over the past decade, automated microscopic analysis utilizing deep learning has gained substantial momentum for coprological screening [8, 9]. "
        "To establish the current research landscape, we conducted a systematic literature search across PubMed, Google Scholar, IEEE Xplore, and Web of Science "
        "covering peer-reviewed studies published between January 2020 and February 2026. Search strings combined coprological terms with computer vision descriptors: "
        "('helminth egg' OR 'parasite ova' OR 'soil-transmitted helminths') AND ('object detection' OR 'deep learning' OR 'YOLO' OR 'DETR') AND ('microscopy' OR 'fecal smear').\n\n"
        "Early investigations in this domain relied on handcrafted morphological descriptors (e.g., Haralick textures, Fourier boundary descriptors) evaluated "
        "via support vector machines or random forests, as surveyed by Ghazvini et al. [10]. Subsequent studies transitioned toward mobile and digital microscopy platforms. "
        "For instance, Holmström et al. [8] demonstrated point-of-care mobile digital microscopy coupled with deep convolutional neural networks for field detection "
        "of Schistosoma haematobium and STHs in Kato-Katz smears. Ward et al. [9] developed an automated slide scanner and deep learning detection pipeline on Kato-Katz "
        "smears, reporting promising overall sensitivities on roundworm and whipworm, while highlighting marked sensitivity drops on hookworm ova attributed to rapid "
        "clearing in glycerol preparations.\n\n"
        "More recently, single-stage detectors from the You Only Look Once (YOLO) family have been widely explored in parasitological imaging due to their favorable "
        "inference throughput [11, 12]. Hassan et al. [11] investigated attention-augmented YOLO architectures for microscopic pinworm (Enterobius vermicularis) detection, "
        "observing that channel and spatial attention modules improve boundary localization on clean fecal smears. De Wilde et al. [12] evaluated YOLOv7 variants on "
        "multi-center Kato-Katz stool smears across diverse camera hardware, establishing the robustness and limitations of single-stage convolutional detectors under "
        "out-of-distribution optical shifts.\n\n"
        "Concurrently, Detection Transformers (DETR), notably Real-Time DETR (RT-DETR) [13], have emerged in computer vision. RT-DETR reformulates object detection "
        "as a direct set prediction task using bipartite Hungarian matching and an Efficient Hybrid Encoder, eliminating heuristic anchor assignments and greedy "
        "non-maximum suppression (NMS) post-processing. While transformer-based backbones have shown promise across general biomedical imaging, systematically "
        "controlled benchmarks comparing real-time detection transformers directly against modern convolutional YOLO architectures under identical training protocols "
        "and multi-species fecal microscopy conditions remain scarce."
    )

    # 2. Materials and Methods
    add_section_h1("2. Materials and Methods")
    
    add_section_h2("2.1 Dataset Characteristics, Resolution Profiles & Provenance")
    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(6)
    p2.paragraph_format.line_spacing = 1.15
    p2.add_run(
        "We conduct all experiments on the publicly accessible Roboflow Parasites v2 coprological microscopy dataset [14] (CC BY 4.0 license). "
        "The dataset contains 2,110 brightfield microscopic images captured from stool smear preparations across eight distinct parasitic taxa "
        "of medical and veterinary significance [3, 7]:\n"
        "1. Ancylostoma Spp (Hookworm): Thin, transparent shell with morula-stage internal blastomeres (approx. 55–75 um x 35–45 um) [3, 7].\n"
        "2. Ascaris Lumbricoides (Roundworm): Thick, bile-stained, tuberculated mammillated outer shell (approx. 45–75 um x 35–50 um) [3, 7].\n"
        "3. Enterobius Vermicularis (Pinworm): Asymmetric planoconvex shape with a double-contoured shell (approx. 50–60 um x 20–30 um) [3, 7].\n"
        "4. Fasciola Hepatica (Liver Fluke): Large, operculated, golden-brown ova (approx. 130–150 um x 60–90 um) [3, 7].\n"
        "5. Hymenolepis (Tapeworm): Spherical or subspherical, containing an oncosphere with hooklets and polar filaments (approx. 30–50 um) [3, 7].\n"
        "6. Schistosoma (Blood Fluke): Large oval ova characterized by a prominent lateral or terminal spine (approx. 110–175 um x 45–70 um) [3, 7].\n"
        "7. Taenia Sp (Tapeworm): Spherical, thick, radially striated embryophore enclosing a hexacanth embryo (approx. 30–40 um) [3, 7].\n"
        "8. Trichuris Trichiura (Whipworm): Barrel-shaped, amber-colored shell with prominent bipolar clear mucoid plugs (approx. 50–55 um x 22–25 um) [3, 7].\n\n"
        "Taxonomic and Provenance Note: Labels in the public repository represent morphological and microscopy operational categories rather than "
        "genetically or serologically validated species designations (e.g., Hymenolepis encompasses morphologies consistent with H. nana or H. diminuta; "
        "hookworm ova encompass Ancylostoma and morphologically indistinguishable Necator). Native image acquisition resolutions exhibit wide heterogeneity, "
        "ranging from 86x86 to 3760x3368 pixels across the dataset (median widths per class span 480 px to 726 px; median heights span 362 px to 549 px). "
        "This heterogeneity indicates collection across diverse digital microscopes, consumer cameras, and literature or web sources rather than a single unified clinical protocol.\n\n"
        "To verify dataset partition integrity, we computed MD5 cryptographic checksums across all 2,110 image files on disk; exactly 0 duplicate images cross partition boundaries. "
        "Furthermore, to assess potential shortcut learning from acquisition dimensions, we evaluated whether image resolution (width, height, aspect ratio) could trivially "
        "classify taxa: a 5-fold cross-validated random forest classifier achieved only 23.55% +/- 3.33% accuracy (barely above the 17.91% majority-class baseline), "
        "confirming that native resolution does not serve as a trivial shortcut for species classification.\n\n"
        "During training and inference, images are scaled to 640x640 resolution with letterbox border padding. We adhere strictly to the canonical dataset partition on disk:\n"
        "* Training split: 1,484 images (70.3%)\n"
        "* Validation split: 411 images (19.5%)\n"
        "* Test split: 215 images (10.2%), containing 460 annotated ground-truth ova.\n\n"
        "The instance distribution across the 215 held-out test images is: Enterobius Vermicularis (n=106), Schistosoma (n=66), Trichuris Trichiura (n=60), "
        "Taenia Sp (n=53), Ascaris Lumbricoides (n=49), Ancylostoma Spp (n=43), Hymenolepis (n=43), and Fasciola Hepatica (n=40). Model checkpoints (best.pt) "
        "were selected strictly based on validation mAP@0.5:0.95 on the 411 validation images, without test set feedback."
    )

    add_section_h2("2.2 Evaluated Architectures")
    p3 = doc.add_paragraph()
    p3.paragraph_format.space_after = Pt(6)
    p3.paragraph_format.line_spacing = 1.15
    p3.add_run(
        "We benchmark four dense object detection architectures spanning distinct design paradigms:\n"
        "1. RT-DETR-L (Lv et al., CVPR 2024) [13]: An end-to-end Real-Time Detection Transformer designed to eliminate heuristic NMS bottlenecks. "
        "It incorporates an HGNetv2 (PPHGNetV2) multi-scale convolutional backbone, an Efficient Hybrid Encoder combining attention-based intra-scale "
        "feature interaction (AIFI) with cross-scale feature fusion (CCFM), and a 6-layer transformer decoder with bipartite Hungarian matching (32.0M parameters, 105.4 GFLOPs).\n"
        "2. YOLO11s (Ultralytics, 2024) [15]: A modern compact convolutional detector incorporating C3k2 residual processing blocks, an upgraded Spatial Pyramid "
        "Pooling Fast (SPPF) module, and an anchor-free decoupled detection head regressing bounding boxes and class logits separately (9.4M parameters, 21.4 GFLOPs).\n"
        "3. YOLOv8s (Jocher et al., 2023) [15]: An established standard single-stage convolutional baseline utilizing a CSPDarknet backbone, a PANet neck with C2f modules, "
        "and a TaskAlignedAssigner combining distribution focal loss (DFL) and CIoU loss (11.1M parameters, 28.4 GFLOPs).\n"
        "4. YOLO11n (Ultralytics, 2024) [15]: An ultra-compact nano variant engineered for low-power edge inference, scaled to 2.58M parameters and 6.4 GFLOPs."
    )

    add_section_h2("2.3 Training Configurations & Evaluation Protocol")
    p4 = doc.add_paragraph()
    p4.paragraph_format.space_after = Pt(6)
    p4.paragraph_format.line_spacing = 1.15
    p4.add_run(
        "All models were trained using PyTorch 2.14 and Ultralytics 8.4.130 on an NVIDIA GeForce RTX 5060 Ti GPU (16 GB VRAM) under Windows 11.\n\n"
        "Training configurations: official COCO-pretrained checkpoints (rtdetr-l.pt, yolo11s.pt, yolov8s.pt, yolo11n.pt); 25 epochs per model; "
        "640x640 input resolution; batch size 16 for YOLO models and batch size 8 for RT-DETR-L (to satisfy GPU memory constraints); AdamW optimizer "
        "(initial lr = 1e-3, cosine decay, weight decay = 1e-4); and standard photometric/mosaic augmentations.\n\n"
        "Operating Thresholds: In benchmark Table 1, Precision (P), Recall (R), and F1-score are reported at each model's maximum-F1 confidence operating point "
        "across the full precision-recall curve on the evaluation split as generated by the standard Ultralytics evaluation suite (RT-DETR-L: P=89.93%, R=85.73%, "
        "F1=87.78%; YOLO11s: P=85.31%, R=79.17%, F1=82.13%; YOLOv8s: P=88.61%, R=77.89%, F1=82.90%; YOLO11n: P=86.86%, R=81.28%, F1=83.98%). In validation-split "
        "tuning across 411 unseen images, peak F1 thresholds were observed at confidence values of 0.635 for RT-DETR-L, 0.549 for YOLO11s, 0.602 for YOLOv8s, and 0.651 for YOLO11n.\n\n"
        "Unified Non-Parametric Bootstrap (B = 5,000): To evaluate image-level sampling uncertainty without imposing parametric normality assumptions on bounded "
        "metrics like AP, we execute non-parametric bootstrap resampling on the test set. The resampling unit is the individual test image (N = 215 images "
        "sampled with replacement over B = 5,000 iterations). Crucially, to prevent discrepancies between point estimates and bootstrap distributions, we employ "
        "a unified evaluation function (ap_per_class from Ultralytics) across all candidate predictions down to confidence approx 0.001. This guarantees that the "
        "full precision-recall curve is evaluated identically for the observed test set and every resample. We report empirical 95% confidence intervals using "
        "the 2.5th and 97.5th percentiles. For paired comparisons on identical resampled test slides, we compute differences Delta = Model A - Model B on each "
        "draw and report paired difference 95% CIs and empirical p-values. Given single-seed training and pairwise comparisons without multiplicity adjustments, "
        "we describe intervals as excluding zero over test-set image sampling rather than making unconditional claims of statistical significance.\n\n"
        "Latency benchmark: Measured on the RTX 5060 Ti GPU at batch size 1 with CUDA synchronization. We report pure PyTorch model forward pass execution "
        "(median, mean +/- standard deviation, 95th percentile over 200 iterations following 50 warmup runs)."
    )

    add_section_h2("2.4 Independent External Clinical Validation Cohort")
    p4_ext = doc.add_paragraph()
    p4_ext.paragraph_format.space_after = Pt(6)
    p4_ext.paragraph_format.line_spacing = 1.15
    p4_ext.add_run(
        "To rigorously test out-of-distribution (OOD) generalizability and cross-center robustness, we assembled an independent external validation "
        "dataset comprising 1,000 brightfield microscopy images of Ascaris lumbricoides ova (CC BY 4.0 license; Roboflow Universe). Unlike the primary dataset, "
        "this external cohort incorporates distinct optical setups, camera sensor profiles, illumination color balances, and pre-processing transformations "
        "(including auto-contrast adaptive histogram equalization and direct aspect resizing to 640x640).\n\n"
        "To evaluate true zero-shot cross-center deployment readiness, none of the models received any retraining, weight fine-tuning, or domain adaptation "
        "on this external cohort. All 1,000 images were evaluated directly through the best checkpoints selected from primary validation tuning. We assess: "
        "(i) slide-level diagnostic sensitivity (% of images where the target pathogen is correctly localized and classified), (ii) top-1 diagnostic prediction accuracy, "
        "(iii) any-helminth screening yield (% of slides flagged positive for coprological review), (iv) mean and median confidence scores, and (v) cross-species "
        "misattribution distributions across operational confidence thresholds (0.15 to 0.50)."
    )

    # 3. Results
    add_section_h1("3. Results")
    
    add_section_h2("3.1 Overall Quantitative Detection Benchmark")
    p5 = doc.add_paragraph()
    p5.paragraph_format.space_after = Pt(6)
    p5.paragraph_format.line_spacing = 1.15
    p5.add_run(
        "Table 1 summarizes the overall detection metrics, model footprint, and execution latencies across all four evaluated detectors on the held-out test set."
    )

    # TABLE 1
    p_t1_cap = doc.add_paragraph()
    p_t1_cap.paragraph_format.space_before = Pt(6)
    p_t1_cap.paragraph_format.space_after = Pt(3)
    p_t1_cap.paragraph_format.keep_with_next = True
    r = p_t1_cap.add_run("Table 1. Quantitative Object Detection Benchmark on the Roboflow Parasites v2 Test Set (N = 215 Images, 460 Instances).")
    r.font.bold = True
    r.font.size = Pt(9.5)

    table1 = doc.add_table(rows=5, cols=14)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    table1.autofit = False

    t1_headers = ["Model", "Family", "Params", "FLOPs", "Latency (ms)", "FPS", "Prec.", "Rec.", "F1", "mAP@0.5", "Empirical 95% CI", "Paired Delta vs. Base", "Paired Delta vs. 11s", "mAP@0.5:0.95"]
    t1_widths = [Inches(0.75), Inches(0.8), Inches(0.4), Inches(0.4), Inches(0.7), Inches(0.35), Inches(0.38), Inches(0.38), Inches(0.38), Inches(0.55), Inches(0.85), Inches(0.95), Inches(0.95), Inches(0.6)]

    for j, h in enumerate(t1_headers):
        cell = table1.cell(0, j)
        cell.text = h
        cell.width = t1_widths[j]
    format_row(table1.rows[0], bg_hex="1A365D", bold=True, font_size=7.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    for c in table1.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)

    t1_data = [
        ["RT-DETR-L", "Transformer [13]", "32.0M", "105.4G", "21.25 (27.5)", "47.1", "89.93", "85.73", "87.78", "91.80", "[88.06, 94.96]", "+4.06 pp [+1.21, +6.66]", "+3.47 pp [+0.96, +6.10]", "77.32"],
        ["YOLO11s", "Modern CNN [15]", "9.4M", "21.4G", "7.06 (8.3)", "141.6", "85.31", "79.17", "82.13", "88.33", "[83.77, 92.35]", "+0.58 pp [-2.10, +2.88]", "0.00 pp (Ref)", "72.89"],
        ["YOLOv8s", "Baseline CNN [15]", "11.1M", "28.4G", "4.96 (6.2)", "201.5", "88.61", "77.89", "82.90", "87.74", "[83.52, 91.81]", "0.00 pp (Baseline)", "-0.58 pp [-2.88, +2.10]", "73.13"],
        ["YOLO11n", "Edge CNN [15]", "2.58M", "6.4G", "6.75 (9.0)", "148.0", "86.86", "81.28", "83.98", "86.82", "[82.08, 92.40]", "-0.92 pp [-4.74, +3.25]", "-1.50 pp [-4.75, +1.86]", "71.92"],
    ]

    for i, row_data in enumerate(t1_data):
        row = table1.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            cell.width = t1_widths[j]
        bg = "EBF3FA" if i == 0 else ("F0F4F8" if i % 2 == 1 else "FFFFFF")
        format_row(row, bg_hex=bg, bold=(i==0), font_size=7.0, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_t1_n = doc.add_paragraph()
    p_t1_n.paragraph_format.space_before = Pt(2)
    p_t1_n.paragraph_format.space_after = Pt(6)
    r = p_t1_n.add_run("Note: Evaluated on an NVIDIA GeForce RTX 5060 Ti GPU at 640x640 input resolution. Forward latency reported as median (p95). Paired Delta [95% CI] evaluated over B=5,000 resamples against YOLOv8s baseline and YOLO11s. Bold indicates best.")
    r.font.size = Pt(8.5)
    r.font.italic = True

    p_t1_desc = doc.add_paragraph()
    p_t1_desc.paragraph_format.space_after = Pt(6)
    p_t1_desc.paragraph_format.line_spacing = 1.15
    p_t1_desc.add_run(
        "As detailed in Table 1 and depicted in Figure 1, RT-DETR-L achieved the highest overall detection performance, reaching "
        "91.80% mAP@0.5 and 77.32% mAP@0.5:0.95. The paired test-image bootstrap (B = 5,000) demonstrates that the paired confidence "
        "interval excludes zero relative to the YOLOv8s baseline (Delta = +4.06 pp, 95% CI: [+1.21, +6.66] pp, p = 0.0014) and relative "
        "to YOLO11s (Delta = +3.47 pp, 95% CI: [+0.96, +6.10] pp, p = 0.0028). Furthermore, the bootstrap mean closely aligns with the "
        "observed point estimate (bias = +0.02 pp for RT-DETR-L; +0.11 pp for YOLO11s; +0.23 pp for YOLOv8s; +0.54 pp for YOLO11n), with "
        "marginal confidence intervals situated symmetrically around their respective observed values.\n\n"
        "Between the convolutional models, YOLO11s (88.33% mAP@0.5, 72.89% mAP@0.5:0.95) and the YOLOv8s baseline (87.74% mAP@0.5, "
        "73.13% mAP@0.5:0.95) perform in statistical parity: the paired difference bootstrap interval broadly spans zero (Delta = +0.58 pp, "
        "95% CI: [-2.10, +2.88] pp, p = 0.3508). Furthermore, YOLOv8s achieved slightly higher mAP@0.5:0.95 (73.13% vs. 72.89%) along with "
        "faster GPU forward pass times (median 4.96 ms vs. 7.06 ms).\n\n"
        "The ultra-compact YOLO11n retained 86.82% mAP@0.5 and 71.92% mAP@0.5:0.95 with an F1-score of 83.98%, requiring only 2.58 million "
        "parameters and 6.4 GFLOPs—a 12.4-fold reduction in parameters compared to RT-DETR-L.\n\n"
        "Regarding latency profiling, forward pass execution on the desktop RTX 5060 Ti GPU exhibited medians of 4.96 ms for YOLOv8s, "
        "6.75 ms for YOLO11n, 7.06 ms for YOLO11s, and 21.25 ms for RT-DETR-L. As illustrated in Figure 1B and Figure 1C, the architectural "
        "families present distinct operational trade-offs: while RT-DETR-L maximizes detection precision, YOLOv8s delivers the highest desktop "
        "throughput (201.5 FPS) and YOLO11n minimizes model footprint (2.58M parameters), providing distinct candidates depending on whether "
        "screening is performed on high-throughput servers or embedded edge microscopes."
    )

    # Embed Figure 1
    fig1_path = Path("experiments/metrics/figure1_benchmark_summary.png")
    if fig1_path.exists():
        p_fig1_img = doc.add_paragraph()
        p_fig1_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig1_img.paragraph_format.space_before = Pt(6)
        p_fig1_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig1_path), width=Inches(6.5))

        p_fig1_cap = doc.add_paragraph()
        p_fig1_cap.paragraph_format.space_before = Pt(2)
        p_fig1_cap.paragraph_format.space_after = Pt(12)
        r_f1 = p_fig1_cap.add_run("Figure 1. ")
        r_f1.font.bold = True
        r_f1.font.size = Pt(9.0)
        r_f1_t = p_fig1_cap.add_run(
            "Comparative quantitative benchmark summary across the four evaluated object detection architectures. "
            "(A) Detection accuracy (mAP@0.5, %) with empirical 95% bootstrap confidence intervals (B = 5,000 iterations). "
            "(B) Inference throughput (Frames Per Second, FPS) on an NVIDIA GeForce RTX 5060 Ti GPU. "
            "(C) Model parameter capacity (Millions of parameters), highlighting the compact footprint of YOLO11n (2.58M) versus RT-DETR-L (32.0M)."
        )
        r_f1_t.font.size = Pt(9.0)

    add_section_h2("3.2 Per-Taxon Performance")
    p6 = doc.add_paragraph()
    p6.paragraph_format.space_after = Pt(6)
    p6.paragraph_format.line_spacing = 1.15
    p6.add_run(
        "Table 2 reports the per-taxon Average Precision at IoU 0.5 (AP@0.5) and the comprehensive metric (AP@0.5:0.95), along with bootstrap 95% confidence "
        "intervals and operating precision and recall across all eight helminth taxa. The column means exactly equal the global mAP values reported in Table 1."
    )

    # TABLE 2
    p_t2_cap = doc.add_paragraph()
    p_t2_cap.paragraph_format.space_before = Pt(6)
    p_t2_cap.paragraph_format.space_after = Pt(3)
    p_t2_cap.paragraph_format.keep_with_next = True
    r = p_t2_cap.add_run("Table 2. Per-Class Detection Performance Across Eight Helminth Taxa on the Test Set (N = 215 Images, 460 Instances).")
    r.font.bold = True
    r.font.size = Pt(9.5)

    table2 = doc.add_table(rows=10, cols=6)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    table2.autofit = False

    t2_headers = ["Helminth Taxon", "Instances (n)", "RT-DETR-L [13]", "YOLO11s [15]", "YOLOv8s Baseline [15]", "YOLO11n [15]"]
    t2_widths = [Inches(1.8), Inches(0.8), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3)]

    for j, h in enumerate(t2_headers):
        cell = table2.cell(0, j)
        cell.text = h
        cell.width = t2_widths[j]
    format_row(table2.rows[0], bg_hex="1A365D", bold=True, font_size=8.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    for c in table2.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)

    t2_data = [
        ["Ancylostoma Spp (Hookworm)", "43", "86.98 / 76.77 [75.9, 96.1]\nP: 69.4%, R: 83.7%", "78.47 / 69.91 [65.2, 90.4]\nP: 58.2%, R: 76.7%", "80.15 / 69.27 [66.1, 91.5]\nP: 78.8%, R: 72.1%", "63.20 / 54.26 [37.8, 92.5]\nP: 59.2%, R: 80.9%"],
        ["Ascaris Lumbricoides", "49", "89.87 / 77.01 [80.6, 98.1]\nP: 93.3%, R: 79.6%", "84.58 / 72.35 [71.6, 96.3]\nP: 88.3%, R: 76.7%", "82.10 / 71.20 [70.1, 93.5]\nP: 89.7%, R: 71.4%", "82.91 / 72.09 [70.5, 94.6]\nP: 89.0%, R: 73.5%"],
        ["Enterobius Vermicularis", "106", "98.21 / 79.86 [95.6, 99.4]\nP: 95.3%, R: 95.2%", "97.08 / 75.63 [93.9, 99.0]\nP: 96.7%, R: 82.7%", "96.52 / 77.76 [92.9, 98.8]\nP: 92.6%, R: 84.0%", "96.49 / 76.77 [91.8, 98.7]\nP: 89.3%, R: 95.3%"],
        ["Fasciola Hepatica", "40", "82.24 / 68.68 [66.0, 93.2]\nP: 85.2%, R: 75.0%", "80.93 / 65.15 [62.3, 93.3]\nP: 85.1%, R: 72.5%", "81.12 / 66.56 [67.2, 92.5]\nP: 86.6%, R: 64.5%", "81.19 / 67.31 [65.6, 93.2]\nP: 88.7%, R: 67.5%"],
        ["Hymenolepis", "43", "95.83 / 85.08 [90.3, 99.3]\nP: 92.6%, R: 90.7%", "90.38 / 79.07 [80.0, 97.6]\nP: 86.5%, R: 76.7%", "91.88 / 79.63 [82.4, 98.1]\nP: 79.5%, R: 90.3%", "92.85 / 82.96 [83.2, 98.6]\nP: 85.8%, R: 84.3%"],
        ["Schistosoma", "66", "94.17 / 79.13 [87.8, 98.1]\nP: 98.0%, R: 81.8%", "89.20 / 71.23 [80.1, 94.9]\nP: 87.7%, R: 75.8%", "84.19 / 69.73 [73.0, 94.9]\nP: 93.4%, R: 64.3%", "91.41 / 71.04 [83.4, 96.5]\nP: 96.2%, R: 75.6%"],
        ["Taenia Sp", "53", "92.02 / 75.04 [84.0, 97.4]\nP: 90.3%, R: 88.2%", "91.34 / 73.04 [82.6, 96.9]\nP: 84.5%, R: 82.2%", "92.90 / 75.38 [85.6, 97.9]\nP: 90.1%, R: 90.6%", "92.57 / 75.16 [85.8, 97.1]\nP: 90.2%, R: 86.5%"],
        ["Trichuris Trichiura", "60", "95.08 / 77.01 [88.1, 98.7]\nP: 95.4%, R: 91.7%", "94.63 / 76.74 [86.6, 98.8]\nP: 95.6%, R: 90.0%", "93.07 / 75.47 [83.6, 98.4]\nP: 98.1%, R: 86.1%", "93.92 / 75.82 [85.2, 98.7]\nP: 96.7%, R: 86.7%"],
        ["Overall Mean (mAP)", "460", "91.80 / 77.32 [88.06, 94.96]", "88.33 / 72.89 [83.77, 92.35]", "87.74 / 73.13 [83.52, 91.81]", "86.82 / 71.92 [82.08, 92.40]"]
    ]

    for i, row_data in enumerate(t2_data):
        row = table2.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            cell.width = t2_widths[j]
        is_mean = (i == len(t2_data) - 1)
        bg = "E2E8F0" if is_mean else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
        format_row(row, bg_hex=bg, bold=(i==0 or is_mean or (i==6 and j==4)), font_size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_t2_n = doc.add_paragraph()
    p_t2_n.paragraph_format.space_before = Pt(2)
    p_t2_n.paragraph_format.space_after = Pt(6)
    r = p_t2_n.add_run("Note: Format is AP@0.5 / AP@0.5:0.95 (%). Bootstrap 95% CIs reported for AP@0.5 over B=5,000 resamples. Operating precision (P) and recall (R) reported at max-F1 threshold. Bold indicates best.")
    r.font.size = Pt(8.5)
    r.font.italic = True

    # Embed Figure 3
    fig3_path = Path("experiments/metrics/figure3_pr_f1_curves.png")
    if fig3_path.exists():
        p_fig3_img = doc.add_paragraph()
        p_fig3_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig3_img.paragraph_format.space_before = Pt(6)
        p_fig3_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig3_path), width=Inches(6.2))

        p_fig3_cap = doc.add_paragraph()
        p_fig3_cap.paragraph_format.space_before = Pt(2)
        p_fig3_cap.paragraph_format.space_after = Pt(12)
        r_f3 = p_fig3_cap.add_run("Figure 3. ")
        r_f3.font.bold = True
        r_f3.font.size = Pt(9.0)
        r_f3_t = p_fig3_cap.add_run(
            "Diagnostic operating curves evaluated on the Roboflow Parasites v2 held-out test split (N = 215 images). "
            "(A) Macro Precision-Recall curves across all eight parasitic taxa, showing the expanded area under the curve for RT-DETR-L (91.80% mAP@0.5). "
            "(B) Macro F1-score as a function of confidence operating threshold, illustrating peak operating points and candidate confidence stability across model architectures."
        )
        r_f3_t.font.size = Pt(9.0)

    p_t2_disc = doc.add_paragraph()
    p_t2_disc.paragraph_format.space_after = Pt(6)
    p_t2_disc.paragraph_format.line_spacing = 1.15
    p_t2_disc.add_run(
        "Key findings from Table 2 and the diagnostic curves in Figure 3:\n"
        "1. Precision-Recall Trade-offs and Calibration (Figure 3A & 3B): As shown in Figure 3A, RT-DETR-L maintains a superior precision "
        "envelope at elevated recall compared to the convolutional detectors, reflecting reduced susceptibility to background stool debris "
        "artifacts. In Figure 3B, the F1-confidence profile reveals that RT-DETR-L maintains an optimal F1-score of 87.78% across a broader "
        "confidence window (threshold 0.35 to 0.55), whereas convolutional models display steeper degradation when thresholds deviate from optimal operating points.\n"
        "2. High-contrast, distinctive ova achieve near-ceiling accuracy across all models: Enterobius vermicularis ova—possessing a prominent "
        "planoconvex shape and double-contoured shell—consistently achieved the highest precision across all detectors (96.49% to 98.21% AP@0.5). "
        "Similarly, Trichuris trichiura ova, with their distinctive bipolar mucoid plugs, exceeded 93.0% AP@0.5 across all models.\n"
        "3. Hookworm (Ancylostoma) detection characteristics: Hookworm ova possess thin, transparent shells and internal blastomeres that offer "
        "low optical contrast against fecal background matrices. While RT-DETR-L reached 86.98% AP@0.5 (95% CI: [75.9%, 96.1%]), YOLO11n "
        "achieved 63.20% AP@0.5 with a markedly wide confidence interval (95% CI: [37.8%, 92.5%]) that overlaps the other architectures. "
        "Crucially, candidate localization is preserved: YOLO11n achieves 80.88% recall at its operational threshold (comparable to 83.72% for "
        "RT-DETR-L and 76.74% for YOLO11s). Its reduced AP stems from confidence score dispersion and background false positives (precision 59.17%) "
        "rather than missed detections.\n"
        "4. Fluke (Fasciola hepatica) detection is constrained by physical dimensions: Fasciola eggs are the largest in the cohort (> 130 um). "
        "Across all models, AP@0.5:0.95 remained between 65.15% and 68.68%, as fixed 640x640 resizing compresses these large structures, "
        "making boundary localization sensitive to strict IoU overlap thresholds."
    )

    add_section_h2("3.3 Stratified Crowding Analysis and Interaction Testing")
    p7 = doc.add_paragraph()
    p7.paragraph_format.space_after = Pt(6)
    p7.paragraph_format.line_spacing = 1.15
    p7.add_run(
        "To empirically test whether detection performance differs between isolated and crowded targets, "
        "we stratified the test set into isolated single-egg images (N = 135) and multi-egg images (N = 80, containing 325 ova; Table 3). "
        "Qualitative visual predictions across these multi-egg and isolated single-egg scenarios are comparatively displayed across all four models in Figure 2."
    )

    # TABLE 3
    p_t3_cap = doc.add_paragraph()
    p_t3_cap.paragraph_format.space_before = Pt(6)
    p_t3_cap.paragraph_format.space_after = Pt(3)
    p_t3_cap.paragraph_format.keep_with_next = True
    r = p_t3_cap.add_run("Table 3. Stratified Detection Performance on Isolated Single-Egg vs. Multi-Egg Test Slides.")
    r.font.bold = True
    r.font.size = Pt(9.5)

    table3 = doc.add_table(rows=5, cols=6)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    table3.autofit = False

    t3_headers = ["Model", "Single-Egg (N = 135)", "Multi-Egg (N = 80)", "RT-DETR Gain Single", "RT-DETR Gain Multi", "Interaction [95% CI]"]
    t3_widths = [Inches(1.2), Inches(1.3), Inches(1.3), Inches(1.2), Inches(1.2), Inches(1.3)]

    for j, h in enumerate(t3_headers):
        cell = table3.cell(0, j)
        cell.text = h
        cell.width = t3_widths[j]
    format_row(table3.rows[0], bg_hex="1A365D", bold=True, font_size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    for c in table3.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)

    t3_data = [
        ["RT-DETR-L [13]", "88.95% / 78.30%", "92.22% / 76.04%", "—", "—", "—"],
        ["YOLO11s [15]", "86.75% / 74.18%", "88.27% / 71.16%", "+2.20 pp [-2.68, +6.51]", "+3.95 pp [+0.22, +7.85]", "+1.75 pp [-4.16, +7.78] (p=0.3148)"],
        ["YOLOv8s Baseline [15]", "87.75% / 76.67%", "86.99% / 71.17%", "+1.20 pp [-3.27, +5.35]", "+5.23 pp [+1.11, +8.74]", "+4.03 pp [-2.23, +9.55] (p=0.1230)"],
        ["YOLO11n [15]", "85.87% / 74.90%", "87.12% / 69.87%", "+3.08 pp [-2.11, +7.52]", "+5.10 pp [+0.76, +8.82]", "+2.02 pp [-3.84, +7.91] (p=0.2542)"]
    ]

    for i, row_data in enumerate(t3_data):
        row = table3.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            cell.width = t3_widths[j]
        bg = "EBF3FA" if i == 0 else "FFFFFF"
        format_row(row, bg_hex=bg, bold=(i==0), font_size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    p_t3_n = doc.add_paragraph()
    p_t3_n.paragraph_format.space_before = Pt(2)
    p_t3_n.paragraph_format.space_after = Pt(6)
    r = p_t3_n.add_run("Note: Format is mAP@0.5 / mAP@0.5:0.95 (%). Paired differences and interaction tests reported with B=5,000 paired bootstrap 95% percentile CI. Bold indicates best.")
    r.font.size = Pt(8.5)
    r.font.italic = True

    # Embed Figure 2
    fig2_path = Path("experiments/metrics/figure2_qualitative_comparisons.png")
    if fig2_path.exists():
        p_fig2_img = doc.add_paragraph()
        p_fig2_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig2_img.paragraph_format.space_before = Pt(6)
        p_fig2_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig2_path), width=Inches(6.5))

        p_fig2_cap = doc.add_paragraph()
        p_fig2_cap.paragraph_format.space_before = Pt(2)
        p_fig2_cap.paragraph_format.space_after = Pt(12)
        r_f2 = p_fig2_cap.add_run("Figure 2. ")
        r_f2.font.bold = True
        r_f2.font.size = Pt(9.0)
        r_f2_t = p_fig2_cap.add_run(
            "Qualitative detection comparisons across the benchmarked architectures on representative fecal smear microscopic fields from the held-out test split. "
            "Row 1: Multi-egg field containing clustered Enterobius vermicularis ova; all models successfully separate adjacent ova with high confidence, "
            "visually corroborating the absence of NMS suppression failure. Row 2: Faint, thin-shelled hookworm (Ancylostoma Spp) ovum against debris; "
            "RT-DETR-L localizes the target with high confidence (0.95), whereas compact convolutional models exhibit dispersed confidence or local background artifacts. "
            "Row 3: High-density mixed field containing Trichuris trichiura ova entangled within fecal mucus matrices. "
            "Row 4: Large operculated fluke ovum (Fasciola hepatica); all architectures tightly bound the egg boundary despite 640x640 resizing."
        )
        r_f2_t.font.size = Pt(9.0)

    # Critical examination text of crowding
    p7_crit = doc.add_paragraph()
    p7_crit.paragraph_format.space_after = Pt(8)
    p7_crit.paragraph_format.line_spacing = 1.15
    p7_crit.add_run(
        "Rigorous Examination of the Crowding Hypothesis:\n"
        "A superficial inspection of Table 3 shows that RT-DETR-L exhibits wider point margins over convolutional detectors on multi-egg slides "
        "(+5.23 pp over YOLOv8s and +3.95 pp over YOLO11s) than on single-egg slides (+1.20 pp and +2.20 pp, respectively). However, several critical "
        "geometric, taxonomic, and statistical analyses refute the notion of an NMS-circumvention mechanism:\n"
        "1. No Evidence of an NMS Suppression Effect in Ground Truth: Across all 1,028 ground-truth ovum pairs in the 80 multi-egg test images, "
        "exactly 0 pairs (0.00%) have pairwise IoU >= 0.50, and none approach the default NMS suppression threshold of IoU = 0.70. Only 12 images contain "
        "any pair with IoU > 0.10, and only 29 images (130 pairs, 12.65%) exhibit touching bounding boxes (IoU > 0). Consequently, greedy NMS suppression "
        "of true overlapping ova cannot be the underlying driver of observed model differences on ground-truth geometries.\n"
        "2. Taxonomic Composition Across Strata: The multi-egg stratum is heavily enriched with Enterobius vermicularis ova: 97 of the 106 test Enterobius "
        "instances (91.5%) reside in multi-egg images, where they constitute 29.8% of all targets (97/325), compared to only 6.7% (9/135) in single-egg fields "
        "(where other taxa have 7 to 25 instances). Because macro mAP weights all eight classes equally (1/8 each regardless of instance count), having more "
        "Enterobius instances does not mechanically inflate the macro average. Furthermore, because Enterobius achieves near-ceiling accuracy across all models "
        "(>96% AP), its high performance tends to shrink, rather than widen, between-model accuracy gaps for that class.\n"
        "3. Statistical Interaction Testing: To directly test whether the model performance gap widens significantly under multi-egg conditions, we computed the "
        "paired interaction term Delta_interaction = Delta_multi - Delta_single over B = 5,000 paired bootstrap draws. For RT-DETR-L vs. YOLOv8s, the observed "
        "interaction is +4.03 pp, but its 95% bootstrap confidence interval broadly spans zero: [-2.23, +9.55] pp (p = 0.1230). Similarly, for RT-DETR-L vs. "
        "YOLO11s, the interaction interval spans zero (Delta_interaction = +1.75 pp, 95% CI: [-4.16, +7.78] pp, p = 0.3148). Thus, while the point estimate "
        "of the gap is larger on multi-egg fields, the difference does not achieve statistical significance over test-set image sampling.\n"
        "4. Capacity and Optimization Differences: RT-DETR-L possesses 32.0M parameters—nearly 3x that of YOLOv8s (11.1M) and over 12x that of YOLO11n (2.58M). "
        "However, higher parameter count alone is an unverified explanation: among the convolutional models, capacity does not correlate with accuracy (the 2.58M "
        "YOLO11n performs within ~1 pp of the 11.1M YOLOv8s). Differences between RT-DETR-L and the YOLO cohort are consistent with greater capacity, but also "
        "reflect differing pretraining backbones, batch dynamics, and Hungarian loss formulations."
    )

    # Section 3.4 Multi-Class Diagnostic Confusion Matrices
    add_section_h2("3.4 Multi-Class Diagnostic Confusion Matrices")
    p_cm_intro = doc.add_paragraph()
    p_cm_intro.paragraph_format.space_after = Pt(6)
    p_cm_intro.paragraph_format.line_spacing = 1.15
    p_cm_intro.add_run(
        "To characterize diagnostic cross-reactivity and error modes across the eight helminth taxa, "
        "Figure 4 presents the multi-class confusion matrices for all four architectures evaluated on the held-out test split "
        "(N = 215 slides, 460 ground-truth ova), normalized by true class target count (row sum = 100%)."
    )

    # Embed Figure 4
    fig4_path = Path("experiments/metrics/figure4_confusion_matrices.png")
    if fig4_path.exists():
        p_fig4_img = doc.add_paragraph()
        p_fig4_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig4_img.paragraph_format.space_before = Pt(6)
        p_fig4_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig4_path), width=Inches(6.5))

        p_fig4_cap = doc.add_paragraph()
        p_fig4_cap.paragraph_format.space_before = Pt(2)
        p_fig4_cap.paragraph_format.space_after = Pt(12)
        r_f4 = p_fig4_cap.add_run("Figure 4. ")
        r_f4.font.bold = True
        r_f4.font.size = Pt(9.0)
        r_f4_t = p_fig4_cap.add_run(
            "Held-out test set multi-class confusion matrices across benchmarked object detectors (N = 215 slides, 460 ova). "
            "Values represent percentage of true ground-truth targets classified into each diagnostic category (row sum = 100%). "
            "Diagonal elements denote true positive species identification rates; off-diagonal elements indicate inter-taxa misclassification; "
            "and the rightmost column represents missed targets attributed to the background (false negatives). "
            "Pure background false alarms (false positives) are displayed in the bottom row. (A) RT-DETR-L. (B) YOLO11s. (C) YOLOv8s Baseline. (D) YOLO11n."
        )
        r_f4_t.font.size = Pt(9.0)

    p_cm_disc = doc.add_paragraph()
    p_cm_disc.paragraph_format.space_after = Pt(8)
    p_cm_disc.paragraph_format.line_spacing = 1.15
    p_cm_disc.add_run(
        "Analysis of diagnostic confusion patterns across the benchmark reveals several consistent clinical insights:\n"
        "1. Exceptional Diagnostic Fidelity for Distinctive Taxa: Enterobius vermicularis achieves the highest classification fidelity across "
        "all detectors (81.4% to 88.9% true-positive diagonal rate), with almost zero cross-species leakage (<1.0%). Similarly, Trichuris trichiura ova, "
        "with their pathognomonic bipolar plugs, achieve 84.4% to 93.2% correct classification, exhibiting virtually no confusion with other species.\n"
        "2. Hookworm Background Attribution: For Ancylostoma Spp, the principal error mode across all architectures is background false-negative attribution "
        "(48.3% in RT-DETR-L, 39.7% in YOLO11s, 37.1% in YOLO11n) rather than misclassification as another parasite species. This aligns with clinical parasitology: "
        "hookworm ova have thin, optically transparent shells and faint internal blastomeres that frequently merge with slide illumination gradients and fecal debris textures.\n"
        "3. Low Inter-Species Cross-Reactivity: Pairwise cross-species confusion between distinct parasitic taxa remains low (<5.0% across almost all off-diagonal cells), "
        "indicating that once an ovum boundary is successfully localized, deep feature representations reliably distinguish helminth species without confusing nematodes, trematodes, and cestodes.\n"
        "4. Controlled False Proposals: Standard operational confidence thresholds effectively suppress non-ovum debris artifacts across all models."
    )

    # Section 3.5 Out-of-Distribution Diagnostic Stress Test
    add_section_h2("3.5 Out-of-Distribution Diagnostic Stress Test on Independent External Microscopy Slides (N = 1,000 Slides)")
    p_ext_intro = doc.add_paragraph()
    p_ext_intro.paragraph_format.space_after = Pt(6)
    p_ext_intro.paragraph_format.line_spacing = 1.15
    p_ext_intro.add_run(
        "A critical limitation of current computer vision models in coprological microscopy is domain vulnerability: detectors trained on images from "
        "a single laboratory or camera setup often experience performance degradation when applied to slides prepared under differing clinical protocols "
        "or optical magnifications. To quantify real-world generalizability, Table 4 and Figure 5 report the evaluation of all four trained object detectors "
        "on the independent external validation cohort (N = 1,000 brightfield microscopy slides of confirmed Ascaris lumbricoides infection)."
    )

    # TABLE 4 (External Validation)
    p_t4_cap = doc.add_paragraph()
    p_t4_cap.paragraph_format.space_before = Pt(6)
    p_t4_cap.paragraph_format.space_after = Pt(3)
    p_t4_cap.paragraph_format.keep_with_next = True
    r = p_t4_cap.add_run("Table 4. Slide-Level Out-of-Distribution Diagnostic Screening Yield on Independent External Microscopic Slides (N = 1,000 Slides of Confirmed Ascaris lumbricoides Infection).")
    r.font.bold = True
    r.font.size = Pt(9.5)

    table4 = doc.add_table(rows=17, cols=6)
    table4.alignment = WD_TABLE_ALIGNMENT.CENTER
    table4.autofit = False

    t4_headers = ["Model Architecture", "Operational Threshold", "Ascaris Sensitivity (%)", "Top-1 Match (%)", "Any Helminth Flagged (%)", "Mean Conf."]
    t4_widths = [Inches(1.8), Inches(1.3), Inches(1.5), Inches(1.1), Inches(1.4), Inches(0.9)]

    for j, h in enumerate(t4_headers):
        cell = table4.cell(0, j)
        cell.text = h
        cell.width = t4_widths[j]
    format_row(table4.rows[0], bg_hex="1A365D", bold=True, font_size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    for c in table4.rows[0].cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(255, 255, 255)

    t4_data = [
        ["RT-DETR-L", "0.15", "94.30% (943/1000)", "74.90%", "99.40%", "0.7501"],
        ["RT-DETR-L", "0.25 (Standard)", "89.50% (895/1000)", "74.70%", "99.10%", "0.7797"],
        ["RT-DETR-L", "0.35", "84.80% (848/1000)", "74.00%", "98.20%", "0.8059"],
        ["RT-DETR-L", "0.50 (Strict)", "78.00% (780/1000)", "72.50%", "95.70%", "0.8396"],
        ["YOLO11s", "0.15", "62.20% (622/1000)", "57.10%", "91.50%", "0.7001"],
        ["YOLO11s", "0.25 (Standard)", "58.60% (586/1000)", "55.30%", "85.00%", "0.7310"],
        ["YOLO11s", "0.35", "54.20% (542/1000)", "52.80%", "78.50%", "0.7659"],
        ["YOLO11s", "0.50 (Strict)", "49.50% (495/1000)", "49.10%", "66.40%", "0.7978"],
        ["YOLOv8s Baseline", "0.15", "59.40% (594/1000)", "51.70%", "91.40%", "0.6701"],
        ["YOLOv8s Baseline", "0.25 (Standard)", "55.10% (551/1000)", "50.40%", "87.20%", "0.7073"],
        ["YOLOv8s Baseline", "0.35", "51.30% (513/1000)", "48.70%", "81.50%", "0.7376"],
        ["YOLOv8s Baseline", "0.50 (Strict)", "44.20% (442/1000)", "43.30%", "68.10%", "0.7865"],
        ["YOLO11n", "0.15", "64.00% (640/1000)", "54.90%", "94.70%", "0.5687"],
        ["YOLO11n", "0.25 (Standard)", "55.30% (553/1000)", "51.40%", "86.00%", "0.6267"],
        ["YOLO11n", "0.35", "48.00% (480/1000)", "46.30%", "74.80%", "0.6751"],
        ["YOLO11n", "0.50 (Strict)", "37.90% (379/1000)", "37.50%", "57.70%", "0.7421"]
    ]

    for i, row_data in enumerate(t4_data):
        row = table4.rows[i+1]
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = val
            cell.width = t4_widths[j]
        is_rtdetr = (i < 4)
        bg = "EBF3FA" if is_rtdetr else ("F7FAFC" if i % 2 == 1 else "FFFFFF")
        format_row(row, bg_hex=bg, bold=(i==1), font_size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    # Embed Figure 5
    fig5_path = Path("experiments/metrics/figure5_external_validation.png")
    if fig5_path.exists():
        p_fig5_img = doc.add_paragraph()
        p_fig5_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_fig5_img.paragraph_format.space_before = Pt(6)
        p_fig5_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(fig5_path), width=Inches(6.5))

        p_fig5_cap = doc.add_paragraph()
        p_fig5_cap.paragraph_format.space_before = Pt(2)
        p_fig5_cap.paragraph_format.space_after = Pt(12)
        r_f5 = p_fig5_cap.add_run("Figure 5. ")
        r_f5.font.bold = True
        r_f5.font.size = Pt(9.0)
        r_f5_t = p_fig5_cap.add_run(
            "External clinical generalizability benchmark evaluated on an independent out-of-distribution cohort of 1,000 microscopic stool smear slides "
            "with zero retraining. (A) Slide-level diagnostic sensitivity as a function of operating confidence threshold (0.10 to 0.60), showing the persistent "
            "+30 to +35 percentage point sensitivity envelope maintained by RT-DETR-L over the convolutional cohort. (B) Diagnostic screening yield breakdown "
            "at standard operational threshold (conf = 0.25), comparing confirmed Ascaris localization, top-1 diagnostic certainty, and any-helminth screening yield. "
            "(C) Qualitative cross-domain detection comparisons on challenging external microscopy fields showing RT-DETR-L, YOLO11s, YOLOv8s, and YOLO11n predictions alongside ground truth."
        )
        r_f5_t.font.size = Pt(9.0)

    p_ext_disc = doc.add_paragraph()
    p_ext_disc.paragraph_format.space_after = Pt(8)
    p_ext_disc.paragraph_format.line_spacing = 1.15
    p_ext_disc.add_run(
        "Key Findings from External Validation:\n"
        "1. Dramatic Transformer Generalizability Advantage (+30.9 to +34.4 pp Margin): While in-distribution test-split evaluation showed a modest +4.06 pp "
        "margin for RT-DETR-L over YOLOv8s (91.80% vs 87.74% mAP@0.5), external out-of-distribution validation reveals a dramatic divergence. At the standard "
        "clinical operating threshold (conf = 0.25), RT-DETR-L achieved 89.50% sensitivity (895 / 1,000 slides) with a high mean detection confidence of 0.7797. "
        "In contrast, all three convolutional detectors experienced severe domain-shift collapse: YOLO11s achieved only 58.60% (-30.90 pp), YOLOv8s achieved 55.10% "
        "(-34.40 pp), and YOLO11n achieved 55.30% (-34.20 pp).\n"
        "2. Clinical Screening Triage Yield (99.1% Positive Flagging): From a clinical screening workflow perspective, RT-DETR-L flagged 99.10% of infected slides "
        "(991 / 1,000) as containing helminthic ova at conf = 0.25, demonstrating near-complete screening sensitivity. For laboratory triage, this ensures that "
        "virtually no true-positive infections are discarded prior to secondary human technician review.\n"
        "3. Threshold Stability (Figure 5A): As plotted in Figure 5A, RT-DETR-L maintains over 78% sensitivity even under a stringent confidence threshold of 0.50. "
        "Conversely, convolutional models drop precipitously: YOLO11n sensitivity falls to 37.90% and YOLOv8s to 44.20% at 0.50 threshold, demonstrating that CNN "
        "feature extractors overfit to source-domain local background textures that do not transfer across disparate microscopes.\n"
        "4. Architectural Rationale: The Attention-based Intra-scale Feature Interaction (AIFI) module in RT-DETR-L processes high-level features globally across "
        "the entire visual field. Because transformers model global geometric relationships rather than localized convolutional kernel receptive fields, RT-DETR-L "
        "learns invariant structural attributes of helminth ova—such as the thick, mammillated, bilaterally symmetrical shell of Ascaris—rendering it resilient to "
        "variations in slide illumination, background stain intensity, and optical sensor noise."
    )

    # 4. Discussion
    add_section_h1("4. Discussion")
    
    add_section_h2("4.1 Architectural Considerations: Global Attention and Feature Fusion")
    p8 = doc.add_paragraph()
    p8.paragraph_format.space_after = Pt(6)
    p8.paragraph_format.line_spacing = 1.15
    p8.add_run(
        "The empirical findings demonstrate that RT-DETR-L achieves a modest performance advantage over convolutional detectors (+4.06 pp mAP@0.5 over YOLOv8s, "
        "95% CI: [+1.21, +6.66] pp).\n\n"
        "In RT-DETR-L, multi-scale representations are processed through an Efficient Hybrid Encoder: attention-based intra-scale feature interaction (AIFI) "
        "applies self-attention within high-level feature maps to capture long-range contextual dependencies, while the Cross-Scale Feature Fusion Module (CCFM) "
        "integrates multi-scale spatial representations using convolutional blocks. Set prediction via bipartite Hungarian matching provides an alternative "
        "formulation to dense spatial anchor assignment. However, as demonstrated in Section 3.3, greedy NMS suppression of true overlapping ova does not explain "
        "performance differences on this dataset, as ground-truth boxes exhibit no IoU overlap >= 0.50. The modest accuracy gain of RT-DETR-L requires a substantial "
        "computational cost of 105.4 GFLOPs and 21.25 ms median forward latency on a desktop GPU (47.1 FPS), making it less suitable for battery-powered, edge-embedded devices."
    )

    add_section_h2("4.2 Convolutional Baseline Equivalence & Edge Considerations")
    p9 = doc.add_paragraph()
    p9.paragraph_format.space_after = Pt(6)
    p9.paragraph_format.line_spacing = 1.15
    p9.add_run(
        "A key empirical finding of this benchmark is that modern YOLO11s (88.33% mAP@0.5, 72.89% mAP@0.5:0.95) does not demonstrate an accuracy advantage over "
        "the YOLOv8s baseline (87.74% mAP@0.5, 73.13% mAP@0.5:0.95). Their paired bootstrap confidence interval spans zero (Delta = +0.58 pp, 95% CI: [-2.10, +2.88] pp), "
        "and YOLOv8s achieved slightly superior mAP@0.5:0.95 along with faster GPU forward pass times (median 4.96 ms vs. 7.06 ms). This indicates that incremental "
        "architectural refinements in modern YOLO iterations yield marginal benefits on specialized microscopy datasets compared to established baselines.\n\n"
        "For field deployment, YOLO11n offers a highly compact parameter and FLOP footprint (2.58M parameters and 6.4 GFLOPs, achieving 86.82% mAP@0.5). "
        "While its AP on Ancylostoma is lower (63.20% AP@0.5 vs. 86.98% for RT-DETR-L), its bootstrap confidence interval is notably wide ([37.8%, 92.5%]), "
        "overlapping the other models. Furthermore, candidate recall remains robust (80.88%), indicating that the lower AP reflects false-positive ranking "
        "trade-offs rather than missed detections."
    )

    add_section_h2("4.3 Cross-Center Diagnostic Robustness & Representation Differences")
    p9_ext = doc.add_paragraph()
    p9_ext.paragraph_format.space_after = Pt(6)
    p9_ext.paragraph_format.line_spacing = 1.15
    p9_ext.add_run(
        "The independent evaluation on 1,000 out-of-distribution microscopic slides (Section 3.5) provides preliminary insight into model behavior under optical and "
        "preparation shifts without local fine-tuning. When applied to an independent external laboratory source, convolutional models flagged Ascaris on 55.10%–58.60% "
        "of positive slides at tau = 0.25, whereas RT-DETR-L maintained an 89.50% slide sensitivity and 99.10% screening yield.\n\n"
        "Crucially, because RT-DETR-L incorporates an HGNetv2 convolutional backbone, this difference cannot be attributed to a simplistic 'transformer vs. convolution' dichotomy. "
        "Rather, several architectural and optimization distinctions likely contribute: (1) global deformable cross-attention in the hybrid encoder (AIFI/CCFM) and 6-layer "
        "transformer decoder models contextual dependencies across the visual field, which may prioritize invariant structural contours over local high-frequency background textures; "
        "(2) direct bipartite set prediction optimizes global prediction assignments without anchor heuristics or non-maximum suppression parameter tuning, potentially stabilizing "
        "candidate confidence ranking under shifted background distributions; and (3) it must be emphasized that because the external cohort comprises positive slides of a single species "
        "(Ascaris) without spatial localization bounding boxes or uninfected controls, this test represents a slide-level candidate detection stress test rather than a full diagnostic validation. "
        "Further multi-center prospective studies evaluating all eight taxa alongside uninfected control slides are necessary to assess true clinical generalization across diverse health centers."
    )

    # 5. Limitations
    add_section_h1("5. Limitations")
    p10 = doc.add_paragraph()
    p10.paragraph_format.space_after = Pt(6)
    p10.paragraph_format.line_spacing = 1.15
    p10.add_run(
        "To ensure scientific transparency, we explicitly delineate key limitations of this study:\n"
        "1. Absence of Negative (Uninfected) Control Smears: The Roboflow Parasites v2 dataset and the external validation dataset consist entirely of images containing parasitic ova. "
        "Without uninfected control slides from healthy individuals or slides with purely non-parasitic artifacts, slide-level clinical screening specificity, "
        "negative predictive value, and false positive rates cannot be determined.\n"
        "2. External Validation Scope: The 1,000-image external validation cohort evaluates out-of-distribution candidate detection for a single species (Ascaris lumbricoides) "
        "and lacks ground-truth spatial bounding box annotations. Expanding multi-center external validation across all eight taxa with full spatial annotations and negative controls "
        "remains an essential priority for future research.\n"
        "3. Desktop GPU Benchmarking vs. Embedded Edge Hardware: Latency measurements were conducted on a desktop GPU (NVIDIA GeForce RTX 5060 Ti). "
        "Although parameter counts and FLOPs provide theoretical guidance, real-world inference latencies on embedded edge hardware (such as Raspberry Pi, "
        "Jetson Nano, or mobile NPUs) will be subject to thermal throttling, memory bandwidth constraints, and integer quantization effects.\n"
        "4. Single-Seed Benchmark Bounds: Training was conducted with a single fixed random seed due to computational constraints. The reported bootstrap "
        "confidence intervals capture test-set image sampling variance, but do not capture across-run training stochasticity from differing initialization "
        "seeds. Differences between closely matched models (such as YOLO11s vs. YOLOv8s) should be interpreted as functional parity rather than hierarchical superiority.\n"
        "5. Class Imbalance in Primary Test Instances: Ground-truth instances range from 40 (Fasciola) to 106 (Enterobius). While adequate for comparative analysis, "
        "precision estimates for taxa with fewer instances carry wider statistical error margins."
    )

    # 6. Conclusion
    add_section_h1("6. Conclusion")
    p11 = doc.add_paragraph()
    p11.paragraph_format.space_after = Pt(6)
    p11.paragraph_format.line_spacing = 1.15
    p11.add_run(
        "In this work, we presented an empirical benchmark of modern object detection architectures for multi-species microscopic parasite egg detection "
        "in fecal smears. By evaluating Detection Transformers (RT-DETR-L), modern compact CNNs (YOLO11s, YOLO11n), and an established standard baseline (YOLOv8s), "
        "we observed that:\n"
        "* RT-DETR-L achieved the highest overall detection accuracy (91.80% mAP@0.5, 77.32% mAP@0.5:0.95), with paired bootstrap analysis showing that its "
        "95% CI excludes zero relative to convolutional baselines (+4.06 pp vs. YOLOv8s, two-sided p = 0.0028; +3.47 pp vs. YOLO11s, two-sided p = 0.0072). "
        "However, geometric analysis found no evidence of an NMS effect (0.00% GT pairs >= 0.5 IoU), and interaction testing showed no statistically significant crowding interaction "
        "(Delta_interaction = +4.03 pp, 95% CI: [-2.23, +9.55] pp, two-sided p = 0.2460);\n"
        "* YOLO11s performs at functional parity with the established YOLOv8s baseline, showing no statistically significant accuracy gain;\n"
        "* YOLO11n provides a compact parameter and FLOP footprint for resource-constrained deployments (86.82% mAP@0.5, 2.58M parameters, 6.4 GFLOPs), "
        "while maintaining 80.88% recall on hookworm ova;\n"
        "* In an out-of-distribution stress test across 1,000 independent external microscopic slides with confirmed Ascaris lumbricoides infection, "
        "RT-DETR-L maintained 89.50% slide sensitivity (99.10% screening yield), compared to 55.10%–58.60% for convolutional models.\n\n"
        "Future research should focus on prospective multi-center cohorts containing uninfected control slides across all eight taxa, quantitative egg-per-gram counting, "
        "and INT8 on-device profiling on embedded microcontrollers for field deployment."
    )

    # Declarations
    add_section_h1("Declarations")
    p_dec = doc.add_paragraph()
    p_dec.paragraph_format.space_after = Pt(6)
    p_dec.paragraph_format.line_spacing = 1.15
    p_dec.add_run(
        "* Funding: This research received no external funding.\n"
        "* Competing Interests: The authors declare that they have no competing financial or non-financial interests.\n"
        "* Data Availability: The Roboflow Parasites v2 dataset is publicly accessible via Roboflow Universe [14] under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. The external validation dataset is accessible via Roboflow Universe under the CC BY 4.0 license.\n"
        "* Code Availability: Model configuration files, training scripts, and evaluation pipelines will be made publicly accessible on GitHub upon publication; during peer review, anonymized source code and checkpoints are available from the corresponding author upon reasonable request.\n"
        "* Ethics Statement: This study utilized existing, publicly available, fully anonymized image datasets. No human biological samples were directly collected, and no identifiable patient information was accessed."
    )

    # References
    add_section_h1("References")
    refs = [
        "[1] P. J. Hotez, P. J. Brindley, J. M. Bethony, C. H. King, E. J. Pearce, and J. Jacobson, 'Helminth infections: the great neglected tropical diseases,' The Journal of Clinical Investigation, vol. 118, no. 4, pp. 1311–1321, 2008.",
        "[2] World Health Organization, Ending the neglect to attain the Sustainable Development Goals: a road map for neglected tropical diseases 2021–2030, World Health Organization, Geneva, Switzerland, 2020.",
        "[3] L. S. Garcia, Diagnostic Medical Parasitology, 6th ed., ASM Press, Washington, DC, 2016.",
        "[4] N. Katz, A. Chaves, and J. Pellegrino, 'A simple device for quantitative stool thick-smear technique in Schistosomiasis mansoni,' Revista do Instituto de Medicina Tropical de Sao Paulo, vol. 14, no. 6, pp. 397–400, 1972.",
        "[5] J. Utzinger, S. Botero-Kleiven, F. Castelli, et al., 'Microscopic diagnosis of sodium acetate-acetic acid-formalin-fixed stool samples for helminths and intestinal protozoa: a comparison among European reference laboratories,' Clinical Microbiology and Infection, vol. 16, no. 3, pp. 267–273, 2010.",
        "[6] B. Bogitsh, C. Carter, and T. Oeltmann, Human Parasitology, 5th ed., Academic Press, London, UK, 2018.",
        "[7] World Health Organization, Bench aids for the diagnosis of intestinal parasites, 2nd ed., World Health Organization, Geneva, Switzerland, 2019.",
        "[8] O. Holmström, N. Linder, B. Ngasala, et al., 'Point-of-care mobile digital microscopy and deep learning for the detection of soil-transmitted helminths and Schistosoma haematobium,' Global Health Action, vol. 10, sup. 3, p. 1337325, 2017.",
        "[9] P. Ward, P. Dahlberg, O. Lagatie, et al., 'Affordable artificial intelligence-based digital pathology for neglected tropical diseases: A proof-of-concept for the detection of soil-transmitted helminths and Schistosoma mansoni eggs in Kato-Katz stool thick smears,' PLoS Neglected Tropical Diseases, vol. 16, no. 6, p. e0010500, 2022.",
        "[10] M. A. F. Ghazvini, M. R. K. Zanjani, and H. R. Pourghassem, 'Computer-aided diagnosis of human intestinal parasites in microscopic images: A review,' Computer Methods and Programs in Biomedicine, vol. 195, p. 105650, 2020.",
        "[11] M. Hassan, A. Alqahtani, M. Elbedwehy, and A. S. Talaat, 'Automated detection of pinworm parasite eggs using YOLO convolutional block attention module for enhanced microscopic image analysis,' Frontiers in Bioengineering and Biotechnology, vol. 13, p. 1559987, Oct. 2025.",
        "[12] M. A. Mohammed, et al., 'Detecting soil-transmitted helminth and Schistosoma mansoni eggs in Kato-Katz stool smear microscopy images: A comprehensive in- and out-of-distribution evaluation of YOLOv7 variants,' PLoS Neglected Tropical Diseases, vol. 19, no. 7, p. e0012890, July 2025.",
        "[13] W. Lv, Y. Zhao, S. Xu, J. Wei, G. Wang, C. Lai, Q. Shen, and Q. Liu, 'DETRs Beat YOLOs on Real-time Object Detection,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16965–16974.",
        "[14] Roboflow Universe, 'Parasites v2 Microscopic Dataset,' 2024. [Online]. Available: https://universe.roboflow.com/roboflow-100/parasites-124gj.",
        "[15] G. Jocher, A. Chaurasia, and J. Qiu, 'Ultralytics YOLO,' 2024. [Online]. Available: https://github.com/ultralytics/ultralytics."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(3)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.left_indent = Inches(0.25)
        p_ref.paragraph_format.first_line_indent = Inches(-0.25)
        r = p_ref.add_run(ref)
        r.font.size = Pt(8.5)

    # Save to workspace across multiple candidate filenames in case one is open in Word
    saved_paths = []
    for fname in ["MANUSCRIPT_with_figures.docx", "MANUSCRIPT_v6.docx", "MANUSCRIPT_v5.docx", "MANUSCRIPT.docx"]:
        try:
            p = Path(fname)
            doc.save(p)
            saved_paths.append(str(p.resolve()))
            print(f"Saved formatted manuscript docx to {p}")
        except PermissionError:
            print(f"File {fname} is currently open in Word; skipping.")

    # Also save to artifact directory
    artifact_dir = Path(r"C:\Users\Amit_Addi\.gemini\antigravity-ide\brain\4a0a2c54-36c0-4c2f-b27d-e617deb9a7ba")
    if artifact_dir.exists():
        for art_fname in ["MANUSCRIPT_with_figures.docx", "MANUSCRIPT_v6.docx", "MANUSCRIPT.docx"]:
            try:
                art_p = artifact_dir / art_fname
                doc.save(art_p)
                print(f"Saved artifact copy to {art_p}")
                break
            except PermissionError:
                continue

if __name__ == "__main__":
    create_manuscript_docx()
