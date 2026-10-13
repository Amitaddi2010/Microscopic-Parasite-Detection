import json
import os
from pathlib import Path

def convert_coco_to_yolo():
    data_dir = Path("Dataset")
    splits = {
        "train": data_dir / "train",
        "val": data_dir / "valid",
        "test": data_dir / "test"
    }
    
    # Read categories from train annotation
    train_anno_path = splits["train"] / "_annotations.coco.json"
    with open(train_anno_path, "r", encoding="utf-8") as f:
        train_coco = json.load(f)
        
    # Categories: filter out id 0 ('parasites') which has 0 annotations
    categories = [c for c in train_coco["categories"] if c["id"] != 0]
    categories = sorted(categories, key=lambda x: x["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]
    print(f"Detected {len(class_names)} species classes: {class_names}")
    
    yolo_dir = Path("Dataset_YOLO")
    for split_name, split_path in splits.items():
        img_out_dir = yolo_dir / "images" / split_name
        lbl_out_dir = yolo_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        
        anno_file = split_path / "_annotations.coco.json"
        with open(anno_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
            
        img_map = {img["id"]: img for img in coco_data["images"]}
        img_annos = {img["id"]: [] for img in coco_data["images"]}
        for anno in coco_data["annotations"]:
            if anno["category_id"] in id_to_idx:
                img_annos[anno["image_id"]].append(anno)
                
        converted_count = 0
        for img_id, img_info in img_map.items():
            file_name = img_info["file_name"]
            src_img_path = split_path / file_name
            dst_img_path = img_out_dir / file_name
            
            # Copy or hardlink image
            if not dst_img_path.exists():
                try:
                    os.link(src_img_path, dst_img_path)
                except Exception:
                    import shutil
                    shutil.copy2(src_img_path, dst_img_path)
                    
            # Generate YOLO label file
            width = float(img_info["width"])
            height = float(img_info["height"])
            label_lines = []
            
            for anno in img_annos[img_id]:
                cat_idx = id_to_idx[anno["category_id"]]
                bbox = anno["bbox"] # [x, y, w, h] in COCO
                x, y, w, h = bbox
                
                # Convert to YOLO normalized center format [x_center, y_center, w, h]
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                norm_w = w / width
                norm_h = h / height
                
                # Clip to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                label_lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                
            txt_name = Path(file_name).stem + ".txt"
            dst_txt_path = lbl_out_dir / txt_name
            with open(dst_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines) + ("\n" if label_lines else ""))
                
            converted_count += 1
            
        print(f"Split '{split_name}': {converted_count} images & labels processed.")
        
    # Write dataset.yaml for YOLO/RT-DETR
    yaml_content = f"""path: {yolo_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open("dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print("dataset.yaml created successfully.")

if __name__ == "__main__":
    convert_coco_to_yolo()
