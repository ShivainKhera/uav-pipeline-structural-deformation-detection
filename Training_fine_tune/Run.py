from ultralytics import YOLO
import torch
import cv2
import os
from pathlib import Path

def is_image_file(p: Path):
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if __name__ == "__main__":
    # Folders
    in_dir = Path("inputs")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = YOLO("runs/detect/yolov8s_32x24_colored7/weights/best.pt")

    # Collect image paths
    image_paths = [p for p in in_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not image_paths:
        print(f"No images found in {in_dir.resolve()}")
        exit(0)

    # Optional: device selection (cpu, cuda, mps)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable file: {img_path.name}")
            continue

        # Upscale input image to 320x240
        img_upscaled = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)

        # Predict from in-memory array to avoid temp files
        results = model.predict(source=img_upscaled, conf=0.15, verbose=False)

        # Annotated image
        annotated_img = results[0].plot()

        # Build output filenames
        stem = img_path.stem
        annotated_out = out_dir / f"{stem}_upscaled_boxes.png"
        upscaled_out = out_dir / f"{stem}_upscaled.jpg"
        txt_out = out_dir / f"{stem}_detections.txt"

        # Save images
        cv2.imwrite(str(upscaled_out), img_upscaled)
        cv2.imwrite(str(annotated_out), annotated_img)

        # Print and save detection results
        lines = []
        print(f"Detection results for {img_path.name}:")
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = round(float(box.conf.item()), 3)
            xyxy = [round(x.item(), 2) for x in box.xyxy.flatten()]
            line = f"Class: {cls_id}  Confidence: {conf}  Box: {xyxy}"
            print(line)
            lines.append(line)

        # Write a per-image text log
        with open(txt_out, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))
            else:
                f.write("No detections.")

    print(f"Processed {len(image_paths)} image(s). Outputs saved to: {out_dir.resolve()}")
