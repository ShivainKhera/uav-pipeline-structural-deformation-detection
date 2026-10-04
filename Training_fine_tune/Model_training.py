from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8s.pt")
    results = model.train(
        data="./Training_dataset/data.yaml",
        imgsz=[32, 24],
        epochs=100,
        batch=16,
        device=device,
        name="yolov8s_32x24_colored"
    )
    print(results)

