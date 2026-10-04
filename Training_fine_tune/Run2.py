import cv2
from ultralytics import YOLO

input_video = 'video.mp4'  # 0 for camera
output_video = 'output_detected_video.mp4'
model_path = 'runs/detect/yolov8s_32x24_colored7/weights/best.pt'

# Set upscaled resolution 320x240
UPSCALED_SIZE = (320, 240)

model = YOLO(model_path)

# Video input
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, UPSCALED_SIZE)
print("Starting video inference. Press Q to stop early.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    upscaled_frame = cv2.resize(frame, UPSCALED_SIZE, interpolation=cv2.INTER_LINEAR)
    results = model.predict(source=upscaled_frame, conf=0.15, verbose=False)
    annotated = results[0].plot()

    # output video
    out.write(annotated)

    cv2.imshow('Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Done. Output saved to", output_video)
