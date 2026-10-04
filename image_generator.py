import cv2
import os

input_folder = 'Stock images'
output_folder = 'Stock_32x24'

os.makedirs(output_folder, exist_ok=True)

target_width = 32
target_height = 24

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {filename}: unable to read image.")
        continue

    h, w = img.shape[:2]

    # Crop 5% from each border
    x_crop = int(w * 0.05)
    y_crop = int(h * 0.05)
    cropped_img = img[y_crop:h - y_crop, x_crop:w - x_crop]

    # Resize while maintaining aspect ratio
    crop_h, crop_w = cropped_img.shape[:2]
    target_aspect = target_width / target_height
    crop_aspect = crop_w / crop_h

    if crop_aspect > target_aspect:
        # Wide image: crop width
        new_width = int(crop_h * target_aspect)
        x_center = crop_w // 2
        cropped_img = cropped_img[:, x_center - new_width // 2:x_center + new_width // 2]
    elif crop_aspect < target_aspect:
        # Tall image: crop height
        new_height = int(crop_w / target_aspect)
        y_center = crop_h // 2
        cropped_img = cropped_img[y_center - new_height // 2:y_center + new_height // 2, :]

    # Final resize to target size
    resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_img)

print("All images processed and saved in", output_folder)
