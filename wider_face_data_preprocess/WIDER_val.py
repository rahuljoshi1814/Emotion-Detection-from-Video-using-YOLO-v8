import os

# Function to convert bounding box coordinates
def convert_bbox(img_width, img_height, x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Paths
annotations_path = "datasets/wider_face/wider_face_split/wider_face_train_bbx_gt.txt"
images_path = "datasets/wider_face/WIDER_val/images"
output_path = "datasets/wider_face/WIDER_val/labels"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Read annotation file
with open(annotations_path, 'r') as file:
    lines = file.readlines()

image_name = ""
for line in lines:
    line = line.strip()
    if line.endswith('.jpg'):  # Image name
        image_name = line
        image_base = os.path.basename(image_name).replace('.jpg', '.txt')
        label_file_path = os.path.join(output_path, image_base)
        with open(label_file_path, 'w') as lf:
            pass  # Create empty file for each image
    elif line.isdigit():  # Number of faces (ignore it for YOLO)
        continue
    else:  # Bounding box coordinates
        try:
            bbox = list(map(int, line.split()))
            if len(bbox) < 4:
                print(f"Skipping invalid bounding box: {line}")
                continue
            x_min, y_min, width, height = bbox[:4]
            img_width, img_height = 640, 640  # Update if actual image dimensions are known
            x_max, y_max = x_min + width, y_min + height
            x_center, y_center, w, h = convert_bbox(img_width, img_height, x_min, y_min, x_max, y_max)
            with open(label_file_path, 'a') as lf:
                lf.write(f"0 {x_center} {y_center} {w} {h}\n")
        except ValueError as e:
            print(f"Error processing line: {line} -> {e}")
