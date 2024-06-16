from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np

# Specify the path to the model file
model_path = "C:/Users/adit/Downloads/image processing/Training_Yolo_Custom_Object_Detection_files-main/best.pt"

# Load the YOLO model from the specified folder
model = YOLO(model_path)
class_names = model.names

# Specify the path to the input image
image_path = "C:/Users/adit/Downloads/image processing/yolov5win11customobj-main/3.jpeg"
output_path = "C:/Users/adit/Downloads/image processing/yolov5win11customobj-main/7.jpeg"

# Read the image using PIL
image = Image.open(image_path)

# Convert image to numpy array
image_np = np.array(image)

# Use the model to make predictions on the image
results = model(image_np)

# Print results to the terminal and filter boxes
print(f"Results for {os.path.basename(image_path)}:")
object_boxes = {}
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])  # Convert to int
        class_name = class_names[class_id]  # Get class name from class id
        confidence = float(box.conf[0])  # Convert to float
        bbox = box.xyxy[0].tolist()  # Convert to list
        

        # Filter to keep the box with higher confidence for the same object
        if class_id not in object_boxes or object_boxes[class_id]['confidence'] < confidence:
            object_boxes[class_id] = {'bbox': bbox, 'confidence': confidence, 'class_name': class_name}
            print(f"Class: {class_name} (ID: {class_id}), Confidence: {confidence:.2f}, BBox: {bbox}")

# Annotate the image with the filtered bounding boxes
annotated_image = image_np.copy()
for obj in object_boxes.values():
    bbox = obj['bbox']
    confidence = obj['confidence']
    class_name = obj['class_name']
    # Draw bounding box
    cv2.rectangle(annotated_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    # Put class name and confidence
    cv2.putText(annotated_image, f'{class_name} {confidence:.2f}', (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert the annotated image back to PIL format in RGB
annotated_image_pil = Image.fromarray(annotated_image)

# Save the annotated image in RGB format
annotated_image_pil.save(output_path)
print(f"Annotated image saved to {output_path}")

# Convert the annotated image to BGR format for OpenCV display
annotated_image_bgr = cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR)

# Display the resulting frame with OpenCV
cv2.imshow('YOLO Detection', annotated_image_bgr)

# Wait for a key press, if 'q' is pressed, close the window
print("Press 'q' to close the image window.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
