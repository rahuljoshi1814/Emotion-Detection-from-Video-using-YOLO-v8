from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # Use YOLOv8 (you can try other versions if required)

# Open video file or webcam (0 for webcam, replace with a path for a video file)
cap = cv2.VideoCapture(0)  # Change this to a video path if you have a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform face detection using YOLOv8
    results = model(frame)  # Run inference on the frame
    
    # Access the bounding boxes directly
    for result in results[0].boxes:  # Get the first set of results
        xmin, ymin, xmax, ymax = result.xyxy[0]  # Get xyxy format (xmin, ymin, xmax, ymax)
        confidence = result.conf[0]  # Get the confidence score for the detection
        
        # If confidence is above a certain threshold, draw the bounding box
        if confidence > 0.5:  # You can adjust this threshold
            # Draw rectangle around the face (green color, thickness=2)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Show the frame with bounding boxes
    cv2.imshow("Face Detection with YOLOv8", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


