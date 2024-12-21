from ultralytics import YOLO
from keras.models import load_model
import cv2
import numpy as np

# Load the YOLOv8 model for face detection
yolo_model = YOLO("yolov8n.pt")

# Load the trained emotion recognition model
emotion_model = load_model("emotion_recognition_model.h5")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open video file or webcam stream
video_capture = cv2.VideoCapture("path_to_video.mp4")  # Use 0 for webcam

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing (optional)
    resized_frame = cv2.resize(frame, (640, 640))

    # Perform face detection with YOLO
    results = yolo_model.predict(resized_frame)

    # Process detections
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        confidence = detection.conf[0]

        # Crop face from the frame
        face = frame[y_min:y_max, x_min:x_max]

        # Preprocess the face for emotion recognition model
        if face.size > 0:
            face_resized = cv2.resize(face, (48, 48))  # Resize to model input size
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_array = np.expand_dims(face_gray, axis=[0, -1]) / 255.0  # Normalize and reshape

            # Predict emotion
            emotion_prediction = emotion_model.predict(face_array)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Emotion Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

# Define video writer
output_video = cv2.VideoWriter('output_emotions.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                               (int(video_capture.get(3)), int(video_capture.get(4))))

# Inside the video processing loop
output_video.write(frame)  # Save each processed frame
