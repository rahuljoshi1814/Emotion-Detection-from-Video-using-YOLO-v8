import tensorflow as tf
from keras.models import load_model
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the emotion recognition model using Keras
emotion_model = load_model('emotion_recognition_model.h5')  # Keras model

# Transform for emotion model input (adjust according to your model's input requirements)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Load YOLOv8 model
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # Or use a video path for testing

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform face detection using YOLOv8
    results = yolo_model(frame)  # Perform inference on the frame

    # Access the bounding boxes directly
    for result in results[0].boxes:  # Get the first set of results
        xmin, ymin, xmax, ymax = result.xyxy[0]  # Get xyxy format (xmin, ymin, xmax, ymax)
        confidence = result.conf[0]  # Get the confidence score for the detection
        
        # If confidence is above a certain threshold, process for emotion recognition
        if confidence > 0.5:
            # Extract the face from the frame using the bounding box coordinates
            face = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            
            # Preprocess the face for emotion detection
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face = transform(face)  # Transform to tensor
            
            # Add batch dimension and ensure the correct shape (1, 48, 48, 1)
            face = face.unsqueeze(0)  # Add batch dimension (1, 48, 48, 1)

            # Convert face to the correct shape (1, 48, 48, 1) by squeezing the extra dimensions
            face = np.squeeze(face)  # Remove extra dimension (1, 48, 48, 1) to (48, 48, 1)
            face = np.expand_dims(face, axis=0)  # Now the shape becomes (1, 48, 48, 1)

            # Predict the emotion using Keras model
            predictions = emotion_model.predict(face)
            emotion = np.argmax(predictions, axis=1)[0]

            # Emotion labels
            emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            detected_emotion = emotion_labels[emotion]

            # Draw the bounding box and the predicted emotion on the frame
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with bounding boxes and emotion labels
    cv2.imshow("Face Detection with Emotion Recognition", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


