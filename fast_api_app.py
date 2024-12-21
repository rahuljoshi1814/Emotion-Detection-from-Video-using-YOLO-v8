from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO models
try:
    face_model = YOLO("yolov8n.pt", task="detect")  # YOLO for face detection
    emotion_model = YOLO("emotion_recognition_model.h5", task="classify")  # YOLO for emotion classification
    logger.info("YOLO models loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO models: {e}")
    raise HTTPException(status_code=500, detail="Error loading YOLO models")

@app.post("/detect-emotions/")
async def detect_emotions(video: UploadFile = File(...)):
    """
    Endpoint to process a video file for face detection and emotion recognition.
    """
    try:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video.file.read())
            tmp_video_path = tmp_video.name
        logger.info(f"Video saved to {tmp_video_path}")

        # Open the video file
        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            logger.error("Unable to open video file")
            return JSONResponse({"error": "Unable to process the video file."}, status_code=400)

        results = []  # Store results for each frame

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection using YOLO
            face_results = face_model(frame)
            frame_results = []  # Store results for this frame

            for face in face_results[0].boxes:  # Iterate through detected faces
                xmin, ymin, xmax, ymax = map(int, face.xyxy[0])  # Get bounding box coordinates
                confidence = face.conf[0]  # Get confidence score

                if confidence > 0.5:  # Only process faces with sufficient confidence
                    # Crop the detected face for emotion recognition
                    face_crop = frame[ymin:ymax, xmin:xmax]
                    if face_crop.size == 0:
                        continue

                    # Predict emotion using the emotion classification model
                    emotion_results = emotion_model(face_crop)
                    emotion_probs = emotion_results[0].probs  # Get probabilities for each emotion class
                    if emotion_probs is None:
                        continue
                    emotion = emotion_results[0].names[np.argmax(emotion_probs)]  # Get the predicted emotion

                    # Append the detection result for the current face
                    frame_results.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "emotion": emotion,
                        "confidence": float(confidence)
                    })

            # Store the results for this frame
            results.append({"frame": len(results) + 1, "detections": frame_results})

        # Release the video capture object
        cap.release()

        # Return the results as a JSON response
        return {"results": results}

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return JSONResponse({"error": f"Error processing video: {e}"}, status_code=500)
