import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import deque, defaultdict
import av

# Initialize Streamlit Page Config
st.set_page_config(page_title="DetectXpress", page_icon="👁️", layout="wide")

st.title("👁️ DetectXpress - Advanced AI Vision Pro")
st.markdown("### Real-Time Object Detection & Accident Prevention System")
st.markdown("This application uses **YOLOv11** and **WebRTC** to perform real-time detection directly from your browser's webcam.")

# Load models outside to prevent reloading
@st.cache_resource
def load_models():
    print("Loading YOLOv11x...")
    yolo_model = YOLO("yolo11x.pt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return yolo_model, face_cascade, eye_cascade

model, face_cascade, eye_cascade = load_models()
COCO_CLASSES = model.names

# Constants
CRITICAL_ZONE = 5
WARNING_ZONE = 15
ALERT_ZONE = 30
DROWSY_THRESHOLD = 20

KNOWN_HEIGHTS = {
    'person': 1.7, 'child': 1.2, 'car': 1.5, 'truck': 3.0,
    'bus': 3.5, 'bicycle': 1.0, 'motorcycle': 1.2,
    'pedestrian': 1.7
}
FOCAL_LENGTH = 700

# Video Transformer Class for WebRTC
class DetectXpressTransformer(VideoTransformerBase):
    def __init__(self):
        self.eyes_closed_frames = 0
        self.alert_history = deque(maxlen=100)
        self.session_start = time.time()
        self.critical_count = 0
        self.drowsy_count = 0
        
    def estimate_distance(self, bbox_height, object_class):
        if object_class not in KNOWN_HEIGHTS or bbox_height == 0:
            return None
        return round((KNOWN_HEIGHTS[object_class] * FOCAL_LENGTH) / bbox_height, 1)

    def get_danger_level(self, distance):
        if distance is None: return 'UNKNOWN', (128, 128, 128)
        if distance < CRITICAL_ZONE: return 'CRITICAL', (0, 0, 255)
        if distance < WARNING_ZONE: return 'WARNING', (0, 165, 255)
        if distance < ALERT_ZONE: return 'ALERT', (0, 255, 255)
        return 'SAFE', (0, 255, 0)
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_frame = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. YOLO Inference
        results = model(img, conf=0.40, imgsz=640, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[cls]
                    distance = self.estimate_distance(y2 - y1, class_name)
                    danger_level, color = self.get_danger_level(distance)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    if distance: label += f" | {distance}m"
                    
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                    if danger_level == 'CRITICAL':
                        self.critical_count += 1
                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), 10)
                        cv2.putText(annotated_frame, "⚠️ COLLISION RISK!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 2. Face & Drowsiness Detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces[:1]: # Only track primary face
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            
            if len(eyes) >= 2:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 2)
                cv2.putText(annotated_frame, "✅ EYES OPEN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(eyes) == 0:
                self.eyes_closed_frames += 1
                cv2.putText(annotated_frame, "😴 EYES CLOSED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            if self.eyes_closed_frames > DROWSY_THRESHOLD:
                self.drowsy_count += 1
                cv2.putText(annotated_frame, "😴 WAKE UP! DROWSY DRIVER!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Draw mini-dashboard on frame
        cv2.putText(annotated_frame, f"Critical Alerts: {self.critical_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Drowsy Alerts: {self.drowsy_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.markdown("### 📹 Live Feed")
st.info("Click **Start** to allow browser camera access and begin processing.")

webrtc_streamer(
    key="detectxpress",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DetectXpressTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown("### 📊 Features Running")
col1, col2, col3 = st.columns(3)
with col1:
    st.success("✅ Object Detection (YOLOv11x)")
    st.success("✅ Drowsiness Detection (Haar Cascades)")
with col2:
    st.success("✅ Collision Prediction")
    st.success("✅ Distance Estimation")
with col3:
    st.success("✅ Browser WebRTC Support")
    st.success("✅ Cloud-Ready Architecture")
