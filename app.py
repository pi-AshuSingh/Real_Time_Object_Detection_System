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
st.set_page_config(page_title="DetectXpress Advanced", page_icon="👁️", layout="wide")

st.title("👁️ DetectXpress - Advanced AI Vision Pro")
st.markdown("### Real-Time Object Detection & Accident Prevention System")

# Load models once and cache them globally
@st.cache_resource
def load_models():
    print("Loading YOLOv11m...")
    yolo_model = YOLO("yolo11m.pt")
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
FOCAL_LENGTH = 700

KNOWN_HEIGHTS = {
    'person': 1.7, 'child': 1.2, 'car': 1.5, 'truck': 3.0,
    'bus': 3.5, 'bicycle': 1.0, 'motorcycle': 1.2,
    'pedestrian': 1.7, 'stop sign': 2.0, 'traffic light': 3.0
}

class DetectXpressTransformer(VideoTransformerBase):
    def __init__(self):
        # State tracking (isolated to the thread)
        self.session_start = time.time()
        self.eyes_closed_frames = 0
        self.yawn_frames = 0
        self.frame_count = 0
        self.object_tracker = {}
        self.detection_history = defaultdict(lambda: deque(maxlen=10))
        
        self.stats = {
            'total_detections': 0,
            'pedestrian_warnings': 0,
            'drowsiness_alerts': 0,
            'critical_alerts': 0,
            'alert_history': deque(maxlen=100),
            'last_alert_time': {}
        }
        
    def detect_night_mode(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 60

    def enhance_frame_for_night(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return cv2.fastNlMeansDenoisingColored(enhanced_frame, None, 10, 10, 7, 21)

    def preprocess_frame_for_detection(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(enhanced_frame, 5, 50, 50)

    def estimate_distance(self, bbox_height, object_class):
        if object_class not in KNOWN_HEIGHTS or bbox_height == 0:
            return None
        distance = (KNOWN_HEIGHTS[object_class] * FOCAL_LENGTH) / bbox_height
        
        self.detection_history[object_class].append(distance)
        if len(self.detection_history[object_class]) >= 3:
            recent = list(self.detection_history[object_class])[-5:]
            return round(sum(recent) / len(recent), 1)
        return round(distance, 1)

    def get_danger_level(self, distance):
        if distance is None: return 'UNKNOWN', (128, 128, 128)
        if distance < CRITICAL_ZONE: return 'CRITICAL', (0, 0, 255)
        if distance < WARNING_ZONE: return 'WARNING', (0, 165, 255)
        if distance < ALERT_ZONE: return 'ALERT', (0, 255, 255)
        return 'SAFE', (0, 255, 0)

    def track_objects(self, detections):
        self.frame_count += 1
        current_objects = {}
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            obj_class = det['class']
            
            matched = False
            for obj_id, obj_data in self.object_tracker.items():
                if obj_data['class'] == obj_class:
                    prev_center = obj_data['center']
                    dist = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                    if dist < 100:
                        current_objects[obj_id] = {
                            'class': obj_class, 'center': center, 'bbox': det['bbox'],
                            'history': obj_data['history'] + [center], 'frames': obj_data['frames'] + 1
                        }
                        matched = True
                        break
            if not matched:
                new_id = f"{obj_class}_{self.frame_count}_{idx}"
                current_objects[new_id] = {
                    'class': obj_class, 'center': center, 'bbox': det['bbox'],
                    'history': [center], 'frames': 1
                }
        self.object_tracker = current_objects
        return current_objects

    def estimate_speed(self, obj_data, fps=10):
        if len(obj_data['history']) < 2: return None
        recent_history = obj_data['history'][-10:]
        if len(recent_history) < 2: return None
        
        start_pos = recent_history[0]
        end_pos = recent_history[-1]
        pixel_distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        time_elapsed = len(recent_history) / fps
        if time_elapsed == 0: return None
        return round((pixel_distance / time_elapsed) * 0.01, 1)

    def calculate_eye_aspect_ratio(self, eye_points):
        if len(eye_points) < 2: return 0.3
        v_dists = [abs(eye_points[i][1] - eye_points[i+1][1]) for i in range(len(eye_points)-1)]
        avg_v = sum(v_dists) / len(v_dists) if v_dists else 0
        h = abs(eye_points[0][0] - eye_points[-1][0])
        if h == 0: return 0.3
        return max(0.1, min(0.5, avg_v / h))

    def detect_yawn(self, face_roi):
        h, w = face_roi.shape[:2]
        mouth_region = face_roi[int(h*0.6):, int(w*0.2):int(w*0.8)]
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mouth, 60, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        total = mouth_region.shape[0] * mouth_region.shape[1]
        if total == 0: return False
        return (white_pixels / total) > 0.3

    def log_event(self, event, level='INFO'):
        current_time = time.time()
        last_time = self.stats['last_alert_time'].get(event, 0)
        cooldown = 5.0 if level in ['CRITICAL', 'WARNING'] else 2.0
        
        if (current_time - last_time) < cooldown: return
        
        self.stats['last_alert_time'][event] = current_time
        self.stats['alert_history'].append({
            'timestamp': current_time,
            'event': event,
            'type': event,
            'level': level
        })

    def calculate_safety_score(self):
        current_time = time.time()
        uptime_minutes = (current_time - self.session_start) / 60
        score = 100
        
        if uptime_minutes < 0.08: return 85
        
        recent_30sec = current_time - 30
        recent_60sec = current_time - 60
        recent_90sec = current_time - 90
        
        c_30s, c_60s, c_90s, w_recent, d_recent = 0, 0, 0, 0, 0
        last_crit = 0
        
        for alert in self.stats['alert_history']:
            t = alert['timestamp']
            lvl = alert['level']
            typ = alert['type']
            
            if lvl == 'CRITICAL' and t > last_crit: last_crit = t
            if 'drowsiness' in typ.lower() and t >= recent_60sec: d_recent += 1
            if lvl == 'CRITICAL':
                if t >= recent_30sec: c_30s += 1
                elif t >= recent_60sec: c_60s += 1
                elif t >= recent_90sec: c_90s += 1
            elif lvl == 'WARNING' and t >= recent_60sec: w_recent += 1
                
        score -= c_30s * 15 + c_60s * 5 + c_90s * 2 + d_recent * 10 + w_recent * 3
        
        time_since = current_time - last_crit if last_crit > 0 else uptime_minutes * 60
        if c_30s == 0 and d_recent == 0:
            score += min(30, int(time_since * 0.5))
            if c_60s == 0 and w_recent == 0: score += 15
            
        if self.stats['total_detections'] > 10: score += 5
        return max(0, min(100, int(score)))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Night mode
        if self.detect_night_mode(img):
            img = self.enhance_frame_for_night(img)
            cv2.putText(img, "🌙 NIGHT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        annotated = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = self.preprocess_frame_for_detection(img)
        
        # YOLO inference
        results = model(processed, conf=0.40, iou=0.45, imgsz=640, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf < 0.40: continue
                
                if cls < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[cls]
                    if conf < (0.50 if class_name in ['person', 'car', 'truck', 'bus'] else 0.40): continue
                    
                    self.stats['total_detections'] += 1
                    distance = self.estimate_distance(y2 - y1, class_name)
                    danger_level, color = self.get_danger_level(distance)
                    
                    detections.append({'class': class_name, 'bbox': [x1, y1, x2, y2]})
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    if distance: label += f" | {distance}m"
                    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    if danger_level == 'CRITICAL':
                        self.log_event(f'CRITICAL_{class_name}', 'CRITICAL')
                        self.stats['critical_alerts'] += 1
                        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 255), 10)
                        cv2.putText(annotated, "⚠️ COLLISION RISK!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Speed estimation
        tracked = self.track_objects(detections)
        for obj_id, obj_data in tracked.items():
            speed = self.estimate_speed(obj_data)
            if speed and speed > 0:
                x1, y1, x2, y2 = obj_data['bbox']
                cv2.putText(annotated, f"{speed} m/s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Face & EAR
        faces = face_cascade.detectMultiScale(gray, 1.05, 7, minSize=(80, 80))
        for (x, y, w, h) in faces[:1]:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            
            valid_eyes = [eye for eye in eyes if eye[1] < h * 0.6]
            ear = 0.3
            if len(valid_eyes) >= 2:
                eye_points = [(x + ex + ew//2, y + ey + eh//2) for ex, ey, ew, eh in valid_eyes[:2]]
                ear = self.calculate_eye_aspect_ratio(eye_points)
                
            if len(valid_eyes) >= 2 and ear > 0.20:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 5)
                cv2.putText(annotated, f"✅ EYES OPEN ({ear:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif ear < 0.15:
                self.eyes_closed_frames += 1
                if self.eyes_closed_frames > DROWSY_THRESHOLD:
                    self.stats['drowsiness_alerts'] += 1
                    self.log_event('DROWSINESS', 'CRITICAL')
                    cv2.putText(annotated, "😴 WAKE UP!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(annotated, f"😴 CLOSED ({ear:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                self.eyes_closed_frames = max(0, self.eyes_closed_frames - 2)

        # Draw full dashboard on screen
        score = self.calculate_safety_score()
        color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 50 else (0, 0, 255)
        
        cv2.rectangle(annotated, (10, 10), (300, 130), (0, 0, 0), -1)
        cv2.putText(annotated, f"Safety Score: {score}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, f"Critical Alerts: {self.stats['critical_alerts']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Drowsy Alerts: {self.stats['drowsiness_alerts']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

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
    st.success("✅ Drowsiness Detection (EAR)")
    st.success("✅ Dynamic Safety Score")
with col2:
    st.success("✅ Collision Prediction")
    st.success("✅ Speed Estimation")
    st.success("✅ Distance Estimation")
with col3:
    st.success("✅ Browser WebRTC Support")
    st.success("✅ Night Mode Auto-Enhance")
    st.success("✅ Image Preprocessing")
