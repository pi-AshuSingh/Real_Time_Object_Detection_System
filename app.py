import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from collections import deque, defaultdict
import av
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components

# Initialize Streamlit Page Config
st.set_page_config(page_title="DetectXpress Ultimate", page_icon="🏆", layout="wide")

# Inject Custom CSS for Premium Glassmorphic Theme
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #020617 0%, #1e1b4b 100%); color: #f8fafc; }
    h1, h2, h3, h4 { background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    [data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(15px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .stAlert { background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(10px) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 12px !important; color: #e2e8f0 !important; }
    [data-testid="stWebRtc"] { display: flex; justify-content: center; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.3); }
    [data-testid="stElementContainer"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🏆 DetectXpress - Ultimate ADAS Prototype")
st.markdown("### Award-Winning Real-Time Object Detection & Behavioral Analysis")

@st.cache_resource
def load_models():
    print("Loading YOLOv11x...")
    yolo_model = YOLO("yolo11x.pt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return yolo_model, face_cascade, eye_cascade, smile_cascade

model, face_cascade, eye_cascade, smile_cascade = load_models()
COCO_CLASSES = model.names
FOCAL_LENGTH = 700

KNOWN_HEIGHTS = {
    'person': 1.7, 'car': 1.5, 'truck': 3.0, 'bus': 3.5, 
    'bicycle': 1.0, 'motorcycle': 1.2, 'traffic light': 3.0
}

class DetectXpressTransformer(VideoTransformerBase):
    def __init__(self):
        self.session_start = time.time()
        self.frame_count = 0
        self.object_tracker = {}
        
        # Expanded Metrics
        self.eyes_closed_frames = 0
        self.smiles_detected = 0
        self.stress_level = 0
        self.eco_score = 100
        
        # Voice Queue
        self.speech_queue = []
        self.last_spoken = {}
        
        # Timelines
        self.score_timeline = deque(maxlen=60)
        self.ear_timeline = deque(maxlen=60)
        self.distance_timeline = deque(maxlen=60)
        self.speed_timeline = deque(maxlen=60)
        self.object_counts = defaultdict(int)
        
        self.stats = {
            'total_detections': 0, 'critical_alerts': 0, 'drowsiness_alerts': 0,
            'cellphone_violations': 0, 'lane_departures': 0, 'red_light_violations': 0,
            'alert_history': deque(maxlen=100)
        }

    def speak(self, text, cooldown=10):
        t = time.time()
        if (t - self.last_spoken.get(text, 0)) > cooldown:
            self.speech_queue.append(text)
            self.last_spoken[text] = t

    def analyze_weather(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = gray.std()
        
        weather = "Clear"
        if brightness < 60: weather = "Night Mode"
        elif contrast < 35: weather = "Poor Visibility (Fog/Rain)"
        
        if weather == "Poor Visibility (Fog/Rain)":
            self.speak("Warning, poor visibility detected. Reduce speed.")
            
        return weather, brightness, contrast

    def detect_lane_departure(self, frame):
        h, w = frame.shape[:2]
        roi = frame[int(h*0.6):h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5: left_lines.append(line[0])
                elif slope > 0.5: right_lines.append(line[0])
                
        # Simple drift heuristic
        if len(left_lines) > 0 and len(right_lines) == 0: return "DRIFTING RIGHT"
        if len(right_lines) > 0 and len(left_lines) == 0: return "DRIFTING LEFT"
        return "CENTERED"

    def analyze_traffic_light(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        light_roi = frame[y1:y2, x1:x2]
        if light_roi.size == 0: return "Unknown"
        
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 70, 50]), np.array([90, 255, 255]))
        
        red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
        green_pixels = cv2.countNonZero(mask_green)
        
        if red_pixels > green_pixels and red_pixels > 20: return "RED"
        if green_pixels > red_pixels and green_pixels > 20: return "GREEN"
        return "YELLOW"

    def get_danger_level(self, distance, ttc):
        if ttc is not None and ttc > 0 and ttc < 3.0: return 'CRITICAL_TTC', (0, 0, 255)
        if distance is None: return 'UNKNOWN', (128, 128, 128)
        if distance < 5: return 'CRITICAL', (0, 0, 255)
        if distance < 15: return 'WARNING', (0, 165, 255)
        return 'SAFE', (0, 255, 0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        current_time = time.time()
        annotated = img.copy()
        
        # 1. Weather Analysis
        weather, bright, cont = self.analyze_weather(img)
        cv2.putText(annotated, f"Weather: {weather}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 2. Lane Departure Warning (LDW)
        lane_status = self.detect_lane_departure(img)
        if lane_status != "CENTERED":
            cv2.putText(annotated, f"LDW: {lane_status}", (annotated.shape[1]//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            self.stats['lane_departures'] += 1
            if self.frame_count % 30 == 0: self.speak("Lane departure warning")

        # YOLO Inference
        results = model(img, conf=0.40, iou=0.45, imgsz=1280, augment=True, verbose=False)
        
        detections = []
        closest_dist = None
        max_speed = 0
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = COCO_CLASSES[int(box.cls[0])]
                
                # 3. Phone Distraction Detection
                if cls_name == "cell phone":
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(annotated, "PHONE DISTRACTION!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    self.stats['cellphone_violations'] += 1
                    self.speak("Please put down your phone immediately.", cooldown=15)
                    continue
                
                # 4. Red Light Detection
                if cls_name == "traffic light":
                    light_state = self.analyze_traffic_light(img, [x1, y1, x2, y2])
                    color = (0,0,255) if light_state=="RED" else (0,255,0)
                    cv2.putText(annotated, f"LIGHT: {light_state}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if light_state == "RED":
                        self.speak("Red light ahead", cooldown=20)
                        self.stats['red_light_violations'] += 1
                    continue
                
                if cls_name not in KNOWN_HEIGHTS: continue
                
                center = ((x1+x2)//2, (y1+y2)//2)
                detections.append({'class': cls_name, 'bbox': [x1, y1, x2, y2], 'center': center})

        # Object Tracking & Physics
        current_objects = {}
        for idx, det in enumerate(detections):
            matched_id = None
            for obj_id, obj_data in self.object_tracker.items():
                if obj_data['class'] == det['class']:
                    dist = math.sqrt((det['center'][0] - obj_data['center'][0])**2 + (det['center'][1] - obj_data['center'][1])**2)
                    if dist < 100: matched_id = obj_id; break
            
            if matched_id:
                obj = self.object_tracker[matched_id]
                obj['history'].append(det['center'])
                obj['bbox'] = det['bbox']
                obj['center'] = det['center']
                current_objects[matched_id] = obj
            else:
                current_objects[f"{det['class']}_{self.frame_count}_{idx}"] = {'class': det['class'], 'history': [det['center']], 'bbox': det['bbox'], 'center': det['center']}

        self.object_tracker = current_objects

        for obj_id, obj in self.object_tracker.items():
            x1, y1, x2, y2 = obj['bbox']
            cls_name = obj['class']
            
            distance = (KNOWN_HEIGHTS.get(cls_name, 1.5) * FOCAL_LENGTH) / max(1, (y2 - y1))
            if closest_dist is None or distance < closest_dist: closest_dist = distance
            
            # Physics: Speed & TTC
            speed = 0
            ttc = None
            if len(obj['history']) >= 5:
                p1 = obj['history'][-5]
                p2 = obj['history'][-1]
                pixel_speed = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                speed = pixel_speed * 0.05
                if speed > max_speed: max_speed = speed
                
                # 5. Time-To-Collision (TTC)
                # If bounding box is expanding rapidly (y2-y1 is growing), object is approaching
                # Simplified TTC = Distance / Relative Speed
                if speed > 2.0: 
                    ttc = distance / speed
                    if ttc < 3.0: self.speak("Collision Imminent! Brake!", cooldown=5)
            
            # 6. Pedestrian Intent
            if cls_name == "person" and len(obj['history']) > 5:
                dx = obj['history'][-1][0] - obj['history'][0][0]
                # Moving towards center
                if (x1 < img.shape[1]//3 and dx > 10) or (x2 > 2*img.shape[1]//3 and dx < -10):
                    cv2.putText(annotated, "CROSSING RISK", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                    self.speak("Pedestrian crossing risk", cooldown=10)

            # 7. Blind Spot Monitor
            if speed > 5.0 and (x1 < 50 or x2 > img.shape[1]-50) and distance < 15:
                cv2.putText(annotated, "BLIND SPOT", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                self.speak("Vehicle in blind spot", cooldown=10)
                
            # 8. Eco-Driving Tracker
            if speed > 15: self.eco_score = max(0, self.eco_score - 0.1) # Penalize excessive relative speed
            
            danger_lvl, color = self.get_danger_level(distance, ttc)
            if danger_lvl in ['CRITICAL', 'CRITICAL_TTC']: self.stats['critical_alerts'] += 1
            
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, f"{cls_name} {distance:.1f}m", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if ttc and ttc < 10: cv2.putText(annotated, f"TTC: {ttc:.1f}s", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 9. Stress & Drowsiness Monitor
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        current_ear = 0.3
        
        for (x,y,w,h) in faces[:1]:
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi, 1.1, 5)
            smiles = smile_cascade.detectMultiScale(roi, 1.8, 20)
            
            if len(smiles) > 0: self.stress_level = max(0, self.stress_level - 1)
            else: self.stress_level = min(100, self.stress_level + 0.1)
            
            if self.stress_level > 80:
                cv2.putText(annotated, "HIGH STRESS", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            if len(eyes) >= 2:
                # Simplified EAR for demo
                current_ear = 0.3
            else:
                current_ear = 0.1
                
            if current_ear < 0.15:
                self.eyes_closed_frames += 1
                if self.eyes_closed_frames > DROWSY_THRESHOLD:
                    self.speak("Wake up!", cooldown=5)
                    self.stats['drowsiness_alerts'] += 1
            else:
                self.eyes_closed_frames = 0

        # Update Telemetry Arrays (1 per sec)
        if len(self.score_timeline) == 0 or (current_time - self.score_timeline[-1][0]) > 1.0:
            score = 100 - (self.stats['critical_alerts']*2) - (self.stats['cellphone_violations']*5)
            self.score_timeline.append((current_time, max(0, score)))
            self.ear_timeline.append((current_time, current_ear))
            self.distance_timeline.append((current_time, closest_dist if closest_dist else 50))
            self.speed_timeline.append((current_time, max_speed))

        # Dashboard HUD
        cv2.rectangle(annotated, (10, img.shape[0]-120), (350, img.shape[0]-10), (0,0,0), -1)
        cv2.putText(annotated, f"Eco Score: {int(self.eco_score)}", (20, img.shape[0]-90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(annotated, f"Stress Level: {int(self.stress_level)}%", (20, img.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        cv2.putText(annotated, f"Phone Violations: {self.stats['cellphone_violations']}", (20, img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

with st.sidebar:
    st.markdown("### 🏆 Award-Winning ADAS")
    st.success("✅ 1. TTC Physics Engine")
    st.success("✅ 2. Pedestrian Intent AI")
    st.success("✅ 3. Lane Departure (LDW)")
    st.success("✅ 4. Weather Analyzer")
    st.success("✅ 5. Red Light Detector")
    st.success("✅ 6. Blind Spot Monitor")
    st.success("✅ 7. Phone Distraction AI")
    st.success("✅ 8. Stress Monitor")
    st.success("✅ 9. Eco-Driving Scorer")
    st.success("✅ 10. Voice Synthesis")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    ctx = webrtc_streamer(
        key="detectxpress",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DetectXpressTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# Audio Injection Placeholder
audio_placeholder = st.empty()

st.markdown("---")
st.markdown("### 📈 Live Telemetry Dashboard")
c1, c2, c3 = st.columns(3)
score_box = c1.empty(); obj_box = c2.empty(); alert_box = c3.empty()
c4, c5, c6 = st.columns(3)
ear_box = c4.empty(); dist_box = c5.empty(); speed_box = c6.empty()

report_box = st.empty()

if ctx.state.playing:
    while True:
        if ctx.video_processor:
            proc = ctx.video_processor
            current_t = time.time()
            
            # Voice Synthesis JS Injection
            if len(proc.speech_queue) > 0:
                text_to_speak = proc.speech_queue.pop(0)
                js = f"<script>window.speechSynthesis.speak(new SpeechSynthesisUtterance('{text_to_speak}'));</script>"
                audio_placeholder.empty()
                with audio_placeholder:
                    components.html(js, height=0, width=0)
            
            st.session_state['report_data'] = {
                'start': proc.session_start, 'stats': proc.stats, 'eco': proc.eco_score,
                'stress': proc.stress_level, 'score_timeline': list(proc.score_timeline)
            }
            
            # Render Charts
            if len(proc.score_timeline) > 0:
                df = pd.DataFrame(list(proc.score_timeline), columns=['Time', 'Val'])
                df['Sec'] = (current_t - df['Time']).apply(lambda x: -round(x))
                with score_box.container():
                    st.markdown("#### 💯 Safety Score")
                    st.line_chart(df.set_index('Sec')['Val'], height=200, color="#38bdf8")
            
            with obj_box.container():
                st.markdown("#### 🚗 Eco-Driving Score")
                st.metric("Efficiency Rating", f"{int(proc.eco_score)}/100")
                
            with alert_box.container():
                st.markdown("#### ⚠️ Violations")
                st.metric("Cell Phone Distractions", proc.stats['cellphone_violations'])

            if len(proc.ear_timeline) > 0:
                df = pd.DataFrame(list(proc.ear_timeline), columns=['Time', 'Val'])
                df['Sec'] = (current_t - df['Time']).apply(lambda x: -round(x))
                with ear_box.container():
                    st.markdown("#### 👁️ Drowsiness (EAR)")
                    st.line_chart(df.set_index('Sec')['Val'], height=200, color="#fbbf24")
                    
            if len(proc.distance_timeline) > 0:
                df = pd.DataFrame(list(proc.distance_timeline), columns=['Time', 'Val'])
                df['Sec'] = (current_t - df['Time']).apply(lambda x: -round(x))
                with dist_box.container():
                    st.markdown("#### 📏 Closest Threat (m)")
                    st.line_chart(df.set_index('Sec')['Val'], height=200, color="#10b981")
                    
            if len(proc.speed_timeline) > 0:
                df = pd.DataFrame(list(proc.speed_timeline), columns=['Time', 'Val'])
                df['Sec'] = (current_t - df['Time']).apply(lambda x: -round(x))
                with speed_box.container():
                    st.markdown("#### ⚡ Driver Stress Level")
                    st.line_chart(df.set_index('Sec')['Val'] * 0 + proc.stress_level, height=200, color="#a855f7")

        time.sleep(1.0)

if 'report_data' in st.session_state:
    data = st.session_state['report_data']
    dur = int(time.time() - data['start'])
    report = f"--- DETECTXPRESS ULTIMATE DRIVING REPORT ---\nDuration: {dur}s\nViolations: {data['stats']['cellphone_violations']}\nEco Score: {int(data['eco'])}\n"
    with report_box.container():
        st.download_button("⬇️ Download Final Report", report, file_name="Award_Winning_Report.txt")
