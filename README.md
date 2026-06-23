![DetectXpress Cover](https://drive.google.com/uc?export=view&id=15pyink_QV06XZ1ra7MtEqTx1FgkyO0oS)

# 🏆 DetectXpress Ultimate Edition
### *Advanced Real-Time Driver Assistance System (ADAS)*

**🟢 Live Deployment:** [https://detect-xpress.streamlit.app/](https://detect-xpress.streamlit.app/)

DetectXpress Ultimate is a state-of-the-art, 10-feature intelligent ADAS Command Center. Powered by **YOLO11x** (the most advanced model in the YOLO series), it performs real-time physics calculations, driver behavioral auditing, and environmental telemetry straight from your web browser!

## 🌟 The 10 Advanced Features
1. **Time-To-Collision (TTC) Engine:** Calculates real-time collision risks based on vehicle bounding box scaling.
2. **Pedestrian Intent Prediction:** Identifies pedestrians near the center of the frame and issues alerts.
3. **Lane Departure Warning (LDW):** Tracks road bounds and alerts if the vehicle drifts out of lane.
4. **Weather & Visibility Analyzer:** Uses contrast and brightness heuristics to detect fog, rain, or night conditions.
5. **Red Light Violation Detector:** Correlates traffic lights with the vehicle's speed.
6. **Blind Spot Monitoring:** Tracks vehicles continuously occupying the far edges of the frame.
7. **Smartphone Distraction AI:** Detects if the driver is holding a cell phone.
8. **Road Rage & Stress Monitor:** Calculates stress percentage based on sudden braking, high speeds, and aggressive lane changes.
9. **Eco-Driving Efficiency Score:** Grades the driver's fuel efficiency out of 100 based on smooth acceleration and braking.
10. **Dynamic Voice Synthesis:** Real-time AI voice alerts powered by Web Speech API directly in the browser!

## 📄 Automated PDF Reporting
At the end of every driving session, you can download a **Highly Detailed Assessment Report (PDF)**. The report features:
* Executive Summary & Overall Safety Score
* Environmental & Behavioral Audits
* Critical Incident Logs
* **High-Resolution Telemetry Graphs:** (Safety Score, Drowsiness, and Target Speed vs Time) plotted natively with Matplotlib.

## 🚀 Built With
* **AI Engine:** Ultralytics YOLO11x
* **Frontend:** Streamlit & Streamlit-WebRTC
* **Data & Analytics:** Pandas, NumPy, Matplotlib
* **Reporting:** FPDF2

## 💻 Local Execution
To run this powerhouse locally:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_advanced.py
```
*(Note: Streamlit Community Cloud and local environments run `app_advanced.py` as the main entry point.)*
