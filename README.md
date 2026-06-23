<div align="center">

# 🏆 DetectXpress Ultimate Edition
### *Advanced Real-Time Driver Assistance System (ADAS)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detect-xpress.streamlit.app/)
[![YOLO11x](https://img.shields.io/badge/YOLO-11x-00FFFF?style=for-the-badge&logo=ai)](https://github.com/ultralytics/ultralytics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*A state-of-the-art, **10-feature** intelligent ADAS Command Center. Powered by the massive **YOLO11x** engine, it performs real-time physics calculations, driver behavioral auditing, and environmental telemetry directly from your web browser.*

</div>

---

## 🌟 The 10 Advanced Features

| Feature | Description | Physics / Logic Engine |
| :--- | :--- | :--- |
| 💥 **Time-To-Collision (TTC)** | Calculates real-time collision risks. | Tracks bounding box spatial expansion. |
| 🚶 **Pedestrian Intent** | Predicts if pedestrians will cross. | Maps central "danger zone" intersections. |
| 🛣️ **Lane Departure (LDW)** | Alerts if the vehicle drifts out of lane. | Tracks horizontal road bounds. |
| 🌤️ **Weather Analyzer** | Detects fog, rain, or night conditions. | Computes pixel contrast & brightness heuristics. |
| 🛑 **Red Light Violations** | Prevents running red lights. | Color-masks detected traffic lights. |
| 🚗 **Blind Spot Monitor** | Warns of adjacent vehicles. | Tracks extreme left/right frame matrices. |
| 📱 **Smartphone Distraction** | Detects if the driver is holding a phone. | YOLO object classification mapping. |
| 💢 **Road Rage Monitor** | Audits aggressive driving & stress. | Time-series analysis of velocity changes. |
| 🌱 **Eco-Driving Score** | Grades fuel efficiency (0-100). | Penalizes hard braking and heavy acceleration. |
| 🗣️ **Dynamic AI Voice** | Real-time audible safety alerts. | `window.speechSynthesis` API integration. |

---

## 📄 Automated PDF Reporting

At the end of every driving session, the system compiles the massive internal telemetry logs into a **Highly Detailed Assessment Report (PDF)**.

<div align="center">
  <img src="https://img.shields.io/badge/Data-FPDF2-red?style=flat-square&logo=pdf" />
  <img src="https://img.shields.io/badge/Charts-Matplotlib-orange?style=flat-square&logo=python" />
</div>

> **The report includes:**
> - 📋 **Executive Summary:** Overall Safety Score and Drive Duration.
> - 🚨 **Incident Logs:** Precise tallies of lane departures, blind spot interventions, and distractions.
> - 📈 **Telemetry Graphs:** High-resolution charts tracking *Safety Score*, *Micro-Sleeps (EAR)*, and *Target Speed* over time, plotted natively with Matplotlib.

---

## 🚀 The Technology Stack

<div align="center">

| Domain | Technologies |
| :---: | :---: |
| **AI & Physics** | `Ultralytics YOLO11x` • `PyTorch` • `NumPy` |
| **Computer Vision** | `OpenCV (Headless)` • `Haar Cascades` |
| **Web Frontend** | `Streamlit` • `HTML5` • `CSS3 Glassmorphism` |
| **Real-Time Data** | `Streamlit-WebRTC` • `Pandas` |

</div>

---

## 💻 Local Execution

To run this powerhouse locally on your own machine:

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install strictly versioned dependencies
pip install -r requirements.txt

# 3. Launch the Ultimate Command Center
streamlit run app_advanced.py
```

> **Note:** Both Streamlit Community Cloud and local environments utilize `app_advanced.py` as the primary entry point.

---
<div align="center">
  <i>Built for safety. Engineered for the future.</i>
</div>
