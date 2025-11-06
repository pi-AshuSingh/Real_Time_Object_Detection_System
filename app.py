from flask import Flask, render_template, Response, request
import os
import cv2
from ultralytics import YOLO  # ✅ Use YOLOv8 instead of torch.hub

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Load YOLOv8 model
# 'yolov8s.pt' is small and fast. You can try 'yolov8m.pt' for higher accuracy.
model = YOLO("yolov8s.pt")

# Open webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 🔁 Flip for correct orientation
            frame = cv2.flip(frame, 1)

            # ✅ Run YOLOv8 detection
            results = model(frame, stream=True)

            for r in results:
                annotated_frame = r.plot()

            # Encode and stream frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Upload and detect image
@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    results = model(filepath)
    results[0].save(filename=os.path.join(app.config['UPLOAD_FOLDER'], "detected_" + file.filename))
    return f"✅ Image '{file.filename}' processed! Check 'static/uploads' for result."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
