"""
Example Usage of YOLOv11 Model Integration

This file demonstrates how to integrate the advanced YOLOv11 model
into your Flask application or standalone scripts.

YOLOv11 advantages over YOLOv11:
- 22% fewer parameters (faster loading, less memory)
- 20% faster inference speed
- Higher mAP (better accuracy)
- Better small object detection
"""

from model import YOLOv11Detector, detect_objects
import cv2
import time


def example_1_basic_detection():
    """Example 1: Basic object detection on webcam"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Webcam Detection")
    print("="*60)
    
    # Initialize detector with optimal settings for real-time
    detector = YOLOv11Detector(
        model_size='small',      # Use 'small' for best speed/accuracy balance
        conf_threshold=0.35,     # Confidence threshold
        iou_threshold=0.45,      # Non-max suppression threshold
        use_fp16=True,           # Use FP16 for faster inference (NVIDIA GPU)
        verbose=True
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("\n▶️  Starting detection... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and draw bounding boxes
        annotated_frame, detections = detector.detect_and_draw(
            frame,
            preprocess=True,      # Apply image enhancement
            show_conf=True,       # Show confidence scores
            show_class=True,      # Show class names
            line_thickness=2
        )
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Draw FPS on frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display
        cv2.imshow('YOLOv11 Detection', annotated_frame)
        
        # Print detections (every 30 frames)
        if frame_count % 30 == 0 and detections:
            print(f"\n📊 Frame {frame_count} - Detected {len(detections)} objects:")
            for det in detections[:5]:  # Show first 5
                print(f"  • {det['class_name']}: {det['confidence']:.2f}")
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processed {frame_count} frames")
    print(f"📈 Average FPS: {fps:.2f}")


def example_2_object_tracking():
    """Example 2: Object tracking with unique IDs"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Object Tracking")
    print("="*60)
    
    # Initialize detector
    detector = YOLOv11Detector(
        model_size='small',
        conf_threshold=0.4,
        verbose=True
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("\n▶️  Starting tracking... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects (maintains IDs across frames)
        tracked_objects = detector.track_objects(
            frame,
            persist=True,           # Maintain tracks
            tracker="bytetrack.yaml"  # Use ByteTrack algorithm
        )
        
        # Draw tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            track_id = obj['track_id']
            class_name = obj['class_name']
            conf = obj['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with track ID
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        cv2.imshow('YOLOv11 Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_3_image_detection():
    """Example 3: Detect objects in an image file"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Image File Detection")
    print("="*60)
    
    # Use convenience function for quick detection
    image_path = "test_image.jpg"  # Replace with your image path
    
    try:
        detections = detect_objects(
            image_path=image_path,
            model_size='medium',     # Use larger model for better accuracy
            conf_threshold=0.3,
            save_path="result.jpg"   # Save annotated image
        )
        
        print(f"\n✅ Detected {len(detections)} objects:")
        for det in detections:
            print(f"  • {det['class_name']}: {det['confidence']:.2f}")
        
        print(f"\n💾 Annotated image saved to: result.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_custom_model():
    """Example 4: Use custom trained YOLOv11 model"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Model")
    print("="*60)
    
    # Initialize with custom model
    detector = YOLOv11Detector(
        custom_model_path="path/to/your/custom_model.pt",  # Your custom model
        conf_threshold=0.5,
        verbose=True
    )
    
    print("\n✅ Custom model loaded!")
    print(f"📋 Classes: {list(detector.class_names.values())}")


def example_5_performance_comparison():
    """Example 5: Compare different model sizes"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Model Performance Comparison")
    print("="*60)
    
    model_sizes = ['nano', 'small', 'medium']
    
    # Load test frame
    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Cannot capture test frame")
        return
    
    print("\n🔬 Testing different model sizes on same frame...\n")
    
    for size in model_sizes:
        print(f"Testing {size} model...")
        
        # Initialize detector
        detector = YOLOv11Detector(
            model_size=size,
            conf_threshold=0.3,
            verbose=False
        )
        
        # Warm-up
        for _ in range(3):
            detector.detect(test_frame, preprocess=False)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            detections = detector.detect(test_frame, preprocess=False)
        
        elapsed = time.time() - start_time
        avg_time = (elapsed / num_runs) * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"  • {size.capitalize():8s}: {avg_time:.2f}ms/frame ({fps:.1f} FPS)")
        print(f"    Detections: {len(detections)}")
    
    print("\n✅ Benchmark complete!")


def example_6_flask_integration():
    """Example 6: Integration with Flask (code snippet)"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Flask Integration")
    print("="*60)
    
    code = """
from flask import Flask, Response
from model import YOLOv11Detector
import cv2

app = Flask(__name__)

# Initialize detector once (global)
detector = YOLOv11Detector(
    model_size='small',
    conf_threshold=0.35,
    use_fp16=True,
    verbose=True
)

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Detect and draw
        annotated_frame, detections = detector.detect_and_draw(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\\r\\n'
               b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True)
"""
    
    print("\n📝 Flask Integration Code:")
    print(code)


def main():
    """Main menu"""
    print("\n" + "="*60)
    print("🚀 YOLOv11 Advanced Model - Usage Examples")
    print("="*60)
    
    examples = {
        '1': ('Basic Webcam Detection', example_1_basic_detection),
        '2': ('Object Tracking', example_2_object_tracking),
        '3': ('Image File Detection', example_3_image_detection),
        '4': ('Custom Model', example_4_custom_model),
        '5': ('Performance Comparison', example_5_performance_comparison),
        '6': ('Flask Integration', example_6_flask_integration),
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Exit")
    
    while True:
        choice = input("\n👉 Select example (0-6): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        if choice in examples:
            name, func = examples[choice]
            try:
                func()
            except KeyboardInterrupt:
                print("\n\n⏸️  Interrupted by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
