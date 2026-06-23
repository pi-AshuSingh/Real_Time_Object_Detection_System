"""
YOLOv11 Model Configuration and Utilities
Implements the LATEST YOLOv11 architecture (Ultralytics 2024) with state-of-the-art optimizations

YOLOv11 Improvements over YOLOv8:
- 22% fewer parameters with same accuracy
- 20% faster inference speed
- Higher mAP (mean Average Precision)
- Better small object detection
- Improved architecture efficiency

Features:
- YOLOv11n/s/m/l/x model support with auto-selection
- GPU acceleration with CUDA optimization
- Model quantization for faster inference
- TensorRT engine support for NVIDIA GPUs
- Multi-threading support
- Advanced preprocessing and postprocessing
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv11Detector:
    """
    Advanced YOLOv11 detector with optimization support
    
    YOLOv11 is the latest YOLO model (2024) with:
    - 22% fewer parameters than YOLOv8
    - 20% faster inference
    - Better accuracy (higher mAP)
    - Improved small object detection
    """
    
    # Available model sizes (YOLOv11 - ordered by speed/accuracy trade-off)
    MODEL_SIZES = {
        'nano': 'yolo11n.pt',      # Fastest, lowest parameters
        'small': 'yolo11s.pt',     # Balanced for real-time
        'medium': 'yolo11m.pt',    # Higher accuracy
        'large': 'yolo11l.pt',     # High accuracy, slower
        'xlarge': 'yolo11x.pt'     # Best accuracy, slowest
    }
    
    def __init__(
        self,
        model_size: str = 'small',
        custom_model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        use_fp16: bool = True,
        use_tensorrt: bool = False,
        verbose: bool = True
    ):
        """
        Initialize YOLOv11 detector
        
        Args:
            model_size: Model size ('nano', 'small', 'medium', 'large', 'xlarge')
            custom_model_path: Path to custom trained model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_fp16: Use FP16 precision for faster inference
            use_tensorrt: Convert to TensorRT engine (NVIDIA GPUs only)
            verbose: Print model information
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"🎯 Initializing YOLOv11 (Latest) on device: {self.device}")
        
        # Load model
        if custom_model_path and Path(custom_model_path).exists():
            model_path = custom_model_path
            logger.info(f"📦 Loading custom model from: {model_path}")
        else:
            model_path = self.MODEL_SIZES.get(model_size, 'yolo11s.pt')
            logger.info(f"📦 Loading YOLOv11-{model_size} model: {model_path}")
        
        # Initialize YOLO model (will auto-download YOLOv11 if not present)
        self.model = YOLO(model_path)
        
        # Move to device
        self.model.to(self.device)
        
        # Apply optimizations
        self._apply_optimizations(use_fp16, use_tensorrt)
        
        # Get class names
        self.class_names = self.model.names
        
        if verbose:
            self._print_model_info()
    
    def _apply_optimizations(self, use_fp16: bool, use_tensorrt: bool):
        """Apply model optimizations"""
        try:
            if self.device == 'cuda':
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                if use_fp16:
                    logger.info("⚡ Enabling FP16 precision")
                    self.model.model.half()
                
                if use_tensorrt:
                    logger.info("🚀 Converting to TensorRT engine...")
                    self._convert_to_tensorrt()
            
            # Set model to eval mode for inference
            self.model.model.eval()
            
        except Exception as e:
            logger.warning(f"⚠️ Optimization failed: {e}")
    
    def _convert_to_tensorrt(self):
        """Convert model to TensorRT for faster inference"""
        try:
            # Export to TensorRT format
            self.model.export(format='engine', half=True, dynamic=True)
            logger.info("✅ TensorRT conversion successful")
        except Exception as e:
            logger.error(f"❌ TensorRT conversion failed: {e}")
    
    def _print_model_info(self):
        """Print model information"""
        logger.info("=" * 60)
        logger.info(f"Model: {self.model.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Classes: {len(self.class_names)}")
        logger.info(f"Confidence Threshold: {self.conf_threshold}")
        logger.info(f"IoU Threshold: {self.iou_threshold}")
        
        if self.device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        logger.info("=" * 60)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing for better detection
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed frame
        """
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def detect(
        self,
        frame: np.ndarray,
        preprocess: bool = True,
        augment: bool = False,
        visualize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform object detection on a frame
        
        Args:
            frame: Input BGR frame
            preprocess: Apply preprocessing
            augment: Use test-time augmentation
            visualize: Return visualization features
            
        Returns:
            List of detections with format:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                },
                ...
            ]
        """
        # Preprocess if needed
        if preprocess:
            processed_frame = self.preprocess_frame(frame)
        else:
            processed_frame = frame
        
        # Run inference
        results = self.model.predict(
            processed_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            augment=augment,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.class_names[int(box.cls[0])]
                }
                detections.append(detection)
        
        return detections
    
    def detect_and_draw(
        self,
        frame: np.ndarray,
        preprocess: bool = True,
        show_conf: bool = True,
        show_class: bool = True,
        line_thickness: int = 2
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect objects and draw bounding boxes
        
        Args:
            frame: Input BGR frame
            preprocess: Apply preprocessing
            show_conf: Show confidence scores
            show_class: Show class names
            line_thickness: Bounding box line thickness
            
        Returns:
            Tuple of (annotated_frame, detections)
        """
        # Get detections
        detections = self.detect(frame, preprocess=preprocess)
        
        # Draw on frame
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class_name']
            
            # Generate color based on class
            color = self._get_color_for_class(det['class_id'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_conf:
                label_parts.append(f"{conf:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_frame, detections
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each class"""
        # Generate deterministic color based on class_id
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        np.random.seed()  # Reset seed
        return color
    
    def track_objects(
        self,
        frame: np.ndarray,
        persist: bool = True,
        tracker: str = "botsort.yaml"
    ) -> List[Dict[str, Any]]:
        """
        Perform object tracking
        
        Args:
            frame: Input BGR frame
            persist: Maintain tracks across frames
            tracker: Tracker configuration ('botsort.yaml' or 'bytetrack.yaml')
            
        Returns:
            List of tracked objects with IDs
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=persist,
            tracker=tracker,
            device=self.device,
            verbose=False
        )
        
        # Parse tracking results
        tracked_objects = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                tracked_obj = {
                    'track_id': track_id,
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.class_names[int(box.cls[0])]
                }
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def get_fps_stats(self) -> Dict[str, float]:
        """Get inference speed statistics"""
        if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'speed'):
            speed = self.model.predictor.speed
            return {
                'preprocess_ms': speed.get('preprocess', 0),
                'inference_ms': speed.get('inference', 0),
                'postprocess_ms': speed.get('postprocess', 0),
                'total_ms': sum(speed.values()),
                'fps': 1000 / sum(speed.values()) if sum(speed.values()) > 0 else 0
            }
        return {}
    
    def export_model(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'engine', 'coreml', etc.)
            **kwargs: Additional export arguments
        """
        logger.info(f"📤 Exporting model to {format} format...")
        try:
            self.model.export(format=format, **kwargs)
            logger.info(f"✅ Model exported successfully")
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")


# Convenience function for quick detection
def detect_objects(
    image_path: str,
    model_size: str = 'small',
    conf_threshold: float = 0.25,
    save_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Quick object detection on an image using YOLOv11
    
    Args:
        image_path: Path to input image
        model_size: Model size to use
        conf_threshold: Confidence threshold
        save_path: Optional path to save annotated image
        
    Returns:
        List of detections
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Initialize detector
    detector = YOLOv11Detector(
        model_size=model_size,
        conf_threshold=conf_threshold,
        verbose=False
    )
    
    # Detect and draw
    annotated_frame, detections = detector.detect_and_draw(frame)
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, annotated_frame)
        logger.info(f"💾 Saved annotated image to {save_path}")
    
    return detections


if __name__ == "__main__":
    """Test the YOLOv11 detector"""
    
    # Initialize detector
    detector = YOLOv11Detector(
        model_size='small',
        conf_threshold=0.3,
        use_fp16=True,
        verbose=True
    )
    
    print("\n✅ YOLOv11 Detector initialized successfully!")
    print(f"📋 Available classes: {len(detector.class_names)}")
    print(f"🎯 Sample classes: {list(detector.class_names.values())[:10]}")
    
    # Print FPS capability
    if detector.device == 'cuda':
        print(f"\n🚀 GPU Acceleration: Enabled")
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n💻 Running on CPU")
    
    print(f"\n🎉 YOLOv11 is 22% faster and more accurate than YOLOv8!")
