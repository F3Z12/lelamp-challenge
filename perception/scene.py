"""
Scene Understanding Module
============================
Uses YOLOv8-nano for lightweight, real-time object detection.

Design decisions:
- YOLOv8n (nano): 3.2M params, ~6ms/frame on GPU, ~40ms on CPU.
  Good enough for demo; larger models add latency without helping recall quality.
- We normalize all bounding boxes to 0-1 range so spatial descriptions
  are resolution-independent.
- The model is loaded once at init and reused for every detection sweep.
"""

import numpy as np
from ultralytics import YOLO
from memory.models import Detection


class SceneDetector:
    """
    Detects objects in a camera frame using YOLOv8.

    Usage:
        detector = SceneDetector("yolov8n.pt", confidence=0.5)
        detections = detector.detect(frame)
    """

    def __init__(self, model_name: str = "yolov8n.pt",
                 confidence: float = 0.5):
        print(f"[SceneDetector] Loading {model_name}...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        print(f"[SceneDetector] Ready ({len(self.model.names)} classes)")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run object detection on a BGR frame.

        Returns a list of Detection objects with normalized coordinates,
        filtered to only high-confidence results.
        """
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,      # Suppress per-frame logging
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        h, w = frame.shape[:2]

        for box in r.boxes:
            # Extract box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.model.names.get(cls_id, f"class_{cls_id}")

            # Normalize to 0-1 range
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            detections.append(Detection(
                label=label,
                confidence=conf,
                x=cx, y=cy,
                width=bw, height=bh,
            ))

        return detections

    def detect_and_annotate(self, frame: np.ndarray):
        """
        Run detection and return (detections, annotated_frame).
        The annotated frame has bounding boxes drawn for debug/PiP display.
        """
        import cv2

        detections = self.detect(frame)
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for det in detections:
            # Convert normalized back to pixel coords for drawing
            cx, cy = int(det.x * w), int(det.y * h)
            bw, bh = int(det.width * w / 2), int(det.height * h / 2)
            x1, y1 = cx - bw, cy - bh
            x2, y2 = cx + bw, cy + bh

            color = (100, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{det.label} {det.confidence:.0%}"
            cv2.putText(annotated, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return detections, annotated
