"""
LeLamp Challenge — Central Configuration
All tunable parameters in one place for easy experimentation.
"""
import os

# ── OpenAI ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective for recall queries

# ── Perception: Engagement Detection ────────────────
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5
GAZE_ENGAGED_THRESHOLD = 0.28  # How centered iris must be (lower = stricter)
ENGAGEMENT_HYSTERESIS_FRAMES = 10  # Frames before state change (debounce)

# ── Perception: Object Detection ────────────────────
YOLO_MODEL = "yolov8n.pt"  # Nano model — fast, lightweight
YOLO_CONFIDENCE = 0.5       # Only store high-confidence detections
DETECTION_INTERVAL_SEC = 3.0  # Seconds between detection sweeps

# ── Behavior ────────────────────────────────────────
ATTENTION_SEEK_DELAY_SEC = 5.0  # Seconds of disengagement before seeking
IDLE_TIMEOUT_SEC = 30.0         # Seconds before deep idle

# ── Simulation / Display ────────────────────────────
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 700
FPS = 30
WEBCAM_PIP_SIZE = (240, 180)  # Picture-in-picture webcam size

# ── Memory / Database ──────────────────────────────
DB_PATH = "memory.db"
