"""
Engagement Detection Module
============================
Uses MediaPipe Face Mesh with iris landmarks to determine
whether the user is looking at the camera (engaged) or away (disengaged).

Design decision: MediaPipe over dlib/Haar cascades because it provides
478 landmarks including iris tracking (landmarks 468-477), enabling
accurate gaze estimation without a separate model.
"""

import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from dataclasses import dataclass


class EngagementState(Enum):
    NO_FACE = "no_face"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"


@dataclass
class EngagementResult:
    """Result of a single engagement detection frame."""
    state: EngagementState
    confidence: float
    face_center: tuple | None       # Normalized (x, y) of face center
    gaze_ratio: float | None        # 0 = looking away, 1 = looking directly


class EngagementDetector:
    """
    Detects user engagement via gaze direction.

    Uses MediaPipe Face Mesh with `refine_landmarks=True` to access
    iris landmarks. Computes how centered each iris is within its
    eye opening — if centered, the user is looking at the camera.
    """

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5,
                 gaze_threshold=0.28):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # Enables iris landmarks
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.gaze_threshold = gaze_threshold

    # ── Public API ──────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> EngagementResult:
        """
        Analyze a BGR frame and return engagement state.

        Args:
            frame: BGR image from webcam (numpy array).

        Returns:
            EngagementResult with state, confidence, and gaze info.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return EngagementResult(
                state=EngagementState.NO_FACE,
                confidence=1.0,
                face_center=None,
                gaze_ratio=None,
            )

        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Face center via nose tip (landmark 1)
        nose = landmarks.landmark[1]
        face_center = (nose.x, nose.y)

        # Gaze ratio from iris positions
        gaze_ratio = self._compute_gaze_ratio(landmarks, w, h)

        if gaze_ratio is not None and gaze_ratio > (1.0 - self.gaze_threshold):
            state = EngagementState.ENGAGED
            confidence = gaze_ratio
        else:
            state = EngagementState.DISENGAGED
            confidence = 1.0 - (gaze_ratio or 0.0)

        return EngagementResult(
            state=state,
            confidence=confidence,
            face_center=face_center,
            gaze_ratio=gaze_ratio,
        )

    def draw_debug(self, frame: np.ndarray, result: EngagementResult) -> np.ndarray:
        """Draw engagement debug overlay on the frame."""
        debug = frame.copy()
        color_map = {
            EngagementState.NO_FACE: (128, 128, 128),
            EngagementState.ENGAGED: (0, 255, 100),
            EngagementState.DISENGAGED: (0, 100, 255),
        }
        color = color_map[result.state]

        cv2.rectangle(debug, (0, 0), (debug.shape[1], 35), (0, 0, 0), -1)
        if result.gaze_ratio is not None:
            text = f"{result.state.value.upper()} | Gaze: {result.gaze_ratio:.2f}"
        else:
            text = result.state.value.upper()
        cv2.putText(debug, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if result.confidence:
            bar_w = int(200 * result.confidence)
            cv2.rectangle(debug, (debug.shape[1] - 220, 8),
                          (debug.shape[1] - 220 + bar_w, 28), color, -1)
        return debug

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()

    # ── Private helpers ─────────────────────────────────────

    def _compute_gaze_ratio(self, landmarks, w: int, h: int) -> float | None:
        """
        Average gaze ratio across both eyes.
        Returns 0.0 (looking away) to 1.0 (direct eye contact).
        """
        try:
            left = self._eye_gaze_ratio(
                landmarks, iris_center=468,
                inner_corner=133, outer_corner=33, w=w
            )
            right = self._eye_gaze_ratio(
                landmarks, iris_center=473,
                inner_corner=362, outer_corner=263, w=w
            )
            return (left + right) / 2.0
        except (IndexError, ZeroDivisionError):
            return None

    @staticmethod
    def _eye_gaze_ratio(landmarks, iris_center: int,
                        inner_corner: int, outer_corner: int,
                        w: int) -> float:
        """
        For one eye: how centered is the iris between inner/outer corners?
        1.0 = perfectly centered (looking at camera), 0.0 = at the edge.
        """
        iris_x = landmarks.landmark[iris_center].x * w
        inner_x = landmarks.landmark[inner_corner].x * w
        outer_x = landmarks.landmark[outer_corner].x * w

        eye_center_x = (inner_x + outer_x) / 2.0
        eye_width = abs(outer_x - inner_x)
        if eye_width == 0:
            return 0.5

        offset = abs(iris_x - eye_center_x) / (eye_width / 2.0)
        return max(0.0, min(1.0, 1.0 - offset))
