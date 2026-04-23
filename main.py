"""
LeLamp Challenge — Main Orchestration Loop
============================================
Ties together perception, behavior, memory, and conversation
into one real-time pipeline.

Run:  python main.py
Keys: ESC to quit, Q to ask the lamp a question

Pipeline:
  Webcam → Engagement Detection → Behavior State Machine → Lamp Animation
       └→ Scene Detection (periodic) → Memory Store → LLM Recall (on demand)
"""

import cv2
import time
import threading
import config
from perception.engagement import EngagementDetector, EngagementState
from perception.scene import SceneDetector
from simulation.lamp import LampSimulation, LampState
from memory.store import MemoryStore
from conversation.recall import RecallEngine


def main():
    # ── Initialise components ───────────────────────────────
    print("[LeLamp] Starting up...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        return

    detector = EngagementDetector(
        detection_confidence=config.FACE_DETECTION_CONFIDENCE,
        tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
        gaze_threshold=config.GAZE_ENGAGED_THRESHOLD,
    )

    scene = SceneDetector(
        model_name=config.YOLO_MODEL,
        confidence=config.YOLO_CONFIDENCE,
    )

    memory = MemoryStore(db_path=config.DB_PATH)

    recall = None
    if config.OPENAI_API_KEY:
        recall = RecallEngine(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
        )
        print("[LeLamp] Recall engine ready (OpenAI connected)")
    else:
        print("[LeLamp] WARNING: No OPENAI_API_KEY set — recall disabled")
        print("[LeLamp] Set it with: set OPENAI_API_KEY=sk-...")

    lamp = LampSimulation(
        width=config.WINDOW_WIDTH,
        height=config.WINDOW_HEIGHT,
    )

    # ── State tracking ──────────────────────────────────────
    disengaged_since = None
    hysteresis_counter = 0
    current_engagement = EngagementState.NO_FACE
    prev_time = time.time()
    last_detection_time = 0.0       # When we last ran YOLO
    asking_question = False         # True while user is typing a question

    print("[LeLamp] Ready — look at the camera to engage!")
    print("[LeLamp] Press Q in the Pygame window to ask a question.")
    print()

    # ── Main loop ───────────────────────────────────────────
    running = True
    while running:
        # Timing
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Pygame events (also checks for Q key)
        for event in _get_events():
            if event.type == __import__('pygame').QUIT:
                running = False
            elif event.type == __import__('pygame').KEYDOWN:
                if event.key == __import__('pygame').K_ESCAPE:
                    running = False
                elif event.key == __import__('pygame').K_q and not asking_question:
                    asking_question = True
                    _handle_question(lamp, recall, memory)
                    asking_question = False

        if not running:
            break

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # ── Perception: Engagement ──────────────────────────
        result = detector.detect(frame)

        if result.state == current_engagement:
            hysteresis_counter = 0
        else:
            hysteresis_counter += 1
            if hysteresis_counter >= config.ENGAGEMENT_HYSTERESIS_FRAMES:
                current_engagement = result.state
                hysteresis_counter = 0

        # ── Perception: Scene Detection (periodic) ──────────
        if now - last_detection_time >= config.DETECTION_INTERVAL_SEC:
            last_detection_time = now
            detections = scene.detect(frame)
            if detections:
                new_count = memory.store_detections(detections)
                labels = [d.label for d in detections]
                stats = memory.get_stats()
                print(f"[Scene] Detected: {labels} "
                      f"({new_count} new | {stats['total_entries']} total memories)")

        # ── Behavior (state machine) ────────────────────────
        if current_engagement == EngagementState.ENGAGED:
            lamp.set_state(LampState.ENGAGED)
            stats = memory.get_stats()
            lamp.status_text = (f"User engaged — hello! "
                                f"(memories: {stats['total_entries']})")
            disengaged_since = None

        elif current_engagement == EngagementState.DISENGAGED:
            if disengaged_since is None:
                disengaged_since = now

            elapsed = now - disengaged_since
            if elapsed > config.ATTENTION_SEEK_DELAY_SEC:
                lamp.set_state(LampState.ATTENTION_SEEKING)
                lamp.status_text = "Trying to get your attention..."
            else:
                lamp.set_state(LampState.IDLE)
                remaining = config.ATTENTION_SEEK_DELAY_SEC - elapsed
                lamp.status_text = f"User looked away ({remaining:.1f}s to seek)"

        else:  # NO_FACE
            lamp.set_state(LampState.IDLE)
            lamp.status_text = "No face detected — waiting..."
            disengaged_since = None

        # ── Simulation update & render ──────────────────────
        lamp.update(dt)

        debug_frame = detector.draw_debug(frame, result)
        pip_surface = lamp.frame_to_surface(debug_frame)

        lamp.render(webcam_surface=pip_surface)
        lamp.clock.tick(config.FPS)

    # ── Cleanup ─────────────────────────────────────────────
    print("[LeLamp] Shutting down...")
    cap.release()
    detector.close()
    memory.close()
    lamp.close()


def _get_events():
    """Get Pygame events without importing pygame at module level."""
    import pygame
    return pygame.event.get()


def _handle_question(lamp, recall, memory):
    """
    Pause the loop, get a question from the terminal, query the LLM,
    and display the answer. Lamp switches to LISTENING state while waiting.
    """
    lamp.set_state(LampState.LISTENING)
    lamp.status_text = "Listening... type your question in the terminal"
    lamp.update(0.016)
    lamp.render()
    import pygame
    pygame.display.flip()

    if recall is None:
        print("\n[LeLamp] Recall is disabled (no OPENAI_API_KEY).")
        print("[LeLamp] Current memories:")
        for entry in memory.get_recent(10):
            print(f"  {entry}")
        return

    print("\n" + "=" * 50)
    print("Ask LeLamp a question (or press Enter to cancel):")
    question = input("> ").strip()

    if not question:
        print("[LeLamp] Cancelled.")
        return

    print("[LeLamp] Thinking...")
    answer = recall.answer(question, memory)
    print(f"\n💡 LeLamp: {answer}\n")
    print("=" * 50)

    # Show answer on lamp display briefly
    lamp.status_text = f"LeLamp: {answer[:60]}..."
    lamp.update(0.016)
    lamp.render()
    pygame.display.flip()
    import time as _time
    _time.sleep(2)


if __name__ == "__main__":
    main()
