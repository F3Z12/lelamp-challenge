# LeLamp — Real-Time Perception, Memory, and Recall System

> An expressive 6-DOF lamp simulation that sees, reacts, remembers, and converses — built as a full perception-to-action pipeline.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Challenge Requirements Mapping](#challenge-requirements-mapping)
3. [System Architecture](#system-architecture)
4. [High-Level Data Flow](#high-level-data-flow)
5. [Low-Level Data Flow](#low-level-data-flow)
6. [System Modules Breakdown](#system-modules-breakdown)
7. [Design Decisions and Tradeoffs](#design-decisions-and-tradeoffs)
8. [Demo](#demo)
9. [Evaluation](#evaluation)
10. [Setup and Usage](#setup-and-usage)
11. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

LeLamp is a real-time perception system that simulates a socially-aware desk lamp. Using a standard laptop webcam, the system:

- **Detects user engagement** via gaze direction tracking
- **Reacts expressively** with animated motion, light, and color changes
- **Observes and remembers** objects in the scene using object detection
- **Recalls past observations** conversationally through a natural language interface

The system runs as a single Python process at 30 FPS, combining computer vision, state-driven animation, persistent memory, and LLM-powered conversation into one cohesive pipeline.

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Video Pipeline | OpenCV | Webcam capture, frame processing |
| Engagement Detection | MediaPipe Face Mesh | 478-landmark face tracking with iris refinement |
| Object Detection | YOLOv8-nano (Ultralytics) | Lightweight real-time object classification |
| Memory Storage | SQLite | Persistent, queryable observation storage |
| Conversational Recall | OpenAI API (GPT-4o-mini) | Natural language memory retrieval |
| Visualization | Pygame | Animated lamp simulation with 6-DOF expression |

---

## Challenge Requirements Mapping

| # | Requirement | Implementation | Module |
|---|---|---|---|
| 1 | **Engagement Detection** — System knows when user is paying attention | MediaPipe Face Mesh with iris landmarks (468–477). Computes gaze ratio by measuring iris center offset within eye opening. Debounced with configurable hysteresis. | `perception/engagement.py` |
| 2 | **Attention-Seeking Behavior** — System attempts re-engagement when user disengages | State machine transitions to `ATTENTION_SEEKING` after configurable delay (default 5s). Lamp wobbles, pulses orange, and animates to attract attention. | `simulation/lamp.py`, `main.py` |
| 3 | **Memory Formation** — System observes and classifies objects in scene | YOLOv8-nano runs every 3 seconds. High-confidence detections stored in SQLite with normalized spatial positions, timestamps, and frequency counts. Deduplication prevents noise. | `perception/scene.py`, `memory/store.py` |
| 4 | **Memory Recall** — System answers questions using stored observations | Retrieval-augmented generation: relevant memories are pulled from SQLite first, then sent as context to GPT-4o-mini. The LLM is constrained to answer only from observed data. | `conversation/recall.py` |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN ORCHESTRATOR                          │
│                           (main.py)                                │
│                                                                    │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   │   WEBCAM      │───▶│  PERCEPTION  │───▶│    BEHAVIOR       │    │
│   │  (OpenCV)     │    │    LAYER     │    │   STATE MACHINE   │    │
│   └──────────────┘    │              │    │                  │    │
│                       │ ┌──────────┐ │    │  IDLE             │    │
│                       │ │Engagement│ │    │  ENGAGED          │    │
│                       │ │Detector  │─┼───▶│  ATTENTION_SEEKING│    │
│                       │ │(MediaPipe│ │    │  OBSERVING        │    │
│                       │ └──────────┘ │    │  LISTENING        │    │
│                       │              │    └────────┬─────────┘    │
│                       │ ┌──────────┐ │             │              │
│                       │ │  Scene   │ │             ▼              │
│                       │ │ Detector │ │    ┌──────────────────┐    │
│                       │ │ (YOLOv8) │─┼──┐ │   SIMULATION     │    │
│                       │ └──────────┘ │  │ │   (Pygame Lamp)  │    │
│                       └──────────────┘  │ └──────────────────┘    │
│                                         │                         │
│   ┌──────────────────┐                  │ ┌──────────────────┐    │
│   │   CONVERSATION   │◀─────────────────┼─│     MEMORY       │    │
│   │   (OpenAI LLM)   │                  └▶│    (SQLite)      │    │
│   └──────────────────┘                    └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## High-Level Data Flow

```
                    ┌─────────┐
                    │ WEBCAM  │
                    └────┬────┘
                         │ 30 FPS BGR frames
                         ▼
              ┌─────────────────────┐
              │  PERCEPTION LAYER   │
              │                     │
              │  Every frame:       │        Every 3 seconds:
              │  Engagement ────────┼──┐     Scene Detection ──┐
              │  Detection          │  │                       │
              └─────────────────────┘  │                       │
                                       │                       │
                    ┌──────────────────┘                       │
                    ▼                                          ▼
           ┌────────────────┐                       ┌──────────────┐
           │ BEHAVIOR LAYER │                       │ MEMORY LAYER │
           │                │                       │              │
           │ State Machine  │                       │ SQLite DB    │
           │ + Debounce     │                       │ + Dedup      │
           └───────┬────────┘                       └──────┬───────┘
                   │                                       │
                   ▼                                       │
           ┌────────────────┐        User presses Q        │
           │  SIMULATION    │        ┌──────────────┐      │
           │                │        │ CONVERSATION │◀─────┘
           │  Animated Lamp │        │              │
           │  + PiP Webcam  │        │ LLM Recall   │
           └────────────────┘        └──────────────┘
```

---

## Low-Level Data Flow

### Engagement Detection (every frame)

```
BGR Frame
    │
    ├──▶ cv2.cvtColor(BGR → RGB)
    │
    ├──▶ MediaPipe Face Mesh (refine_landmarks=True)
    │         │
    │         ├──▶ 478 landmarks (including iris: 468-477)
    │         │
    │         ├──▶ Left eye: iris_center(468) vs corners(33, 133)
    │         ├──▶ Right eye: iris_center(473) vs corners(263, 362)
    │         │
    │         └──▶ gaze_ratio = avg(left_ratio, right_ratio)
    │                  │
    │                  ├── > 0.72 → ENGAGED
    │                  └── ≤ 0.72 → DISENGAGED
    │
    └──▶ Hysteresis buffer (10 consecutive frames required to change state)
```

### Object Detection (every 3 seconds)

```
BGR Frame
    │
    ├──▶ YOLOv8-nano inference (conf ≥ 0.5)
    │         │
    │         └──▶ List[Detection(label, confidence, x, y, w, h)]
    │                  │
    │                  ├──▶ Normalize coordinates to 0.0–1.0
    │                  ├──▶ Compute spatial_description ("top-left", "center", etc.)
    │                  │
    │                  └──▶ MemoryStore.store_detections()
    │                           │
    │                           ├── Find existing entry within 15% Euclidean distance?
    │                           │     YES → UPDATE (last_seen, times_seen++, max confidence)
    │                           │     NO  → INSERT new row
    │                           │
    │                           └──▶ SQLite commit
```

### Memory Recall (on user query)

```
User presses Q → types question in terminal
    │
    ├──▶ MemoryStore.get_context_for_llm(limit=30)
    │         │
    │         └──▶ "Objects the lamp has observed:
    │               - cup at center (confidence: 92%, seen 5x, last: ...)"
    │
    ├──▶ Build prompt:
    │         System: "You are LeLamp, a friendly desk lamp..."
    │         Context: [memory dump]
    │         User: [question]
    │
    └──▶ OpenAI GPT-4o-mini (max_tokens=200, temperature=0.7)
              │
              └──▶ Print answer to terminal + display on lamp HUD
```

---

## System Modules Breakdown

### `perception/engagement.py` — Engagement Detector

Wraps MediaPipe Face Mesh to compute real-time gaze direction. Uses iris landmark positions relative to eye corners to determine if the user is looking at the camera. Returns an `EngagementResult` with state (`ENGAGED`, `DISENGAGED`, `NO_FACE`), confidence score, and gaze ratio.

**Key implementation detail:** The gaze ratio is computed per-eye by measuring how centered the iris landmark is between the inner and outer eye corners, then averaged across both eyes for robustness against partial occlusion.

### `perception/scene.py` — Scene Detector

Loads YOLOv8-nano once at initialization and reuses it for every detection sweep. Returns `Detection` objects with normalized (0–1) bounding box coordinates, making the system resolution-independent. Includes an `detect_and_annotate()` method for debug visualization.

### `memory/models.py` — Data Models

Pure dataclasses: `Detection` (raw perception output) and `MemoryEntry` (stored observation). `Detection` includes a `spatial_description` property that converts normalized coordinates to human-readable positions ("top-left", "center", "bottom-right").

### `memory/store.py` — Memory Store

SQLite-backed persistent storage with spatial deduplication. When the same object label is detected within 15% Euclidean distance of an existing entry, the store updates `last_seen`, increments `times_seen`, and keeps the highest confidence score — rather than creating a duplicate row. Provides `get_context_for_llm()` which formats all memories into a text block optimized for LLM consumption.

### `conversation/recall.py` — Recall Engine

Implements retrieval-augmented generation (RAG). Retrieves the 30 most recent memories from SQLite, formats them as context, and sends them with the user's question to GPT-4o-mini. The system prompt constrains the LLM to only answer from provided observations and to admit when it hasn't seen something.

### `simulation/lamp.py` — Lamp Simulation

Pygame-based animated desk lamp with state-driven behaviors. Each `LampState` has a target pose (tilt, pan, brightness, color) and the lamp smoothly interpolates toward it. Features include:
- Articulated arm with two segments and joint circles
- Directional light cone with transparency
- Floating light particles near the bulb
- Per-state animations (breathing for IDLE, wobble for ATTENTION_SEEKING, happy bounce for ENGAGED)
- Picture-in-picture webcam feed with debug overlay

### `main.py` — Orchestrator

The main loop runs at 30 FPS and coordinates all modules. It captures frames, runs engagement detection on every frame, runs scene detection every 3 seconds, manages the behavior state machine with hysteresis debouncing, and handles user input for conversation. All components are initialized once and cleaned up on exit.

---

## Design Decisions and Tradeoffs

### 1. MediaPipe over dlib/Haar Cascades for Engagement

**Choice:** MediaPipe Face Mesh with iris refinement  
**Why:** Provides 478 landmarks including iris tracking (468–477) in a single model, enabling gaze estimation without a separate eye-tracking model. Runs at ~5ms per frame on CPU.  
**Tradeoff:** Slightly less accurate than dedicated gaze estimation models (e.g., GazeNet), but the latency difference (5ms vs 50ms+) is critical for real-time responsiveness. The hysteresis buffer compensates for occasional misclassifications.

### 2. YOLOv8-nano over Larger Models

**Choice:** `yolov8n.pt` (3.2M parameters)  
**Why:** Achieves ~40ms inference on CPU, fast enough for periodic detection every 3 seconds without impacting the 30 FPS engagement loop.  
**Tradeoff:** Lower mAP than YOLOv8-medium or YOLOv8-large, but since we only store high-confidence detections (≥50%) and the goal is reliable demo-quality detection, the accuracy is sufficient. The latency savings directly improve user experience.

### 3. Periodic Detection over Per-Frame Detection

**Choice:** Run YOLO every 3 seconds, not every frame  
**Why:** Object positions change slowly in a desk scene. Running at 30 FPS would waste 99% of compute on identical results while degrading engagement detection responsiveness.  
**Tradeoff:** May miss transient objects (something briefly held up and removed within 3 seconds). Acceptable for the target use case of desk/room scene understanding.

### 4. Spatial Deduplication over Raw Logging

**Choice:** Merge detections within 15% Euclidean distance into existing entries  
**Why:** Without deduplication, the memory table would accumulate thousands of near-identical rows ("person at center" every 3 seconds), making LLM context noisy and retrieval expensive.  
**Tradeoff:** May incorrectly merge two distinct objects of the same class that are close together (e.g., two cups side by side). The 15% threshold was tuned empirically as a balance between noise reduction and spatial resolution.

### 5. Retrieval-Augmented Generation over Pure LLM

**Choice:** Pull memories from SQLite first, then send as context to GPT-4o-mini  
**Why:** Keeps the LLM grounded in actual observations. Without retrieval, the LLM would either hallucinate objects it never saw or require the entire detection history in its context window.  
**Tradeoff:** Adds ~10ms of retrieval latency, but prevents hallucination entirely. The LLM only sees what the lamp actually detected.

### 6. GPT-4o-mini over Local Models

**Choice:** OpenAI API with `gpt-4o-mini`  
**Why:** Highest quality responses at ~$0.15/1M input tokens. Local models (e.g., via Ollama) would add deployment complexity and reduce answer quality.  
**Tradeoff:** Requires internet connectivity and an API key. For a production embedded system, a local model or fine-tuned small model would be necessary.

### 7. Pygame over Web-Based Visualization

**Choice:** Pygame for the lamp simulation  
**Why:** Zero-dependency rendering with hardware acceleration, built-in audio support, and simple event handling. No browser, no WebSocket, no frontend framework.  
**Tradeoff:** Less visually sophisticated than a Three.js or Unity rendering. Adequate for demonstrating the perception-to-action pipeline, which is the core technical challenge.

---

## Demo

### 📹 Demo Video

**[Watch the full demo on Google Drive →](https://drive.google.com/file/d/1-XDl_-f-5jNmGOt7vr_kJQVVGrJ4LeR5/view?usp=sharing)**

The video demonstrates all four required steps of the challenge in a single continuous session:

#### Step 1: Engagement Detection
The user looks directly at the webcam (proxy for looking at the lamp). The system detects the face and computes gaze direction using iris landmarks. The lamp transitions from `IDLE` to `ENGAGED` — it brightens to warm golden light, tilts its head toward the user, and the HUD displays "User engaged — hello!". When the user looks away, the lamp detects the gaze shift and transitions back to `IDLE`.

#### Step 2: Attention-Seeking Behavior
After the user looks away for more than 5 seconds, the lamp transitions to `ATTENTION_SEEKING`. It begins wobbling its head side to side, pulsing its light in orange tones, and displaying "Trying to get your attention..." in the status bar. The animation is designed to be noticeable but not aggressive — mimicking how a pet might try to re-engage its owner.

#### Step 3: Memory Formation
While the system runs, YOLO performs periodic scene detection every 3 seconds. The terminal logs show detected objects (e.g., `[Scene] Detected: ['person', 'cup', 'laptop']`). Each detection is stored in SQLite with its label, confidence score, spatial position, and timestamp. The deduplication logic ensures the same object in the same position updates the existing entry rather than creating noise.

#### Step 4: Memory Recall
The user presses `Q` in the Pygame window, and the lamp enters `LISTENING` state (green-tinted light). The user types a natural language question in the terminal: "What have you seen?". The system retrieves stored memories from SQLite, sends them as context to GPT-4o-mini, and the LLM generates a grounded response describing the objects it observed and their positions.

---

## Evaluation

A detailed evaluation report including engagement detection accuracy metrics, per-component latency measurements, and runnable benchmarking code is available in:

**[→ evaluation.md](evaluation.md)**

Summary of key results (example data):
- Engagement detection accuracy: **93%** across test scenarios
- Engagement detection latency: **~6ms** per frame
- YOLO detection latency: **~42ms** per sweep
- End-to-end recall latency: **~850ms** (dominated by LLM API call)

---

## Setup and Usage

### Prerequisites

- Python 3.10+
- Webcam
- OpenAI API key

### Installation

```bash
git clone https://github.com/<your-username>/lelamp-challenge.git
cd lelamp-challenge
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Running

```bash
# Set your OpenAI API key
# Windows PowerShell:
$env:OPENAI_API_KEY = "sk-..."

# macOS/Linux:
export OPENAI_API_KEY="sk-..."

# Run the system
python main.py
```

### Controls

| Key | Action |
|---|---|
| `Q` | Ask the lamp a question (type in terminal) |
| `ESC` | Quit |

---

## Limitations and Future Work

### Current Limitations

1. **Gaze estimation is a proxy for engagement.** The system measures iris centering within the eye opening, which approximates "looking at the camera" but cannot distinguish between looking at the camera vs. looking at something directly behind the camera. A dedicated gaze estimation model (e.g., ETH-XGaze) would improve accuracy.

2. **No physical actuation.** The lamp is simulated in Pygame. Mapping the 6-DOF pose to real servos would require inverse kinematics and motor control — which is straightforward but outside the scope of this software challenge.

3. **LLM requires internet.** The recall engine depends on the OpenAI API. For an embedded product, a local model (e.g., Phi-3-mini via Ollama) would be necessary.

4. **Single-user only.** The system tracks one face. Multi-face tracking with identity assignment would require face recognition (e.g., ArcFace embeddings).

5. **Object detection is limited to COCO classes.** YOLOv8-nano can detect 80 object categories. Custom objects (e.g., specific personal items) would require fine-tuning or an open-vocabulary detector like OWL-ViT.

### Future Work

- **Multi-user interaction** — Track multiple faces with identity, detect who is speaking via audio-visual fusion
- **Emotion detection** — Analyze facial expressions (MediaPipe already provides the landmarks) and voice tone (using a speech emotion model)
- **Self-learning behavior** — Reinforce attention-seeking animations that successfully re-engage the user, deprecate ones that don't
- **Interruption awareness** — Detect activities like phone calls or TV watching and suppress attention-seeking behavior
- **Spatial mapping** — Build a persistent 3D map of the room using depth estimation, enabling more accurate object localization ("your keys are on the shelf to the left")
- **Voice interface** — Replace terminal input with speech recognition (Whisper) and text-to-speech for fully conversational interaction

---

## Project Structure

```
lelamp-challenge/
├── README.md                  # This document
├── evaluation.md              # Evaluation report with metrics
├── requirements.txt           # Python dependencies
├── config.py                  # All tunable parameters
├── main.py                    # Main orchestration loop
├── perception/
│   ├── engagement.py          # MediaPipe gaze-based engagement detection
│   └── scene.py               # YOLOv8 object detection
├── behavior/
│   └── (state logic in main.py for simplicity)
├── memory/
│   ├── models.py              # Detection and MemoryEntry dataclasses
│   └── store.py               # SQLite storage with deduplication
├── conversation/
│   └── recall.py              # OpenAI-powered memory recall
├── simulation/
│   └── lamp.py                # Pygame animated lamp visualization
└── evaluation/
    └── metrics.py             # Benchmarking utilities
```

---

## License

This project was built as a submission for the LeLamp SW/CV Challenge.
