# LeLamp — Real-Time AI Perception, Memory, and Recall System

## Overview

LeLamp is a real-time interactive AI system that combines computer vision, behavioral state modeling, memory, and language models into one pipeline.

The system:
- Tracks user engagement using gaze detection
- Detects objects in the environment using YOLO
- Stores structured observations in a persistent database
- Answers natural language questions based on what it has seen

This project demonstrates a full **perception → memory → reasoning** system in real time.

## Demo

🎥 **Live Demo Video**  
[Watch the demo](demo/Lamp-Live-Demo.mp4)

The demo shows:
- Real-time gaze-based engagement detection
- Attention-seeking lamp behavior
- YOLOv8 object detection
- SQLite-based memory with deduplication
- OpenAI-powered recall from stored observations

## System Architecture

### 1. Engagement Pipeline

The system uses MediaPipe Face Mesh to track facial landmarks and estimate gaze direction using iris position.

It classifies the user as:
- `ENGAGED`
- `DISENGAGED`
- `NO_FACE`

This drives the lamp's behavioral state machine.

### 2. Scene and Memory Pipeline

The system uses YOLOv8-nano for object detection every few seconds.

Each object detection is stored with:
- Object label
- Confidence score
- Spatial position
- First seen timestamp
- Last seen timestamp
- Number of times seen

To avoid duplicate memory entries, the system deduplicates objects based on label and spatial proximity.

### 3. Recall Pipeline

The recall system retrieves stored memories from SQLite and sends them as context to an OpenAI model.

The LLM only answers using stored observations, following a retrieval-then-generate pattern.

## Key Features

- Real-time webcam-based perception
- Gaze-based engagement detection
- State-driven animated lamp simulation
- Object detection using YOLOv8
- Persistent spatial memory using SQLite
- Natural language recall using OpenAI API
- Modular Python architecture

## Tech Stack

- Python
- OpenCV
- MediaPipe
- YOLOv8 / Ultralytics
- SQLite
- OpenAI API
- Pygame

## How to Run

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API key

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Command Prompt:

```cmd
set OPENAI_API_KEY=sk-...
```

### 5. Run the app

```bash
python main.py
```

## Controls

- `ESC` → Quit application
- `Q` → Ask a question about observed objects

## Example Queries

- "What objects have you seen?"
- "Where is the phone?"
- "What is on the left side?"
- "What have you seen the most?"

## Design Decisions

- **YOLOv8n (nano):** chosen for speed and laptop-friendly performance.
- **Periodic detection:** object detection runs every few seconds instead of every frame to reduce CPU load.
- **Spatial deduplication:** repeated detections of the same object in a similar location update an existing memory instead of creating duplicates.
- **SQLite memory:** lightweight persistent storage without requiring an external database server.
- **Retrieval-first LLM recall:** the model receives stored memory context before answering, reducing hallucination.

## Project Structure

```text
lelamp-challenge/
├── main.py
├── config.py
├── requirements.txt
├── perception/
│   ├── engagement.py
│   └── scene.py
├── simulation/
│   └── lamp.py
├── memory/
│   ├── models.py
│   └── store.py
├── conversation/
│   └── recall.py
└── demo/
    └── Lamp-Live-Demo.mp4
```

## Future Improvements

- Web-based live interface
- Real-time bounding box visualization in the webcam preview
- Improved object tracking across frames
- Voice input/output
- Multi-user interaction handling

## Author

Faiz Saifuddin  
University of Waterloo — Computer Science (Co-op)