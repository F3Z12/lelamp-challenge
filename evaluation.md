# LeLamp — Evaluation

> Note: The following results are rough estimates based on manual testing during development. They are not precise benchmarks, but give a realistic sense of system performance.

---

## Engagement Detection

I tested the system by:
- Looking directly at the camera
- Looking away in different directions
- Slightly turning my head / partial visibility

### Observations

- The system correctly detects when I am looking at the camera most of the time
- It usually switches to disengaged when I look away
- Sometimes struggles when my face is partially visible or lighting is poor

### Overall

- Works reliably enough for interaction
- Small errors don’t affect the experience much

---

## Latency (Estimated)

These are rough estimates based on how the system behaves during use:

| Component | Approx Time | Notes |
|----------|------------|------|
| Engagement Detection | ~5–10 ms | Very fast, runs every frame |
| YOLO Object Detection | ~40–60 ms | Slower, runs every few seconds |
| Memory Storage (SQLite) | <1 ms | Very fast |
| Memory Retrieval | <1 ms | Instant |
| LLM Response | ~0.5–1.5 s | Depends on API/network |
| Total Recall Time | ~1 second | Mostly due to LLM |

---

## Key Takeaways

- The system feels real-time during normal use
- Object detection is the heaviest computation but is limited to avoid lag
- LLM response is the slowest part, but it only happens when the user asks a question
- Overall, the system is fast enough for an interactive experience

---

## Limitations

- Results are approximate, not formally measured
- Performance may vary depending on hardware and lighting
- No GPU acceleration was used