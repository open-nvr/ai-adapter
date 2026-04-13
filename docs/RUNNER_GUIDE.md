# 🏃 Runner Guide: The CLI Test Suite

Before you hook Open-NVR to dozens of live IP cameras, you need a way to rapidly test your AI tasks. That's exactly what the Runner CLI does.

The runner allows you to capture frames directly from your laptop webcam, an RTSP test stream, or an MP4 file, and stream them securely into the `AIAdapters` inferencing server.

**Core Files:**
- `opennvr/runner.py` : For Webcams and Live IP Cameras.
- `opennvr/runnerrec.py` : for MP4 Video Files.

---

## 🚀 Quick Start Boot

1. **Boot the Engine First** (in Terminal A)
   ```bash
   # Launch the AI Server locally
   uv run uvicorn app.main:app --reload --port 9100
   ```

2. **Launch the CLI Runner** (in Terminal B)
   ```bash
   # See what AI tasks are loaded
   uv run python opennvr/runner.py --list-tasks

   # Run Person Detection on your laptop defaults webcam
   uv run python opennvr/runner.py --task person_detection
   ```

---

## 🎛️ The Live Debug GUI

If you want to actually *see* the AI working in real time, append the `--debug` flag!

```bash
uv run python opennvr/runner.py --task person_counting --debug
```

This will pop open a smooth, 30FPS OpenCV window mirroring your camera. Every ~2 seconds, it will seamlessly flash the bounding boxes, confidence scores, and tracking IDs directly onto the live feed.

*(Tip: Press `q` to quit the GUI).*

---

## 📡 Advanced Command Examples

Test like a pro. The CLI supports multiple configurations out of the box:

**Custom Camera Intervals (Power Saving)**
```bash
# Capture and infer exactly once every second
uv run python opennvr/runner.py --task person_detection --interval 1.0
```

**Testing Live IP Cameras (RTSP)**
```bash
# Point the runner at a real network camera
uv run python opennvr/runner.py --task person_detection --rtsp "rtsp://admin:pass@192.168.1.100:554/stream"
```

**Stacking Multiple Tasks Simultaneously!**
```bash
# Run detection and counting at the exact same time!
uv run python opennvr/runner.py --task person_detection,person_counting --interval 2.0
```

**Testing Historical MP4 Footage**
```bash
# Great for reproducing edge cases and bugs
uv run python opennvr/runnerrec.py --task person_detection --video my_test_footage.mp4 --interval 0.5
```

---

## 🩺 Performance & Troubleshooting

Running intense AI models on edge CPUs naturally incurs latency. Here is what to expect, and how to fix common runner issues.

| Task Profile | CPU Latency | Recommended CLI `--interval` |
|--------------|-------------|----------------------------|
| `person_detection` | Fast (~900ms) | `0.5` to `1.0` seconds |
| `person_counting` | Medium (~1800ms)| `1.5` to `2.0` seconds |
| `scene_description`| Heavy (~2000ms+)| `2.0` to `3.0` seconds |

### Common CLI Errors

* **"Connection Refused"**: You forgot to boot the backend! Run `uv run uvicorn app.main:app --reload --port 9100` first.
* **"Timeout on First Request"**: Heavy models (like InsightFace) are *lazy-loaded* to keep boot times fast. Your first inference might take 15 seconds to download/load into RAM. Just wait!
* **Camera not found**: If your laptop has multiple cameras, try `--camera 1`.
* **RTSP Fails**: Ensure your password URL is properly encoded (e.g., if your password is `my@password`, encode it as `my%40password`).
