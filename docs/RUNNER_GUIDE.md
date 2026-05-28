# Runner Guide — the CLI test harness

Before pointing OpenNVR at live IP cameras, you'll want a way to test AI tasks quickly on something local. The Runner CLI does exactly that: capture frames from your laptop webcam, an RTSP test stream, or an MP4 file, and stream them into the AI Adapter server while it's still running on your own machine.

Two scripts: `opennvr/runner.py` for webcams and live IP cameras, and `opennvr/runnerrec.py` for MP4 files.

## Quick start

In one terminal, boot the AI server:

```bash
uv run uvicorn app.main:app --reload --port 9100
```

In another, list the tasks the server discovered, then run one against your laptop's default webcam:

```bash
uv run python opennvr/runner.py --list-tasks
uv run python opennvr/runner.py --task person_detection
```

## Live debug GUI

Append `--debug` to open an OpenCV window mirroring your camera at around 30 fps with bounding boxes, confidence scores, and tracking IDs overlaid every couple of seconds. Press `q` to quit.

```bash
uv run python opennvr/runner.py --task person_counting --debug
```

## Advanced examples

A custom inference interval keeps CPU use down on long sessions:

```bash
uv run python opennvr/runner.py --task person_detection --interval 1.0
```

Point the runner at a real network camera with `--rtsp`:

```bash
uv run python opennvr/runner.py --task person_detection \
  --rtsp "rtsp://admin:pass@192.168.1.100:554/stream"
```

Comma-separate tasks to run several at once on the same frame:

```bash
uv run python opennvr/runner.py --task person_detection,person_counting --interval 2.0
```

For reproducing an edge case from a recording, use the MP4 runner:

```bash
uv run python opennvr/runnerrec.py --task person_detection --video my_test_footage.mp4 --interval 0.5
```

## Performance and troubleshooting

Latency on CPU is task-dependent. Rough guidance for choosing `--interval`:

| Task | CPU latency | Suggested `--interval` |
|---|---|---|
| `person_detection` | ~900 ms | 0.5 – 1.0 s |
| `person_counting` | ~1800 ms | 1.5 – 2.0 s |
| `scene_description` | ~2000 ms+ | 2.0 – 3.0 s |

Common errors. *Connection refused* means the backend isn't running — boot `uv run uvicorn app.main:app --reload --port 9100` first. *Timeout on first request* is expected for heavy models like InsightFace: they're lazy-loaded to keep boot fast, so the first inference can take fifteen seconds while weights download into RAM. *Camera not found* on laptops with multiple cameras is usually fixed with `--camera 1`. *RTSP fails* is most often a credential-encoding issue — URL-encode special characters in your password (`my@password` becomes `my%40password`).
