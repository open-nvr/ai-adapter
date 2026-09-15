# YOLO-Pose Human-Keypoint Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md) wrapping a **YOLO11n-pose** ONNX model. POST a frame (or open a WebSocket stream) and get one COCO-17 skeleton per person back — the body geometry an app needs to decide what someone is *doing*, not just where they are.

It is built for the apps that ask questions about limbs: *did the guard run the wand along this visitor's arms before letting them into the showroom?* (wrist and elbow tracks), *did someone fall?* (torso angle), *which way is this queue facing?* (shoulder orientation). The adapter supplies geometry; the app supplies the rule.

## What it does

Takes a JPEG/PNG frame, returns a person list with pixel-space boxes and 17 keypoints each.

```bash
curl -X POST http://localhost:9009/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@entrance.jpg" \
  -F 'params={"conf": 0.4, "imgsz": 448};type=application/json'
```

Real output, abbreviated — one of the three people the model found in an 810×1080 frame:

```jsonc
{
  "status": "ok",
  "model_name": "yolo11n-pose",
  "model_version": "onnxruntime/yolo11n-pose",
  "inference_ms": 47,
  "result": {
    "persons": [
      {
        "bbox": [51.2, 396.0, 243.5, 909.7],            // [x1, y1, x2, y2], pixels
        "score": 0.8857,
        "keypoints": [                                   // 17 × [x, y, conf]
          [143.5, 444.1, 0.9858],                        // 0  nose
          // … 8 more …
          [151.9, 550.3, 0.8875],                        // 9  left_wrist
          // … 6 more …
          [76.2, 860.3, 0.9873]                          // 16 right_ankle
        ]
      }
      // … two more persons …
    ],
    "keypoint_names": ["nose", "left_eye", "…", "right_ankle"],
    "frame_dimensions": {"w": 810, "h": 1080}
  }
}
```

### Keypoint order (COCO-17)

Keypoints are always a list of exactly 17 `[x, y, confidence]` triples in this order — index 9 is the left wrist in every response, in every frame, forever. The same list is echoed back as `keypoint_names` so a consumer never has to hard-code it.

| # | Joint | # | Joint | # | Joint |
|---|---|---|---|---|---|
| 0 | `nose` | 6 | `right_shoulder` | 12 | `right_hip` |
| 1 | `left_eye` | 7 | `left_elbow` | 13 | `left_knee` |
| 2 | `right_eye` | 8 | `right_elbow` | 14 | `right_knee` |
| 3 | `left_ear` | 9 | `left_wrist` | 15 | `left_ankle` |
| 4 | `right_ear` | 10 | `right_wrist` | 16 | `right_ankle` |
| 5 | `left_shoulder` | 11 | `left_hip` | | |

Left/right are the **subject's** left and right, as COCO defines them — not the viewer's. A joint the model can't see is still returned, with a low confidence; nothing is ever omitted, so the slot index stays meaningful. `adapters/yolo_pose/coco_keypoints.py` also ships `COCO_SKELETON`, the bone list for anyone drawing an overlay.

### Coordinates are pixels, not `[0, 1]`

Unlike the §5.1 `DetectionResult` shape that [`adapters/yolov8/`](../yolov8/README.md) returns, this adapter emits **pixel coordinates in the source frame** and ships `frame_dimensions` alongside. A detection is consumed as a region, so normalizing is free; a pose is consumed as geometry, and normalizing a 16:9 frame to a unit square distorts every limb angle the app is trying to measure. Dividing by `frame_dimensions` to get normalized values is lossless; un-distorting an angle is not.

### Parameters

All optional, all per call, all validated (a bad value is a typed `400 malformed_input`, never a 500).

| Param | Default | Range | What it does |
|---|---|---|---|
| `conf` | `0.4` | 0.0 – 1.0 | Person-confidence floor. Higher than YOLO's usual 0.25 on purpose: a half-confident person produces a plausible-looking but wrong skeleton, which is worse than none. Also accepted as `confidence_threshold`, the spelling the yolov8 adapter uses. |
| `iou` | `0.45` | 0.0 – 1.0 | NMS overlap threshold over person boxes. Also accepted as `iou_threshold` / `nms_threshold`. |
| `imgsz` | `448` | 160 – 1280, multiple of 32 | Model input side, and the CPU/accuracy dial — 320 for more cameras per box, 640 for small or distant subjects (on the sample frame below, 640 is what finds the fourth, half-occluded person). Only tunable if the ONNX was exported with `dynamic=True`; a fixed-size export dictates its own size and says so if you ask for another. |
| `max_persons` | `20` | 1 – 100 | Cap on persons per frame, best-scoring first. A payload and latency guard, not a scene assumption. |

Frames inferred over the **WebSocket** path use these defaults: §6's `frame` message carries metadata only and `handshake` forbids extra fields, so there is nowhere for a per-frame `conf` to ride. Callers that need custom thresholds use HTTP `/infer`. (Same limitation as the yolov8 adapter.)

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | required | auth-exempt |
| `GET /capabilities` | required | sha256 fingerprint, `tasks_advertised=["pose_estimation"]`, `gpu` build-accurate (false on the CPU image), `fair_queuing=per_camera` |
| `GET /hardware/evaluation` | required | load state, onnxruntime providers, core count, effective `imgsz` |
| `GET /metrics` | required | Prometheus exposition incl. `adapter_stream_connections_active` and the domain metrics below |
| `POST /infer` | required | multipart (`frame` file) or JSON (`frame_b64`) |
| `POST /infer/stream` (WS) | required | §6 protocol, inline frames — handshake → frame_meta + bytes → result loop. `stats` answers with this session's real inflight / queue depth / fps |

**Two §6 options are not implemented**, and both are answered by a downgrade in the `handshake_ack` rather than a refusal — read the ack, don't assume your offer was taken. §6.2's shared-memory fast path: the adapter advertises `supports_shared_memory: false`, and `frame_transport: "shared_memory"` comes back as `"websocket"`. §6.3's NATS `result_sink`: results always return over the same socket.

### Domain metrics

| Metric | Why it exists |
|---|---|
| `adapter_pose_persons_total` | Volume of skeletons produced. Flat-lines the moment an upstream camera stops delivering usable frames. |
| `adapter_pose_keypoints_visible_total{keypoint="left_wrist"}` | Per-joint visibility (confidence ≥ 0.5). This is the diagnostic that matters: a camera re-aimed slightly high stops seeing **wrists** long before anyone notices the wand-compliance app has gone quiet, and "left_wrist went to zero on cam-3" is visible in a single scrape. Label set is the model's own 17 joints, so cardinality is bounded by the weights. |

## Permissions

Declared **build-accurately** per §8. `gpu` follows the installed onnxruntime build: the stock image pins the CPU-only `onnxruntime` wheel, declares `gpu=false`, and registers with KAI-C without a GPU-grant prompt — which is the normal deployment here, since the adapter is CPU-first by design. A rebuild against `onnxruntime-gpu` declares `gpu=true` and starts `pending` until an operator grants the scope. `network_egress` is **derived from the configuration the same way**: empty when `YOLO_POSE_MODEL_URL` is unset (the default, and the `sovereignty=local_only` posture — weights are mounted and the container never dials out), and exactly the one host of that URL when an operator configures a first-boot fetch, because that is a call the adapter can genuinely make and §8 treats an undeclared egress host as grounds for removal. Nothing on the steady-state path touches the network either way. No `host_filesystem` scope either — weights belong in a container-owned named volume mounted at `/weights`, not a host bind-mount. The declaration lives in [`main.py`](main.py); authoring rules are in the repo [README](../../README.md#declaring-permissions).

## Getting the weights

The image ships **without** weights (~12 MB of ONNX in a 250 MB image would still have to be versioned somewhere). Ultralytics publishes the `.pt` checkpoint but no pre-built ONNX, so the ONNX is exported locally — one command, once:

```bash
# From the ai-adapter repo root. Needs ultralytics, which the lean `pose`
# extra deliberately does NOT install (it pulls torch, and exporting is a
# one-time authoring task):
pip install --quiet "ultralytics==8.3.240" "onnx>=1.16,<2"
python download_models.py --all      # exports model_weights/yolo11n-pose.onnx
```

or by hand, which is the same thing `download_models.py` does:

```bash
mkdir -p model_weights && cd model_weights
pip install --quiet "ultralytics==8.3.240" "onnx>=1.16,<2"
yolo export model=yolo11n-pose.pt format=onnx opset=12 imgsz=448 dynamic=True
cd ..
```

Either way the export is a **development-machine** step: the runtime image
never installs ultralytics or torch, it is handed a finished `.onnx`.

Then mount that directory at `/weights`. Operators who prefer a first-boot download host the exported file themselves and set `YOLO_POSE_MODEL_URL`; the SDK's `ensure_model_file` streams it into the weights volume once and every later boot finds it already there. A file that is already present **always** wins and no network call is made, which is what makes the `sovereignty=local_only` posture (empty URL, pre-populated volume) work.

## Run locally

```bash
# Install deps (from the ai-adapter repo root)
uv venv && source .venv/bin/activate
uv sync --extra pose

# Weights (see above — the export needs ultralytics, which `pose` omits)
pip install --quiet "ultralytics==8.3.240" "onnx>=1.16,<2"
python download_models.py --all

# Start the service
OPENNVR_ADAPTER_TOKEN=dev-token \
  uv run uvicorn adapters.yolo_pose.main:app --host 0.0.0.0 --port 9009
```

## Run with Docker

```bash
docker build -f adapters/yolo_pose/Dockerfile -t opennvr/yolo-pose-adapter:1.0.0 .

docker run --rm -p 9009:9009 \
  -e OPENNVR_ADAPTER_TOKEN=$(openssl rand -hex 16) \
  -v $(pwd)/model_weights:/weights:ro \
  opennvr/yolo-pose-adapter:1.0.0
```

Pre-built images are published to `ghcr.io/open-nvr/yolo-pose-adapter` on every push and tagged release — see [`.github/workflows/publish-images.yml`](../../.github/workflows/publish-images.yml).

## Operational notes

- **Model:** `yolo11n-pose.onnx`, 11.6 MB on disk, mounted at `/weights`. Not baked into the image; the container boots with no network. Export it with `dynamic=True` (the recipe above and `download_models.py` both do) or the `imgsz` parameter is fixed at whatever size it was exported with.
- **Throughput:** measured on an 8-core x86 laptop CPU (onnxruntime 1.23.2, CPU provider, 810×1080 frame with four people), median wall time per `/infer` call:

  | `imgsz` | ms/frame | fps | persons found |
  |---|---|---|---|
  | 320 | 31 | 32 | 3 |
  | **448** (default) | **47** | **21** | 3 |
  | 640 | 94 | 11 | 4 |

  End-to-end over the WebSocket path (uvicorn, JPEG decode, JSON result serialization, one frame at a time) the same box sustains **12 fps** on that frame at the default `imgsz` — the protocol overhead is real, so size a deployment off this number rather than off the raw inference time.

  A 4-core box of the same generation runs roughly 1.5–2× slower, which puts the default `imgsz=448` at ~6–8 fps end-to-end — **below** the 10 fps target. On four cores, run at `imgsz=320` (31 ms here, so ~10 fps end-to-end there) or give the adapter more cores; either way plan **one camera per adapter instance** and scale out with more instances rather than more streams. Measure before fanning out: these are one machine's numbers, not a promise. `/hardware/evaluation` reports `warn` below four cores for the same reason.
- **Memory:** ~160 MB RSS measured after load, warm-up and 20 inferences at `imgsz=448` (onnxruntime session + OpenCV + numpy), plus the usual FastAPI/uvicorn overhead — budget ~250 MB for the container. There is no per-camera state: the adapter is stateless between frames, so memory does not grow with the number of streams.
- **Cold start:** the model loads and runs one throwaway warm-up inference during lifespan startup, so the first real frame doesn't pay onnxruntime's arena allocation. Measured at 0.4 s for load + warm-up. `/health` reports `loading` until it finishes and `error` if the weights are missing.
- **Concurrency:** `max_inflight=1` — one shared ONNX session, no cross-stream serialization — and `stream_max_concurrent=4`, which is the same honesty applied to the advertisement: ~12 fps end-to-end feeds about one 10 fps camera, and the extra headroom is for connections that are reconnecting, paused or idle rather than for four simultaneous inferring cameras. Nothing enforces the cap per frame yet (no §7.1 `overloaded`, no §6.5 `4004`), which is exactly why it is not higher. `fair_queuing=per_camera` matters more here than for an event-driven adapter: every camera streaming pose is a *steady* load, and without it the busiest entrance starves the rest.
- **Not a tracker.** Persons are per-frame and carry no identity across frames. Pair with [`adapters/bytetrack/`](../bytetrack/README.md) if the app needs stable IDs (wrist *travel* over time, for instance).
- **Sovereignty:** no egress on the steady-state path; the only network call the adapter can ever make is the optional first-boot weights fetch the operator configures.

## Verify conformance

```bash
python -m conformance http://localhost:9009 --token $OPENNVR_ADAPTER_TOKEN
```

Green = KAI-C will accept the adapter.

## Try it — WebSocket (the 10 fps path)

```python
import asyncio, json, websockets

async def main():
    headers = {"Authorization": "Bearer dev-token"}
    # additional_headers on websockets >= 14; extra_headers before that.
    async with websockets.connect("ws://localhost:9009/infer/stream", additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "handshake", "client_id": "wand-app", "camera_id": "entrance-1",
            "frame_transport": "websocket", "expected_input_rate_hz": 10,
        }))
        ack = json.loads(await ws.recv())
        print("session", ack["session_id"])

        with open("entrance.jpg", "rb") as fh:
            frame = fh.read()
        await ws.send(json.dumps({
            "type": "frame", "seq": 1, "ts_ms": 0, "content_type": "image/jpeg",
        }))
        await ws.send(frame)
        result = json.loads(await ws.recv())
        for person in result["result"]["persons"]:
            x, y, conf = person["keypoints"][9]      # left wrist
            print(f"left wrist at ({x}, {y}) conf={conf}")

        await ws.send(json.dumps({"type": "close", "reason": "done"}))

asyncio.run(main())
```

## Layout

```
adapters/yolo_pose/
├── main.py             FastAPI app — AdapterApp construction, §8 permissions
├── service.py          YoloPoseService — ONNX session, letterbox, NMS, §6 WS loop
├── coco_keypoints.py   COCO-17 joint names, stride, skeleton bones
├── Dockerfile          Self-contained image; CPU by default, GPU-rebuildable
└── README.md           you are here
```

## Tests

```bash
pytest tests/test_yolo_pose_service.py
```

67 tests over the load lifecycle (including missing weights and a wrong-model export), the documented output shape, COCO-17 ordering, letterbox un-mapping on a 16:9 frame, NMS, every caller parameter and its rejection path, the §6 WebSocket protocol including a stats reply carrying real values, fingerprint stability and drift, containment of an unforeseen post-processing failure on both transports, the derived egress declaration, auth and correlation_id. The model is stubbed (`tests/_yolo_pose_service_fixtures.py`) — no weights, no network, no GPU — so what is under test is everything between the request bytes and the response JSON.

## Why this model

**YOLO11n-pose, on CPU, at 448 px.** Three constraints picked it:

- **It has to run on the box that's already there.** A jewellery showroom has a small NUC or a mini-tower, not a GPU. The nano pose model is ~12 MB of ONNX running on `onnxruntime` alone — no torch, no CUDA, no 2 GB image. Top-down alternatives (ViTPose, HRNet) are far more accurate on benchmark keypoints and need a GPU to reach double-digit fps; MediaPipe Pose is fast on CPU but is single-person by design, and the frame we care about always has at least two people in it (guard and visitor).
- **The accuracy that matters here is coarse.** The app asks "did the wand pass within a hand's width of this wrist, along this arm?" That is a question about *where the wrist and elbow are within a few centimetres*, not about finger articulation. Nano-scale pose error at entrance-camera framing is comfortably inside that budget — spending 10× the compute to sharpen keypoints that are then quantised into "near the arm / not near the arm" buys nothing.
- **One detector, one pass.** YOLO-pose is bottom-up: person boxes and keypoints come out of the same forward pass, so cost is flat in the number of people. A top-down pipeline (detector, then a pose model per crop) costs a second model invocation per person, which is exactly the wrong scaling for a doorway where a family of five walks in together.

The trade-offs an operator inherits: keypoints are noisier than a top-down model's, so per-frame limb angles jitter and any rule should smooth over a few frames; heavily occluded joints come back with low confidence rather than being omitted, so consumers must apply their own floor; and there is no identity across frames — pair with ByteTrack if you need to follow one wrist over time.

The model swap is a one-line change if a future app needs the accuracy: any Ultralytics pose export with 17 COCO keypoints drops in (`yolo11s-pose`, `yolo11m-pose`) by replacing the ONNX file and `MODEL_NAME`. The adapter validates the model's feature count on first inference and fails typed if the export isn't a 17-keypoint pose model, so a wrong file is caught immediately rather than producing quietly truncated skeletons.
