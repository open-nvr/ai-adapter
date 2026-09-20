# Package-Detection Adapter (Contract v1)

Reference implementation of the [AI Adapter Contract v1](https://github.com/open-nvr/open-nvr/blob/main/docs/AI_ADAPTER_CONTRACT.md) for the **`package_detection`** task: parcels — boxes, padded mailers, envelopes — on a doorstep. A **YOLOv8n fine-tuned on two openly licensed doorstep datasets**, served on `onnxruntime`, CPU by design.

It exists because COCO, which the general object detector runs, has no package class. Tier-0 can tell an app that a person came to the door and left; it cannot tell it whether a parcel is on the step. This adapter answers that one question, on demand.

## Who calls it

OpenNVR's [`package-delivery`](https://github.com/open-nvr/open-nvr/tree/main/examples/package-delivery) app asks KAI-C what can count parcels and takes the best skill on the box, in this order: an adapter advertising `package_detection` (this one), an object detector with a box class, a VQA model, and only then the COCO bag classes as a stand-in. The page shows which is in use as *good / fair / proxy*. Register this adapter and the app moves to *good* within five minutes, with no restart and no change to the app — an app asks for a task, never for an adapter by name.

The adapter is called when somebody leaves the doorstep and on a slow recheck cadence: a handful of frames a day per door, over HTTP `/infer`. There is deliberately **no `/infer/stream`** — a box that wants parcels tracked at frame rate should run the same weights as an `object_detection` adapter, which is a different contract posture and CPU budget.

## What it does

Takes a JPEG/PNG frame, returns §5.1 detections with the label `package`.

```bash
curl -X POST http://localhost:9010/infer \
  -H "Authorization: Bearer $OPENNVR_ADAPTER_TOKEN" \
  -F "frame=@porch.jpg" \
  -F 'params={"conf": 0.35};type=application/json'
```

```jsonc
{
  "status": "ok",
  "model_name": "yolov8n-package",
  "model_version": "onnxruntime/yolov8n-package",
  "inference_ms": 27,
  "result": {
    "detections": [
      {
        "label": "package",
        "confidence": 0.91,
        "bbox": {"x": 0.38, "y": 0.61, "w": 0.24, "h": 0.17},   // normalized [0,1], top-left + size
        "track_id": null,
        "attributes": {"class_id": 0}
      }
    ],
    "frame_dimensions": {"w": 1920, "h": 1080},
    "labels": ["package"],   // index → label, echoed so a consumer never hard-codes it
    "count": 1               // the question this adapter is usually asked
  }
}
```

Boxes are normalized to the **source** frame (the letterbox padding is unmapped), so "is the centre of this box inside my porch zone" is one comparison.

### Parameters (per call, all optional)

| Param | Default | Range | Notes |
|---|---|---|---|
| `conf` (alias `confidence_threshold`) | 0.35 | 0–1 | Score floor. Higher than YOLO's usual 0.25: a phantom parcel becomes a phantom delivery, and this model's precision is its strong suit. |
| `iou` (alias `iou_threshold`) | 0.5 | 0–1 | Class-agnostic NMS threshold. |
| `imgsz` | 416 | 160–1280, multiple of 32 | Model input side. The shipped weights were trained at 416 and exported with dynamic axes. |
| `max_detections` | 50 | 1–300 | Payload guard, not a scene assumption. |

Bad values are typed `malformed_input` 400s with a §7 envelope, never a 500.

## The model

`yolov8n-package.onnx` — YOLOv8n (COCO-pretrained backbone) fine-tuned for one class on:

| Dataset | Images | Licence |
|---|---|---|
| [*package at front door* v2](https://universe.roboflow.com/package-detection/package-at-front-door) | 1,293 (1,223 / 47 / 23 train / val / test) | MIT |
| [*Packages* v6, raw](https://universe.roboflow.com/joseph-nelson/packages-g0ton) | 26 (19 / 4 / 3) | CC0 |

Real doorsteps: boxes on mats, mailers against stone steps, couriers holding parcels, night frames. Training augmentation (cutout, noise, blur) is baked into the front-door set.

Trained 40 epochs at 416 px on 1,242 images. Held out, never trained on:

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 51 | 57 | 0.959 | 0.831 | 0.933 | 0.763 |
| **test** | 26 | 47 | **0.974** | **0.804** | **0.914** | **0.631** |

11.7 MB ONNX; 23–34 ms per frame on two x86 cores at 416 px, no GPU.

**Read the recall, not just the mAP.** Precision is high — when it says *package*, it is one, which is what keeps a phantom delivery off somebody's phone. Recall at 0.80 is the honest number: roughly one parcel in five is missed on a first look. That is survivable for this consumer and not for a per-frame one, because the app re-counts on a cadence and a missed parcel is picked up on the next pass, where a false one would have fired an alert immediately. The default `conf` of 0.35 is set on that trade.

**Known failure modes**, visible in the test set: two parcels side by side are sometimes returned as one box (so `count` reads low where it matters most — a double delivery); a parcel far down a hallway or at a steep angle can come back with a box much larger than the parcel, which matters if you are zone-filtering on the box centre; padded mailers and white boxes on light stone are the weakest cases.

**What it is not.** A model that has seen *your* porch. A parcel in the rain on a dark mat, a camera looking straight down, a shared lobby with a parcel shelf — these are where any general detector wobbles. The adapter is built so that fixing this is a *skill you add*, not a change to the app: fine-tune on a few hundred frames from your own cameras (the Deliveries page's *Not a package* / *Collected* corrections are exactly those frames), mount the new ONNX, and it is picked up on the next `/capabilities` poll.

### Reproduce or re-train

```bash
pip install "ultralytics==8.3.240" onnx onnxslim
export ROBOFLOW_API_KEY=...            # free account → Settings → API keys
python adapters/package_detection/train_package_model.py download
python adapters/package_detection/train_package_model.py train --epochs 40
python adapters/package_detection/train_package_model.py evaluate model_weights/yolov8n-package.onnx
```

Drop your own frames and YOLO-format labels under `package_data/site/{images,labels}` before `train` and they are merged in. A fine-tune with more than one class works unchanged: set `PACKAGE_DETECTION_LABELS=package,envelope` (index order) and each class comes back with its label.

## Run it

```bash
# weights mounted (the default, never touches the network)
docker run --rm -p 9010:9010 \
  -e OPENNVR_ADAPTER_TOKEN=<token> \
  -v $(pwd)/model_weights:/weights:ro \
  ghcr.io/open-nvr/package-detection-adapter:latest

# or fetch the published weights once on first boot
docker run --rm -p 9010:9010 \
  -e OPENNVR_ADAPTER_TOKEN=<token> \
  -e PACKAGE_DETECTION_MODEL_URL=https://github.com/open-nvr/ai-adapter/releases/download/package-detection-v1.0.0/yolov8n-package.onnx \
  -v opennvr_package_detection_weights:/weights \
  ghcr.io/open-nvr/package-detection-adapter:latest
```

Then register with KAI-C (`POST /api/v1/adapters/register`, or *AI Adapters → Add* in the OpenNVR UI). On an OpenNVR install the App Catalog's *AI Adapters* page lists it and installs it.

From a source checkout:

```bash
uv sync --extra package
python download_models.py --all            # fetches the release asset into model_weights/
OPENNVR_ADAPTER_TOKEN=secret uv run uvicorn adapters.package_detection.main:app --port 9010
python -m conformance http://localhost:9010 --token secret
```

## Contract details

- **Tasks advertised:** `package_detection` only. A one-class model must not satisfy an app whose `object_detection` means people and vehicles.
- **Permissions:** `gpu` declared from the installed onnxruntime build (the stock image is CPU-only → `false`); `network_egress` derived from `PACKAGE_DETECTION_MODEL_URL` (empty by default → none declared); no `host_filesystem` (the weights path is a container-owned volume).
- **Scheduling:** `max_inflight=1`, per-camera fair queuing.
- **Fingerprint:** sha256 of the ONNX, recomputed per poll, so a rotated weights file shows up as §11.3 drift.
- **Metrics:** the SDK baseline plus `adapter_package_frames_total{result="packages"|"empty"}` and `adapter_packages_total{label}`. A door whose every frame comes back `empty` after a camera was re-aimed shows in one scrape.
- **Failure modes:** missing weights and no URL → `weights_missing` (503, hardware verdict `blocked`); a non-detection export mounted by mistake → `package_detection.unexpected_model_output`; every request-side error is a typed 400/413/415.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `OPENNVR_ADAPTER_TOKEN` | — | Bearer token; `/health` and `/metrics` stay open. |
| `PACKAGE_DETECTION_WEIGHTS_DIR` | `/weights` in the image, `model_weights/` from source | Where `yolov8n-package.onnx` lives. |
| `PACKAGE_DETECTION_MODEL_URL` | empty | First-boot fetch source; empty means never download. |
| `PACKAGE_DETECTION_LABELS` | `package` | Comma-separated class-index → label map for custom fine-tunes. |
