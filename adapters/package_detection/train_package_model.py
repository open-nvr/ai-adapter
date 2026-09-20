#!/usr/bin/env python3
# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Reproduce (or re-train) the package-detection weights.

This is an AUTHORING script — it needs ``ultralytics`` (and therefore
torch), which the serving image deliberately does not install:

    pip install "ultralytics==8.3.240" onnx onnxslim roboflow

Two openly licensed doorstep datasets are merged into one ``package``
class:

  * *package at front door* v2 — 1,293 images, MIT
    https://universe.roboflow.com/package-detection/package-at-front-door
  * *Packages* v6 (raw) — 26 images, CC0 / public domain
    https://universe.roboflow.com/joseph-nelson/packages-g0ton

Roboflow downloads need a (free) account key in ``ROBOFLOW_API_KEY``.
Nothing else in this repo touches Roboflow; the key is read once here.

Usage:

    # 1. download + merge into ./package_data
    python adapters/package_detection/train_package_model.py download

    # 2. fine-tune YOLOv8n and export the dynamic-axes ONNX the adapter serves
    python adapters/package_detection/train_package_model.py train --epochs 40

    # 3. (optional) evaluate an ONNX on the held-out test split
    python adapters/package_detection/train_package_model.py evaluate model_weights/yolov8n-package.onnx

To fine-tune for YOUR porch: add your own frames + YOLO-format labels
under ``package_data/site/{images,labels}`` (one class, id 0) and run
``train`` again — they are picked up automatically. The
Deliveries page's "Not a package" / "Collected" corrections are the
frames such a set is built from.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO / "package_data"
WEIGHTS_OUT = REPO / "model_weights" / "yolov8n-package.onnx"

SOURCES = [
    # (workspace/project, version, prefix)
    ("package-detection/package-at-front-door", 2, "frontdoor"),
    ("joseph-nelson/packages-g0ton", 6, "cc0"),
]
IMGSZ = 416


def _export_link(project: str, version: int, key: str) -> str:
    import json
    url = f"https://api.roboflow.com/{project}/{version}/yolov8?api_key={key}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        body = json.load(resp)
    if "export" not in body:
        raise SystemExit(f"Roboflow did not return an export for {project}/{version}: {body}")
    return body["export"]["link"]


def download() -> None:
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY (free account → Settings → API keys).")
    merged = DATA / "merged"
    for split in ("train", "valid", "test"):
        (merged / split / "images").mkdir(parents=True, exist_ok=True)
        (merged / split / "labels").mkdir(parents=True, exist_ok=True)
    for project, version, prefix in SOURCES:
        print(f"→ {project} v{version}")
        zip_path = DATA / f"{prefix}.zip"
        urllib.request.urlretrieve(_export_link(project, version, key), zip_path)
        src = DATA / prefix
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(src)
        for split in ("train", "valid", "test"):
            for kind in ("images", "labels"):
                for f in (src / split / kind).glob("*"):
                    shutil.copy(f, merged / split / kind / f"{prefix}_{f.name}")
    (merged / "data.yaml").write_text(
        f"path: {merged}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 1\nnames: ['package']\n")
    for split in ("train", "valid", "test"):
        print(f"  {split}: {len(list((merged / split / 'images').glob('*')))} images")
    print(f"✓ merged dataset at {merged}")


def _add_site_frames(merged: Path) -> None:
    site = DATA / "site"
    if not (site / "images").is_dir():
        return
    n = 0
    for f in (site / "images").glob("*"):
        label = site / "labels" / (f.stem + ".txt")
        if not label.exists():
            continue
        shutil.copy(f, merged / "train" / "images" / f"site_{f.name}")
        shutil.copy(label, merged / "train" / "labels" / f"site_{label.name}")
        n += 1
    if n:
        print(f"  + {n} site frames from {site}")


def train(epochs: int, batch: int, device: str) -> None:
    from ultralytics import YOLO

    merged = DATA / "merged"
    if not (merged / "data.yaml").exists():
        raise SystemExit("Run `download` first.")
    _add_site_frames(merged)
    model = YOLO("yolov8n.pt")   # COCO-pretrained backbone; auto-downloads
    model.train(
        data=str(merged / "data.yaml"), epochs=epochs, imgsz=IMGSZ, batch=batch,
        device=device, project=str(DATA / "runs"), name="package", exist_ok=True,
        patience=12, close_mosaic=8, seed=0, plots=False,
    )
    best = DATA / "runs" / "package" / "weights" / "best.pt"
    export(best)


def export(pt: Path) -> None:
    from ultralytics import YOLO

    # dynamic=True matters: a fixed-size export rejects every imgsz but
    # the one it was exported at, and imgsz is the adapter's CPU dial.
    out = YOLO(str(pt)).export(format="onnx", dynamic=True, imgsz=IMGSZ, opset=12, simplify=True)
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(out, WEIGHTS_OUT)
    print(f"✓ {WEIGHTS_OUT} ({WEIGHTS_OUT.stat().st_size / 1e6:.1f} MB)")


def evaluate(weights: Path) -> None:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(DATA / "merged" / "data.yaml"), split="test", imgsz=IMGSZ,
        device="cpu", plots=False)
    print(f"test  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  "
          f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", default="cpu")
    e = sub.add_parser("export")
    e.add_argument("pt", type=Path)
    v = sub.add_parser("evaluate")
    v.add_argument("weights", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "download":
        download()
    elif args.cmd == "train":
        train(args.epochs, args.batch, args.device)
    elif args.cmd == "export":
        export(args.pt)
    elif args.cmd == "evaluate":
        evaluate(args.weights)


if __name__ == "__main__":
    main(sys.argv[1:])
