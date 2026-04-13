#!/usr/bin/env python3
# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Download model weight files required by enabled adapters.

Usage
-----
    # Download only what's enabled in config (default):
    uv run python download_models.py

    # Download everything regardless of config (e.g. for CI / full Docker builds):
    uv run python download_models.py --all

How it works
------------
The script reads TASK_ADAPTER_MAP and CONFIG["adapters"] from
app/config/config.py to determine which adapters are enabled, then
cross-references MODEL_REGISTRY below to find the download URLs and
filenames for each enabled adapter's weights.

Adding a new model
------------------
1. Add an entry to MODEL_REGISTRY below, keyed by adapter name.
2. Add the public download URL (or "ultralytics_export" for YOLO .onnx).
3. Add the weights_path that matches what CONFIG["adapters"] expects.

When users run `uv run python download_models.py`, only the models
for their enabled adapters will be fetched — keeping fresh installs lean.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
# Maps adapter_name → list of weight files it needs.
# Each entry is a dict with:
#   "filename"  : destination filename under model_weights/
#   "url"       : direct download URL, OR "ultralytics_export" for YOLO auto-export
#   "size_hint" : approximate size string shown during download
MODEL_REGISTRY: dict[str, list[dict]] = {
    "yolov8_adapter": [
        {
            "filename": "yolov8n.onnx",
            "url": "ultralytics_export",   # generated via ultralytics YOLO().export()
            "size_hint": "~6 MB",
        }
    ],
    "yolov11_adapter": [
        {
            "filename": "yolo11m.pt",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
            "size_hint": "~40 MB",
        }
    ],
    # insightface_adapter downloads its weights automatically via the InsightFace
    # library (buffalo_l pack) on first inference — no manual download needed.
    # blip_adapter and huggingface_adapter fetch from HuggingFace Hub on first use.
}


def _download_file(url: str, dest_path: Path) -> bool:
    """Download a file from `url` to `dest_path` with a progress indicator."""
    print(f"  Downloading {dest_path.name} ({url})")

    def _reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize > 0:
            downloaded = min(blocknum * blocksize, totalsize)
            pct = downloaded * 100.0 / totalsize
            bar_width = 30
            filled = int(bar_width * downloaded / totalsize)
            bar = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(f"\r  [{bar}] {pct:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, _reporthook)
        size_mb = dest_path.stat().st_size / 1024 / 1024
        print(f"\n  ✓ {dest_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"\n  ✗ Failed to download {dest_path.name}: {exc}")
        return False


def _ultralytics_export(filename: str, dest_path: Path) -> bool:
    """Generate an ONNX file from the corresponding YOLO .pt via Ultralytics."""
    pt_name = filename.replace(".onnx", ".pt")
    print(f"  Generating {filename} via Ultralytics (downloads {pt_name} first)...")
    try:
        from ultralytics import YOLO
        import shutil

        model = YOLO(pt_name)          # auto-downloads .pt if missing
        exported_path = model.export(format="onnx")
        if exported_path and os.path.exists(exported_path):
            shutil.move(exported_path, dest_path)
            # Clean up the .pt used for export (optional — saves ~13MB)
            if os.path.exists(pt_name):
                os.remove(pt_name)
            size_mb = dest_path.stat().st_size / 1024 / 1024
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ Ultralytics export succeeded but output not found")
            return False
    except ImportError:
        print(
            "  ✗ ultralytics not installed — run `uv sync --extra yolo` first"
        )
        return False
    except Exception as exc:
        print(f"  ✗ Export failed: {exc}")
        return False


def _load_enabled_adapters() -> set[str]:
    """
    Read app/config/config.py and return the set of adapter names that are
    both listed in CONFIG["adapters"] AND have "enabled": True.
    """
    try:
        import app.config.config as cfg
        return {
            name
            for name, settings in cfg.CONFIG.get("adapters", {}).items()
            if settings.get("enabled", False)
        }
    except Exception as exc:
        print(f"  Warning: could not read config ({exc}). Falling back to --all mode.")
        return set(MODEL_REGISTRY.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model weights for enabled OpenNVR AI adapters."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download weights for ALL adapters in the registry, "
             "regardless of what is enabled in config.py. "
             "Use this for CI pipelines or full Docker builds.",
    )
    args = parser.parse_args()

    model_weights_dir = Path(__file__).parent / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        target_adapters = set(MODEL_REGISTRY.keys())
        print("Mode: --all (downloading weights for every registered adapter)\n")
    else:
        target_adapters = _load_enabled_adapters() & set(MODEL_REGISTRY.keys())
        all_enabled = _load_enabled_adapters()
        no_weights_needed = all_enabled - set(MODEL_REGISTRY.keys())
        print(f"Mode: config-aware (enabled adapters: {sorted(all_enabled)})")
        if no_weights_needed:
            print(
                f"  Note: {sorted(no_weights_needed)} fetch their own weights "
                f"at first inference — no manual download needed."
            )
        print()

    if not target_adapters:
        print("Nothing to download.")
        return

    downloaded = skipped = failed = 0

    for adapter_name in sorted(target_adapters):
        entries = MODEL_REGISTRY.get(adapter_name, [])
        if not entries:
            continue

        print(f"[{adapter_name}]")
        for entry in entries:
            dest = model_weights_dir / entry["filename"]
            size_hint = entry.get("size_hint", "")

            if dest.exists():
                size_mb = dest.stat().st_size / 1024 / 1024
                print(f"  ✓ {entry['filename']} already exists ({size_mb:.1f} MB) — skipping")
                skipped += 1
                continue

            print(f"  → {entry['filename']} {size_hint}")
            url = entry["url"]
            if url == "ultralytics_export":
                ok = _ultralytics_export(entry["filename"], dest)
            else:
                ok = _download_file(url, dest)

            if ok:
                downloaded += 1
            else:
                failed += 1
        print()

    print("─" * 50)
    print(f"Done.  Downloaded: {downloaded}  |  Skipped: {skipped}  |  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
