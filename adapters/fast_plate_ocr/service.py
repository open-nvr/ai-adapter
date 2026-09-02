# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
FastPlateOcrService — license-plate OCR implementation of
``AdapterService``.

Wraps the `fast-plate-ocr <https://github.com/ankandrew/fast-plate-ocr>`_
ONNX models. The adapter is deliberately single-purpose: it takes a
**pre-cropped license-plate image** and returns the recognized
characters. Vehicle detection and plate-region cropping happen
upstream — the canonical pipeline is YOLOv8 (vehicle / plate ROI) →
this adapter (OCR on the crop). The `license-plate-recognition`
example app drives that chain end-to-end.

Why a plate-specific OCR engine instead of generic Tesseract / Paddle?

* Generic OCR is trained on document-style text on light backgrounds.
  Plates have a mix of dark-on-light and light-on-dark, weird fonts,
  reflective surfaces, and high motion blur. A plate-specific model
  trained on plate fonts is dramatically more accurate at small image
  sizes.
* ``fast-plate-ocr`` ships its own ONNX weights (Apache-2.0) and runs
  on ``onnxruntime`` alone — no PyTorch, no Paddle, no heavyweight
  framework. The whole install is ~30 MB on top of the SDK.

The adapter advertises ``BodyShape.IMAGE`` and ``modalities_in=[image]``;
streaming WS is intentionally NOT supported because LPR is event-driven
(one inference per detected vehicle), not frame-rate.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import (
    AdapterService,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ServiceError,
)

logger = logging.getLogger(__name__)

# Default model identifier — ``fast-plate-ocr`` ships multiple weight
# bundles tuned for different regions. ``cct-xs-v1-global-model`` is
# the smallest and most general; operators can override via the
# ``OPENNVR_LPR_MODEL`` env var without code changes.
DEFAULT_MODEL_ID: str = "cct-xs-v1-global-model"
# Overall confidence is the MIN per-character probability (see
# _parse_recognizer_output): clean reads score ~0.99, misreads and
# hallucinations cluster under ~0.45. Tunable per-install via
# OPENNVR_LPR_MIN_CONFIDENCE — raise for fewer false reads, lower for
# fewer misses.
DEFAULT_MIN_CONFIDENCE: float = 0.45

# Plate LOCALIZATION stage (field fix: the platform sends whole-VEHICLE
# crops, but fast-plate-ocr is a pure OCR model that reads whatever it
# is pointed at — on an uncropped vehicle it hallucinates character
# shapes; the first production read was a garbage "1023" off a van
# whose plate was not even legible). ``open-image-models`` (same
# author, same ONNX/CPU footprint, ~7 MB weights downloaded on first
# load) localizes the plate first; OCR then runs on the crop, which
# takes a 15%-confidence hallucination to a >99% exact read.
DEFAULT_DETECTOR_ID: str = "yolo-v9-t-384-license-plate-end2end"
DEFAULT_DETECT_CONFIDENCE: float = 0.35
#: Margin added around the detected plate box before OCR (fraction of
#: box size per side) — a sliver of context helps the OCR model.
_CROP_MARGIN_X: float = 0.10
_CROP_MARGIN_Y: float = 0.20
#: When NO plate is localized we still OCR the whole image — a tight
#: plate crop (the original documented contract) may legitimately not
#: re-detect — but at a RAISED confidence floor, because whole-image
#: OCR of a scene is exactly the garbage generator described above. A
#: real plate crop reads at ~0.99 and sails through; scene noise
#: almost never reaches 0.75.
DEFAULT_UNLOCALIZED_MIN_CONFIDENCE: float = 0.75
#: Minimum localized-plate width (px) worth OCRing. Below this the
#: characters are a few pixels tall and the model guesses — a distant
#: car should produce NO read, not a wrong one (a wrong read is a
#: false "unknown vehicle" alarm downstream). Overridable via
#: OPENNVR_LPR_MIN_PLATE_PX.
DEFAULT_MIN_PLATE_PX: int = 40

# Maximum image bytes per call. A typical plate crop is well under
# 100 KB; the 2 MB cap protects against an upstream pipeline shipping
# a full uncropped frame by mistake.
MAX_IMAGE_BYTES: int = 2 * 1024 * 1024


class FastPlateOcrService(AdapterService):
    """Stateful façade around ``fast_plate_ocr.LicensePlateRecognizer``."""

    def __init__(
        self,
        model_id: str | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._model_id: str = model_id or os.getenv(
            "OPENNVR_LPR_MODEL", DEFAULT_MODEL_ID
        )
        try:
            self._min_confidence: float = float(os.getenv(
                "OPENNVR_LPR_MIN_CONFIDENCE", "") or min_confidence)
        except ValueError:
            self._min_confidence = float(min_confidence)
        # Localization config. OPENNVR_LPR_DETECTOR="" disables the
        # detection stage entirely (pure-OCR, the pre-1.1 behaviour).
        self._detector_id: str = os.getenv(
            "OPENNVR_LPR_DETECTOR", DEFAULT_DETECTOR_ID
        ).strip()
        try:
            self._detect_confidence: float = float(os.getenv(
                "OPENNVR_LPR_DETECT_CONF", "") or DEFAULT_DETECT_CONFIDENCE)
        except ValueError:
            self._detect_confidence = DEFAULT_DETECT_CONFIDENCE
        try:
            self._unlocalized_floor: float = float(os.getenv(
                "OPENNVR_LPR_UNLOCALIZED_MIN_CONFIDENCE", "")
                or DEFAULT_UNLOCALIZED_MIN_CONFIDENCE)
        except ValueError:
            self._unlocalized_floor = DEFAULT_UNLOCALIZED_MIN_CONFIDENCE
        try:
            self._min_plate_px: int = int(os.getenv(
                "OPENNVR_LPR_MIN_PLATE_PX", "") or DEFAULT_MIN_PLATE_PX)
        except ValueError:
            self._min_plate_px = DEFAULT_MIN_PLATE_PX
        self._detector: Any | None = None
        # True when a CONFIGURED detector failed to load — degraded
        # mode keeps the raised unlocalized floor (we cannot vouch for
        # inputs). Explicitly disabling via OPENNVR_LPR_DETECTOR=""
        # is different: the operator asserts they send tight crops,
        # so the caller's floor applies unchanged.
        self._detection_degraded: bool = False
        # Recognizer is built lazily inside ``load()`` so module
        # import does not trigger an ONNX-weights download.
        self._recognizer: Any | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        # (path, st_size, st_mtime_ns) → sha256. Avoids re-hashing the
        # ONNX weights on every /capabilities poll while still firing
        # the §11.3 drift event if the file is rotated underneath us.
        self._fingerprint_stat_key: tuple[str, int, int] | None = None
        self._lock = threading.Lock()
        self._started_at: datetime = datetime.now(timezone.utc)

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            # Domain metrics (registration is idempotent in the SDK).
            self.metrics.register_counter(
                "adapter_plate_reads_total",
                "Plate OCR reads, by whether they met the confidence floor.",
                label_key="result",
                allowed_values=("accepted", "below_threshold"))
            try:
                # Import inside ``load()`` per the project's
                # "lazy heavy imports" convention — keeps adapter
                # discovery fast and avoids pulling onnxruntime into
                # discovery-only code paths.
                from fast_plate_ocr import LicensePlateRecognizer

                self._recognizer = LicensePlateRecognizer(self._model_id)
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "FastPlateOcrService ready: model=%s fingerprint=%s",
                    self._model_id, self._fingerprint_cache,
                )
                # Localization stage — an ENHANCER, never a gate: if
                # the detector cannot load, the adapter stays healthy
                # and runs pure OCR (with the raised unlocalized
                # floor), and the WARN below is the operator signal.
                if self._detector_id:
                    try:
                        from open_image_models import create_detector

                        self._detector = create_detector(
                            self._detector_id,
                            conf_thresh=self._detect_confidence,
                        )
                        logger.info(
                            "plate localization ready: detector=%s "
                            "conf>=%.2f", self._detector_id,
                            self._detect_confidence,
                        )
                    except Exception:
                        self._detector = None
                        self._detection_degraded = True
                        logger.warning(
                            "plate detector %s failed to load — running "
                            "OCR-only. Whole-image reads use the raised "
                            "%.2f confidence floor; expect misses on "
                            "vehicle crops until this is fixed.",
                            self._detector_id, self._unlocalized_floor,
                            exc_info=True,
                        )
                else:
                    logger.info(
                        "plate localization disabled via "
                        "OPENNVR_LPR_DETECTOR=\"\" — pure-OCR mode")
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception(
                    "FastPlateOcrService failed to load model %s",
                    self._model_id,
                )

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        # Recompute on every call so KAI-C's /capabilities poll
        # detects an on-disk weight rotation. If the file is missing
        # (cloud-cached model, network-mounted weights down), fall
        # back to the cached value so we don't lose identity entirely.
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_id,
            version=f"fast-plate-ocr/{self._model_id}",
            framework="onnx",
            modalities_in=["image"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (
                f"fast-plate-ocr runs on CPU via onnxruntime; "
                f"model {self._model_id!r} loaded."
            )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Model failed to load: {self._load_error}"

        uptime_s = int(
            (datetime.now(timezone.utc) - self._started_at).total_seconds()
        )
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "model_id": self._model_id,
                "min_confidence": self._min_confidence,
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Read a license plate from an image.

        Accepts EITHER a tight plate crop (the original contract) or a
        whole vehicle / scene image: a localization stage finds the
        plate first and OCR runs on the crop. When no plate is
        localized, the whole image is OCR'd at a raised confidence
        floor (see DEFAULT_UNLOCALIZED_MIN_CONFIDENCE).

        ``payload['__file__']`` holds the raw image bytes
        (multipart upload or base64-decoded JSON). Result shape:

        .. code-block:: python

            {
                "plate_text": "ABC1234",
                "confidence": 0.93,
                "characters": [
                    {"char": "A", "confidence": 0.97},
                    {"char": "B", "confidence": 0.91},
                    ...
                ],
                "accepted": True,
                "min_confidence_applied": 0.30,
                "model_id": "cct-xs-v1-global-model",
            }
        """
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="model_not_loaded",
                message=self._load_error or "fast-plate-ocr model not yet loaded",
                transient=transient,
                http_status=503,
                retry_after_ms=2000 if transient else None,
            )

        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="missing_image",
                message=(
                    "fast-plate-ocr expects a license-plate crop in the "
                    "request body (multipart 'frame' field or JSON "
                    "'frame_b64'). No image bytes were provided."
                ),
                transient=False,
                http_status=400,
            )

        # Optional per-call override of the confidence threshold.
        # Validated up front so a malformed value fails with a typed
        # TRANSPORT_ERROR envelope rather than a downstream KeyError /
        # TypeError from the ONNX wrapper.
        threshold = payload.get("min_confidence", self._min_confidence)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_min_confidence",
                message=f"min_confidence must be a float; got {threshold!r}",
                transient=False,
                http_status=400,
            )
        if not (0.0 <= threshold <= 1.0):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="min_confidence_out_of_range",
                message=(
                    "min_confidence must be in [0.0, 1.0]; "
                    f"got {threshold}"
                ),
                transient=False,
                http_status=400,
            )

        # Decode raw bytes → numpy array. ``fast-plate-ocr`` 1.x's
        # ``LicensePlateRecognizer.run()`` only accepts
        # ``str | list[str] | numpy.ndarray | list[numpy.ndarray]`` —
        # passing the request body bytes directly would crash inside
        # the library with an opaque AttributeError. The decode also
        # gives us a clean place to surface "invalid image bytes" as
        # a typed TRANSPORT_ERROR envelope rather than letting it
        # masquerade as a model error.
        #
        # cv2.IMREAD_COLOR gives a 3-channel HxWx3 BGR array; fast-
        # plate-ocr's PlateConfig handles the colour-mode conversion
        # internally (the library docstring says it accepts grayscale
        # OR colour and converts per the loaded config).
        try:
            image_array = _decode_image_bytes(image_bytes)
        except _ImageDecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_image",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc

        # ── Plate localization (the vehicle-crop fix) ──────────────
        # Detected: OCR the plate crop at the caller's floor.
        # Not detected (or detector off/failed): OCR the whole image,
        # but at the RAISED unlocalized floor — a genuine tight plate
        # crop still passes (~0.99), scene hallucinations don't.
        detection_info: dict[str, Any] = {
            "attempted": self._detector is not None,
            "found": False,
            "confidence": None,
            "box": None,
            # The image the box (if any) is measured in — width, height
            # of exactly what the caller sent. Consumers rejecting
            # CLIPPED (partial) reads need the box AND the frame it was
            # measured in; shipping both together means the geometry can
            # never be judged against the wrong image (open-nvr#378).
            "image_size": [int(image_array.shape[1]),
                           int(image_array.shape[0])],
            "model_id": self._detector_id if self._detector else None,
        }
        ocr_input = image_array
        effective_threshold = threshold
        if self._detector is not None:
            try:
                candidates = self._detector.predict(image_array)
            except Exception:
                logger.exception("plate detector failed; falling back "
                                 "to whole-image OCR")
                candidates = []
            best = None
            for det in candidates or []:
                conf = float(getattr(det, "confidence", 0.0) or 0.0)
                if conf < self._detect_confidence:
                    continue
                if best is None or conf > best[0]:
                    best = (conf, det.bounding_box)
            if best is not None:
                conf, box = best
                bw = int(box.x2) - int(box.x1)
                if bw < self._min_plate_px:
                    # Plate localized but too small to read honestly:
                    # return a clean non-read instead of a guess.
                    detection_info.update(
                        found=True,
                        confidence=round(conf, 4),
                        box=[int(box.x1), int(box.y1),
                             int(box.x2), int(box.y2)],
                        too_small=True,
                        min_plate_px=self._min_plate_px,
                    )
                    try:
                        self.metrics.inc_counter(
                            "adapter_plate_reads_total",
                            label_value="below_threshold")
                    except Exception:  # pragma: no cover
                        pass
                    return InferResponse(
                        model_name=self._model_id,
                        model_version=f"fast-plate-ocr/{self._model_id}",
                        inference_ms=0,
                        result={
                            "plate_text": "",
                            "confidence": 0.0,
                            "characters": [],
                            "accepted": False,
                            "min_confidence_applied": threshold,
                            "model_id": self._model_id,
                            "plate_detection": detection_info,
                        },
                    )
                crop = _crop_with_margin(image_array, box)
                if crop is not None:
                    ocr_input = crop
                    detection_info.update(
                        found=True,
                        confidence=round(conf, 4),
                        box=[int(box.x1), int(box.y1),
                             int(box.x2), int(box.y2)],
                    )
        if not detection_info["found"] and (
                detection_info["attempted"] or self._detection_degraded):
            effective_threshold = max(threshold, self._unlocalized_floor)

        started = time.monotonic()
        try:
            raw = self._recognizer.run(  # type: ignore[union-attr]
                ocr_input, return_confidence=True
            )
        except Exception as exc:
            logger.exception("fast-plate-ocr inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="ocr_failed",
                message=f"fast-plate-ocr raised during inference: {exc}",
                transient=True,
                http_status=500,
                retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        plate_text, characters, overall_conf = _parse_recognizer_output(raw)

        # Apply the confidence floor — adapter returns the best
        # candidate it has, marked with a status flag so the caller
        # can decide whether to drop the alert.
        accepted = overall_conf >= effective_threshold
        try:
            self.metrics.inc_counter(
                "adapter_plate_reads_total",
                label_value="accepted" if accepted else "below_threshold")
        except Exception:  # pragma: no cover - metrics must never break infer
            pass

        return InferResponse(
            model_name=self._model_id,
            model_version=f"fast-plate-ocr/{self._model_id}",
            inference_ms=elapsed_ms,
            result={
                "plate_text": plate_text,
                "confidence": round(overall_conf, 4),
                "characters": characters,
                "accepted": accepted,
                "min_confidence_applied": effective_threshold,
                "model_id": self._model_id,
                # Additive (v1.1): how the input was localized. found=
                # False + a low read means "no plate visible here", not
                # a broken adapter — consumers can tell the difference.
                "plate_detection": detection_info,
            },
        )

    # ── helpers ────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        """sha256 of the underlying ONNX model file, or a synthetic
        fingerprint based on the model identifier when the file path
        can't be discovered (e.g. the recognizer caches in-memory
        only).

        Caches the sha256 keyed on (path, size, mtime_ns) so we don't
        re-hash a large weights file every 60 seconds when KAI-C
        polls /capabilities. The §11.3 drift event still fires on
        file rotation — the stat key changes, the hash gets
        recomputed, and the returned fingerprint differs."""
        path = _recognizer_model_path(self._recognizer)
        if path and os.path.isfile(path):
            stat = os.stat(path)
            stat_key = (path, stat.st_size, stat.st_mtime_ns)
            if (
                self._fingerprint_stat_key == stat_key
                and self._fingerprint_cache
                and self._fingerprint_cache.startswith("sha256:")
            ):
                return self._fingerprint_cache
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(64 * 1024), b""):
                    h.update(chunk)
            fp = f"sha256:{h.hexdigest()}"
            self._fingerprint_stat_key = stat_key
            self._fingerprint_cache = fp
            return fp
        # Fallback: identifier-derived fingerprint. Still stable per
        # model_id, so the §11.3 drift check will only fire on a
        # model swap, never spuriously.
        digest = hashlib.sha256(self._model_id.encode("utf-8")).hexdigest()
        return f"sha256-id:{digest}"


def _parse_recognizer_output(raw: Any) -> tuple[str, list[dict[str, Any]], float]:
    """Translate fast-plate-ocr's varied return shapes into a single
    canonical tuple. The library has moved through several return
    shapes across versions; we accept all of them and produce the
    same wire shape regardless:

    * ``PlatePrediction`` object with ``.plate`` + ``.char_probs``
      (fast-plate-ocr 1.1.0, the shipped version — overall confidence
      is the MIN of the per-character probabilities) or ``.plate`` +
      ``.confidence`` (other versions/forks).
    * Bare string ``"ABC1234"``.
    * Tuple ``("ABC1234", 0.93)``.
    * List of any of the above (batch).
    * Two-tuple of parallel arrays ``([texts], [confs])``.

    Returns ``(plate_text, characters, overall_confidence)``. When the
    library doesn't report per-character confidences we synthesise the
    list from the overall score so the §5 wire shape is still complete.
    """
    text: str
    overall: float

    # Unwrap the outer list/tuple if present — we only ever need the
    # first prediction (this adapter is single-plate; batch shape is
    # handled by repeating the call).
    if isinstance(raw, (list, tuple)) and raw:
        # Detect ``([texts], [confs])`` shape: a 2-tuple of two
        # parallel sequences. Be defensive — make sure raw[0] is a
        # sequence too, else we're looking at a normal list whose
        # first element happens to be a string.
        if (
            isinstance(raw, tuple)
            and len(raw) == 2
            and isinstance(raw[0], (list, tuple))
            and isinstance(raw[1], (list, tuple))
            and raw[0] and raw[1]
        ):
            first = raw[0][0]
            overall = float(raw[1][0])
        else:
            first = raw[0]
            overall = None  # type: ignore[assignment]  # filled in below
    else:
        first = raw
        overall = None  # type: ignore[assignment]

    # Pull text + confidence off the (now-unwrapped) first prediction.
    # Most-specific dispatch first; the str case is a catch-all.
    char_confs: list[float] | None = None
    plate_attr = getattr(first, "plate", None)
    if plate_attr is not None:
        # PlatePrediction-style object. fast-plate-ocr 1.1.0's REAL
        # shape carries ``char_probs`` (per-character probabilities),
        # NOT ``.confidence`` — the old branch here defaulted a missing
        # ``.confidence`` to 1.0, which stamped EVERY production read
        # as fully confident: the acceptance floor was inert, garbage
        # like "1023" shipped as an accepted read, and raising the
        # floor changed nothing. Aggregate as MIN of the per-character
        # probabilities: a plate read is only as trustworthy as its
        # least certain character (measured: a clean read mins ~0.999,
        # a hallucination mins ~0.08 — mean would blur that apart).
        text = str(plate_attr)
        if overall is None:
            probs = getattr(first, "char_probs", None)
            if probs is not None:
                try:
                    char_confs = [float(p) for p in list(probs)]
                except (TypeError, ValueError):
                    char_confs = None
            if char_confs:
                overall = min(char_confs)
            else:
                attr_conf = getattr(first, "confidence", None)
                # No per-char probs AND no confidence attribute →
                # treat as UNTRUSTED (0.0), never as certain (1.0):
                # an unknown-confidence read must not outrank the
                # acceptance floor.
                overall = float(attr_conf) if attr_conf is not None else 0.0
    elif isinstance(first, dict):
        # Dict-shape prediction (some forks).
        text = str(first.get("plate") or first.get("text") or "")
        if overall is None:
            overall = float(first.get("confidence") or 0.0)
    elif isinstance(first, (list, tuple)) and len(first) >= 2:
        text = str(first[0])
        if overall is None:
            overall = float(first[1])
    elif isinstance(first, str):
        text = first
        if overall is None:
            overall = 0.0   # bare string = confidence unknown = untrusted
    else:
        text = str(first)
        if overall is None:
            overall = 0.0

    text = text.strip()
    if char_confs and len(char_confs) >= len(text):
        characters = [
            {"char": ch, "confidence": round(char_confs[i], 4)}
            for i, ch in enumerate(text)
        ]
    else:
        characters = [
            {"char": ch, "confidence": round(overall, 4)} for ch in text
        ]
    return text, characters, overall


def _recognizer_model_path(recognizer: Any) -> str | None:
    """Best-effort dig into the recognizer for the ONNX model path
    so the fingerprint reflects on-disk bytes. fast-plate-ocr's
    internal attribute names have moved across versions, so we
    accept any of the documented names and shrug if none is set."""
    if recognizer is None:
        return None
    for attr in ("model_path", "_model_path", "onnx_path", "_onnx_path"):
        value = getattr(recognizer, attr, None)
        if isinstance(value, str):
            return value
    return None


# ── Image decoding ─────────────────────────────────────────────────


def _crop_with_margin(image, box) -> Any | None:
    """Crop the detected plate box plus a small context margin,
    clamped to the frame. Returns None for a degenerate box (zero
    area after clamping) so the caller falls back to whole-image OCR
    rather than feeding the OCR an empty array."""
    h, w = image.shape[:2]
    bw = max(0, int(box.x2) - int(box.x1))
    bh = max(0, int(box.y2) - int(box.y1))
    mx = int(bw * _CROP_MARGIN_X)
    my = int(bh * _CROP_MARGIN_Y)
    x1 = max(0, int(box.x1) - mx)
    y1 = max(0, int(box.y1) - my)
    x2 = min(w, int(box.x2) + mx)
    y2 = min(h, int(box.y2) + my)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]


class _ImageDecodeError(Exception):
    """Raised when request body bytes can't be decoded as an image.

    Separate from ``ServiceError`` so the call site can wrap it with
    the right ``ErrorCategory.TRANSPORT_ERROR`` envelope — keeps the
    decode logic itself dependency-free for easier unit testing."""


def _decode_image_bytes(image_bytes: bytes):
    """Decode request body bytes into a 3-channel BGR ``numpy.ndarray``.

    fast-plate-ocr 1.x's ``LicensePlateRecognizer.run()`` only accepts
    paths or numpy arrays, NOT raw bytes. We decode here so the
    adapter contract (image bytes in the request body) matches what
    the library actually consumes. ``cv2.imdecode`` handles JPEG /
    PNG / WebP / BMP via OpenCV's built-in codecs — same dependency
    surface the InsightFace and YOLOv8 adapters already use, so no
    new install footprint.
    """
    # Lazy import — keeps the module importable in test environments
    # that haven't installed opencv-python yet (same pattern the
    # InsightFace service uses).
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    if not image_bytes:
        raise _ImageDecodeError(
            "empty request body — fast-plate-ocr expects a plate "
            "crop in the request body (multipart 'frame' field or "
            "JSON 'frame_b64')"
        )

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise _ImageDecodeError(
            "could not decode image bytes — expected JPEG, PNG, "
            "WebP, or BMP (opencv handles the codec sniff)"
        )
    return image
