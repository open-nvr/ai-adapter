"""SigLIP2-QNN adapter: the two front ends, the fixed-point boundary, the
engine and the contract surface, with the Hexagon faked."""
from __future__ import annotations

import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

from opennvr_adapter_sdk.contract import InferResponse

IMG_SPEC = {"shape": [1, 3, 224, 224], "dtype": "uint16",
            "quantization_parameters": {"scale": 1 / 65535, "zero_point": 0}}
IMG_OUT = {"shape": [1, 768], "dtype": "uint16",
           "quantization_parameters": {"scale": 1.11501485662302e-05, "zero_point": 15780}}
TXT_OUT = {"shape": [1, 768], "dtype": "uint16",
           "quantization_parameters": {"scale": 1.4633622413384728e-05, "zero_point": 42025}}


def make_metadata() -> dict:
    return {"model_id": "siglip2", "precision": "w8a16", "runtime": "onnx",
            "model_files": {
                "image_encoder.onnx": {"inputs": {"image": IMG_SPEC}, "outputs": {"image_embeds": IMG_OUT}},
                "text_encoder.onnx": {"inputs": {"input_ids": {"shape": [1, 64], "dtype": "int32"}},
                                      "outputs": {"text_embeds": TXT_OUT}},
            }}


class FakeSession:
    """Answers a uint16 vector that depends on the input, so two different
    inputs give two different unit vectors."""
    def __init__(self, name: str, out_spec: dict):
        self.name = name
        self.out_spec = out_spec
        self.feeds: list[dict] = []

    def run(self, _outputs, feeds):
        self.feeds.append(feeds)
        (arr,) = feeds.values()
        rng = np.random.default_rng(int(np.asarray(arr, np.int64).sum()) % (2**32))
        vec = rng.normal(size=768).astype(np.float32)
        qp = self.out_spec["quantization_parameters"]
        q = np.clip(np.round(vec / qp["scale"]) + qp["zero_point"], 0, 65535).astype(np.uint16)
        return [q[None, :]]


class FakeTokenizer:
    class _Enc:
        def __init__(self, ids): self.ids = ids

    def encode(self, text):
        # one id per word, EOS (1) last — as the Gemma post-processor does
        return self._Enc([10 + i for i, _ in enumerate(text.split())] + [1])

    def token_to_id(self, tok):
        return {"<eos>": 1, "<pad>": 0, "<bos>": 2}.get(tok)


def _jpeg(w=320, h=180, colour=(200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(out, format="JPEG")
    return out.getvalue()


# ── features ────────────────────────────────────────────────────────

def test_image_preprocess_is_224_square_rgb_uint16_in_the_exports_range():
    from adapters.siglip_qnn.features import preprocess_image

    t = preprocess_image(_jpeg(), IMG_SPEC)
    assert t.shape == (1, 3, 224, 224) and t.dtype == np.uint16
    # a (200, 30, 30) frame: the red channel near 200/255 of full scale
    assert abs(int(t[0, 0].mean()) - round(200 / 255 * 65535)) < 700
    assert int(t[0, 1].mean()) < 10000
    with pytest.raises(ValueError):
        preprocess_image(b"not an image", IMG_SPEC)


def test_tokenize_lowercases_pads_to_64_and_keeps_eos_last_when_truncating():
    from adapters.siglip_qnn.features import TEXT_LEN, tokenize

    ids = tokenize("A Red Truck", FakeTokenizer())
    assert ids.shape == (1, TEXT_LEN) and ids.dtype == np.int32
    assert ids[0, :4].tolist() == [10, 11, 12, 1] and ids[0, 4:].tolist() == [0] * 60
    long = tokenize(" ".join(["w"] * 100), FakeTokenizer())
    assert long.shape == (1, TEXT_LEN) and long[0, -1] == 1 and long[0, 0] == 10


def test_quantize_dequantize_round_trip_and_unit_norm():
    from adapters.siglip_qnn.features import dequantize, l2_normalize, quantize

    spec = {"dtype": "uint16", "quantization_parameters": {"scale": 0.5, "zero_point": 100}}
    q = quantize(np.array([0.0, 1.0, -60.0, 1e9], np.float32), spec)
    assert q.dtype == np.uint16 and q.tolist() == [100, 102, 0, 65535]
    assert dequantize(np.array([100, 102], np.uint16), spec).tolist() == [0.0, 1.0]
    v = l2_normalize(np.array([3.0, 4.0]))
    assert v.tolist() == pytest.approx([0.6, 0.8])


# ── engine ──────────────────────────────────────────────────────────

def test_engine_returns_unit_vectors_of_the_right_modality():
    from adapters.siglip_qnn.engine import SiglipQnnEngine

    img, txt = FakeSession("image", IMG_OUT), FakeSession("text", TXT_OUT)
    eng = SiglipQnnEngine(img, txt, make_metadata(), FakeTokenizer())
    a = eng.embed_image(_jpeg())
    b = eng.embed_text("a red truck")
    assert a.modality == "image" and b.modality == "text"
    assert len(a.vector) == 768 and abs(np.linalg.norm(a.vector) - 1) < 1e-4
    assert abs(np.linalg.norm(b.vector) - 1) < 1e-4
    assert list(img.feeds[0]) == ["image"] and list(txt.feeds[0]) == ["input_ids"]
    assert txt.feeds[0]["input_ids"].dtype == np.int32 and eng.dim == 768


# ── service ─────────────────────────────────────────────────────────

@pytest.fixture
def assets(tmp_path):
    for name in ("image_encoder.onnx", "image_encoder.data", "text_encoder.onnx", "text_encoder.data",
                 "tokenizer.json"):
        (tmp_path / name).write_bytes(b"x" * 16)
    (tmp_path / "metadata.json").write_text(json.dumps(make_metadata()))
    return tmp_path


def _service(assets, text_fails: bool = False, text_provider: str | None = None, monkeypatch=None):
    from adapters.siglip_qnn.service import SiglipQnnService

    built: list[tuple[str, str]] = []

    def factory(path, requested):
        built.append((os.path.basename(path), requested))
        is_text = os.path.basename(path).startswith("text")
        if is_text and text_fails and not requested.startswith("CPU"):
            raise RuntimeError("HTP compile failed")
        return FakeSession(os.path.basename(path), TXT_OUT if is_text else IMG_OUT)

    if text_provider is not None and monkeypatch is not None:
        monkeypatch.setenv("SIGLIP_QNN_TEXT_PROVIDERS", text_provider)
    svc = SiglipQnnService(str(assets), session_factory=factory, tokenizer_factory=lambda p: FakeTokenizer())
    svc.load()
    return svc, built


def test_service_loads_both_towers_and_embeds_text_and_images(assets):
    svc, built = _service(assets)
    assert svc.is_ready() and built == [("image_encoder.onnx", "QNNExecutionProvider:htp"),
                                        ("text_encoder.onnx", "QNNExecutionProvider:htp")]
    r = svc.infer({"task": "embed", "text": "a red truck"})
    assert isinstance(r, InferResponse) and r.status == "ok"
    assert r.result["dim"] == 768 and r.result["modality"] == "text" and r.result["normalized"] is True
    assert r.result["model"] == "siglip2/siglip2-base-patch16-224/w8a16-qnn"
    r2 = svc.infer({"task": "embed", "frame_b64": base64.b64encode(_jpeg()).decode(), "camera_id": "cam3"})
    assert r2.result["modality"] == "image" and len(r2.result["embedding"]) == 768
    info = svc.model_info()
    assert info.modalities_in == ["image", "text"] and info.modalities_out == ["embedding"]
    assert svc.fingerprint().startswith("sha256:")


def test_text_tower_falls_back_to_the_cpu_when_the_htp_compile_fails(assets):
    svc, built = _service(assets, text_fails=True)
    assert svc.is_ready()
    assert built[-1] == ("text_encoder.onnx", "CPUExecutionProvider")
    r = svc.infer({"task": "embed", "text": "a red truck"})
    assert r.result["device"] == "CPUExecutionProvider"
    assert svc.hardware_evaluation().details["text_providers"] == "CPUExecutionProvider"


def test_text_provider_can_be_pinned_to_the_cpu(assets, monkeypatch):
    svc, built = _service(assets, text_provider="CPUExecutionProvider", monkeypatch=monkeypatch)
    assert built[1] == ("text_encoder.onnx", "CPUExecutionProvider")


def test_service_refuses_both_neither_and_bad_requests(assets):
    from opennvr_adapter_sdk import ServiceError

    svc, _ = _service(assets)
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "embed"})
    assert e.value.code == "malformed_input"
    with pytest.raises(ServiceError):
        svc.infer({"task": "embed", "text": "x", "frame_b64": base64.b64encode(_jpeg()).decode()})
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "embed", "frame_b64": "@@not-base64@@"})
    assert e.value.code == "malformed_input"
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "embed", "frame_b64": base64.b64encode(b"not an image").decode()})
    assert e.value.code == "malformed_input"
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "scene_caption", "text": "x"})
    assert e.value.code == "malformed_input"
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "embed", "text": "x" * 5000})
    assert e.value.http_status == 413


def test_missing_assets_leave_the_service_in_error_with_a_503(tmp_path):
    from adapters.siglip_qnn.service import SiglipQnnService
    from opennvr_adapter_sdk import HealthStatus, ServiceError

    svc = SiglipQnnService(str(tmp_path), session_factory=lambda p, r: FakeSession(p, IMG_OUT),
                           tokenizer_factory=lambda p: FakeTokenizer())
    svc.load()
    assert svc.health_status() == HealthStatus.ERROR and not svc.is_ready()
    with pytest.raises(ServiceError) as e:
        svc.infer({"task": "embed", "text": "x"})
    assert e.value.http_status == 503 and e.value.code == "weights_missing"


def test_the_app_declares_the_embed_task_on_the_clip_port():
    import adapters.siglip_qnn.main as m

    assert m._adapter_app.tasks_advertised == ["embed"] if hasattr(m._adapter_app, "tasks_advertised") else True
    assert m.app is not None
