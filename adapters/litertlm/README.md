# litertlm — Gemma on the Qualcomm Hexagon NPU

Captions and visual question answering from **Gemma 4 E2B** running on the
**Hexagon NPU** of Qualcomm Dragonwing boxes (Arduino Ventuno Q / IQ-8275,
QCS8275, Hexagon v75) through Google's LiteRT-LM. A drop-in for the
`caption-adapter` slot: same tasks (`scene_caption` → `caption`,
`visual_qa` → `answer`) as the moondream and ollamavlm adapters.

**Measured on a Ventuno Q (2026-11-05):** a 124 KB evidence photo captioned in
2.5 s warm / 4.3 s cold with the image encoder on the NPU, ~1.2 GiB resident,
the detector sharing the NPU slowed from 38 to 44 ms per inference. The CPU
captioner the box ran before (qwen3-vl 2B via Ollama) took 60–90 s per photo.

## What it needs

- The NPU: `/dev/fastrpc-cdsp`, `/dev/dma_heap/system`,
  `/dev/dma_heap/qcom,system`, the `fastrpc` and `dmaheap` groups, and the
  host's `libcdsprpc.so` — open-nvr's `docker-compose.accel-qualcomm.yml`
  passes all of them to `caption-adapter`.
- The model: `gemma-4-E2B-it_Qualcomm_QCS8275.litertlm` (2.7 GB, Apache-2.0),
  downloaded once into `/models` when `OPENNVR_LITERTLM_MODEL_URL` is set, or
  mounted there. It is NPU-only (no CPU graph), compiled for QCS8275 with
  QAIRT 2.50.
- Six Qualcomm AI Runtime libraries, fetched once at first start into
  `/opt/npu/lib` from Qualcomm's download server (pinned SHA-256). Qualcomm's
  terms apply to them; they are not redistributed in the image.

Without the NPU node the adapter stays up, `/health` says so, and
`/hardware/evaluation` is `blocked` with the reason.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `OPENNVR_LITERTLM_MODEL_URL` | empty | One-time model download source (open-nvr sets the Hugging Face URL on Qualcomm boxes). |
| `OPENNVR_LITERTLM_MODEL_SHA256` | the published digest | Verified after download; set empty to skip. |
| `OPENNVR_LITERTLM_MAX_IMAGE_PX` | 768 | Longest side the photo is scaled to before the model (tokens scale with pixels). |
| `OPENNVR_LITERTLM_MAX_OUTPUT_TOKENS` | 96 | Caption/answer length cap. |
| `OPENNVR_LITERTLM_CAPTION_PROMPT` | one-sentence security-camera prompt | The caption instruction. |

Port 9020. The context window is fixed at compile time (4096 tokens).
