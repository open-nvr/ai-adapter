# Adapter Template

A boilerplate-free starting point for a new contract v1 AI adapter.
Run one command, you get a working adapter directory matching the
shape every other adapter in this repo follows. Replace the
inference body, add your tests, ship.

## What it gives you

```
adapters/<your-adapter>/
├── __init__.py        package doc string
├── main.py            FastAPI app construction via AdapterApp
├── service.py         AdapterService impl — your model code goes here
├── Dockerfile         contract-compliant container build
└── README.md          adapter docs in the standard shape
```

Plus a stub test file at `tests/test_<your-adapter>_service.py` you
can grow from there.

What's already wired:

- Every contract endpoint (`/health`, `/capabilities`, `/hardware/
  evaluation`, `/metrics`, `/infer`, `/infer/stream` returning 501).
- Auth, correlation_id propagation, Prometheus metrics, multipart +
  JSON body parsing — all from the SDK, no code in your adapter.
- The `load() → is_ready() → infer()` lifecycle, with `/health`
  reporting `loading` / `ok` / `error` correctly out of the box.
- A test scaffold covering load lifecycle, malformed-input rejection,
  and the contract envelope shape — replace the toy logic with your
  model's expected behaviour.

What you fill in:

- `service.py`'s `load()`, `model_info()`, `hardware_evaluation()`,
  and `infer()` methods. Comments mark each with a `# TODO`.
- The Dockerfile's `pip install` line — add your model's ML deps.
- The README's "What it does" + "Operational notes" + "Why this
  model" sections.

## Usage

From the repo root:

```bash
./templates/adapter-template/scaffold.sh <adapter-slug> <port> <body-shape>
```

- `<adapter-slug>` — lowercase, hyphenated. E.g. `fall-detection`,
  `pose-estimation`, `audio-events`. Becomes the directory name
  (with underscores, since Python won't import hyphens) and the GHCR
  image name (with hyphens).
- `<port>` — 4-digit port the adapter listens on. The convention is
  9001-9006 are taken (piper/yolov8/whisper/fast-plate-ocr/insightface/blip);
  9007 is bytetrack; pick the next free port (9008+).
- `<body-shape>` — one of `IMAGE`, `AUDIO`, `TEXT`, `GENERIC`. Drives
  the SDK's body parser. Most vision adapters want `IMAGE`; most ASR
  adapters want `AUDIO`; tracker / LLM / post-processor adapters want
  `TEXT`.

Example:

```bash
./templates/adapter-template/scaffold.sh fall-detection 9008 IMAGE
```

Produces `adapters/fall_detection/` and
`tests/test_fall_detection_service.py`. Open the generated files and
follow the `# TODO` markers.

## After scaffolding

1. **Fill in the model wrapper.** `service.py`'s `load()` should
   eagerly load your weights / model state. `infer()` should run one
   inference call.
2. **Update the Dockerfile.** Add your ML library to the `pip install`
   line (also list it in your adapter's `pyproject.toml` if you keep
   one). Use a per-adapter image — not a shared monolith — so a heavy
   torch dep doesn't bleed into other adapters.
3. **Write real tests.** The scaffold ships skeleton tests. Replace
   the assertions with what your adapter actually guarantees:
   detection shape, error envelope on malformed input, fingerprint
   stability across loads.
4. **Run the conformance suite.** Once your adapter boots:
   ```bash
   python -m conformance http://localhost:<port> --token $OPENNVR_ADAPTER_TOKEN
   ```
   A green conformance run means KAI-C will register your adapter.
5. **Add to the publish workflow.** Append your adapter to the matrix
   in `.github/workflows/publish-images.yml` so the GHCR image gets
   built and pushed on every release.

## Why this exists

Without the template, every new adapter author rebuilds the same six
files from scratch, gets the contract endpoints slightly wrong, and
discovers their `/capabilities` doesn't match what KAI-C expects only
after pushing. The template enforces the "every adapter looks the
same" promise the gallery makes — read one adapter's `main.py` and
you know where everything lives in the others.

See [`adapters/bytetrack/`](../../adapters/bytetrack/) for a recent
adapter built straight from this template — closest reference for
a `BodyShape.TEXT` adapter; [`adapters/yolov8/`](../../adapters/yolov8/)
is the canonical `BodyShape.IMAGE` example with WebSocket streaming.
