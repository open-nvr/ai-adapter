# opennvr-adapter-conformance

CLI that validates an AI adapter against the [AI Adapter Contract v1](../../open-nvr/docs/AI_ADAPTER_CONTRACT.md).

Point it at any reachable adapter URL. It probes every mandatory endpoint, validates wire shapes against the Pydantic types in `app/interfaces/contract.py`, and reports PASS / WARN / FAIL / SKIP. A green run means KAI-C will accept the adapter.

## Use

```bash
# Pretty output
python -m conformance http://localhost:9001 --token <bearer>

# Machine-readable
python -m conformance http://localhost:9001 --json > report.json

# Disable colour (useful in CI)
python -m conformance http://localhost:9001 --no-colour
```

The CLI exits non-zero if any check FAILs — wire it into CI to gate adapter PRs.

## What it checks

| Check | Asserts |
|---|---|
| `base_url` | http/https scheme; warns on non-loopback HTTP |
| `health` | GET /health returns 200 within 1s; body validates as `HealthResponse` |
| `capabilities` | GET /capabilities returns valid `CapabilitiesResponse`; warns if `model.fingerprint` is null |
| `hardware_evaluation` | GET /hardware/evaluation returns valid `HardwareEvaluationResponse` |
| `metrics` | GET /metrics emits Prometheus exposition with the §3.4 baseline metrics |
| `infer` | POST /infer with a sample payload returns either `InferResponse` or a typed `FailureEnvelope` |
| `infer_stream` | Adapters that declared `infer_stream.supported = false` MUST refuse with HTTP 501; supported adapters get a basic reachability check (full WS handshake is A2.2) |

## Sample output

```
OpenNVR Adapter Contract v1 — conformance report for http://localhost:9001

  [PASS] base_url                        http://localhost:9001
  [PASS] health                      8ms  status=ok
  [PASS] capabilities               12ms  adapter=piper-tts@1.0.0
  [PASS] hardware_evaluation         3ms  verdict=ok
  [PASS] metrics                     5ms  18 lines of Prometheus exposition
  [PASS] infer                     412ms  inference_ms=410
  [PASS] infer_stream                4ms  HTTP 501 with FailureEnvelope as expected (adapter does not stream).

  7 pass · 0 warn · 0 fail · 0 skip

  GREEN — KAI-C will accept this adapter.
```

## Use in tests

The runner is duck-typed on the client object — pass FastAPI's `TestClient` directly to run conformance in-process with no real network:

```python
from fastapi.testclient import TestClient
from conformance.runner import ConformanceRunner

with TestClient(my_adapter_app) as client:
    runner = ConformanceRunner(base_url="", client=client)
    report = runner.run_all()
    assert report.is_green
```

This is exactly how `tests/test_conformance_against_piper.py` validates Piper.

## Adding sample payloads for new tasks

The runner looks up a sample input by adapter's `tasks_advertised`. If your new adapter advertises a task name the runner doesn't recognise, add an entry to `ConformanceRunner.SAMPLE_INFER_PAYLOADS` (`conformance/runner.py`); otherwise the `infer` check WARNs instead of exercising the endpoint.

## Roadmap

- v1 (now): validates the HTTP surface end-to-end.
- v1.1: full WebSocket handshake exercise (lands with YOLOv8 in A2.2).
- v1.2: tamper-detection probe — re-fetch `/capabilities` after N seconds and verify `model.fingerprint` is stable.
- v1.5: integration with KAI-C registry — `kaic register --conformance-check` rejects registration unless this runs green.
