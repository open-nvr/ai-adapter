# Being operable

Two audiences, and they want different things.

## KAI-C reads `/capabilities`

Identity, model info and fingerprint, the tasks advertised, permissions,
scheduling and cost. It re-reads every 60 seconds, and a changed
`model.fingerprint` is what drift detection keys on.

Declare permissions honestly: KAI-C refuses to register an adapter
asking for more than the operator granted, so over-declaring blocks the
deployment and under-declaring means the access you need is not there.

## The operator reads `/metrics`

The SDK already exports the §3.4 baseline — in-flight calls, inference
latency and outcome per task, a model-loaded gauge, and
`adapter_model_info` identity labels carrying the fingerprint.

Register your own for what only the model knows:

```python
--8<-- "cookbook/06_capabilities_and_metrics.py:41:52"
```

A counter an operator can alert on beats a log line nobody reads:
*"this camera has been sending unusable frames since 3am"* is a metric,
not a `logger.warning`.

## Health has to be honest

`/health` must go red when the model did not load. A green dot on a dead
adapter routes real work into a hole, and the operator has no way to
tell. The facade derives this from the loader, so the only way to get it
wrong is to swallow the exception yourself.

`is_ready()` is a bool, so on its own it cannot tell *"still loading"*
from *"the load failed"* — `/health` would say `loading` forever, and
both Docker's healthcheck and `opennvr-adapter validate` would pass a
dead adapter. If you implement `AdapterService` directly rather than
through the facade, override `health_status()` and return the real
`HealthStatus`:

```python
def health_status(self) -> HealthStatus | None:
    return self._state      # OK / DEGRADED / LOADING / ERROR
```

Returning `None` keeps the old bool-derived behaviour. The conformance
runner FAILs on `error` and WARNs on `loading`, so neither goes green.

## Weights: baked in or fetched

| | Baked into the image | Fetched on first load |
|---|---|---|
| Reproducible | yes | only if the URL is pinned |
| Air-gapped site | works | needs the volume pre-populated |
| Image size | large | small |
| Fingerprint | fixed at build | fixed after first boot |

`ensure_model_file` handles both: a file already present always wins and
nothing is downloaded, so the same image works in a deployment running
`sovereignty=local_only` where the operator pre-populated the volume.

## Threads: leave the box something

An adapter is one tenant on a box that also records, detects and — on a
voice install — transcribes and speaks. A runtime left to its defaults
takes every core for one request: ONNX Runtime, CTranslate2 and numpy's
BLAS all fan out, and the measured result on a 4-core box was one Piper
synthesis at ~390 % CPU and a voice turn of 71 s. Cap the intra-op
threads and make the cap an environment knob with a sane default, named
`OPENNVR_<ADAPTER>_THREADS` (the shipped adapters: `OPENNVR_PIPER_THREADS`,
default 2; `OPENNVR_WHISPER_CPU_THREADS`, CPU only). Log the value the
cap resolved to at start-up and report it in `/hardware` details, so
"set but ignored" — a CUDA box, a misspelt variable — is one line away
rather than a CPU graph. Document it in the adapter's README next to the
other knobs; the reference adapters' READMEs have a *Tuning on CPU*
table for exactly this.

Full examples:
[`06_capabilities_and_metrics.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/06_capabilities_and_metrics.py),
[`08_weights_and_packaging.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/08_weights_and_packaging.py).
