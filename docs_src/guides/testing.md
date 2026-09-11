# Testing

## Conformance is the bar

```bash
opennvr-adapter validate .                    # in-process, no server
opennvr-adapter conform http://localhost:9001 --token $TOKEN
```

`validate` drives the real ASGI app through FastAPI's test client, so it
needs no port and belongs in CI on every commit. A green run means KAI-C
will accept the adapter — the only assurance available without a
deployment to try it in.

As a test:

```python
--8<-- "cookbook/07_testing_and_conformance.py:43:57"
```

## The four tests worth writing about your model

1. **It answers in the contract shape**, with coordinates in 0–1.
2. **A body the model cannot use is a 400, not a 500** — the distinction
   decides whether KAI-C retries.
3. **`/capabilities` advertises a task and a fingerprint** — without the
   first the adapter gets no work, without the second it is exempt from
   drift detection.
4. **Health is honest about a failed load.**

The scaffold ships all four; keep them green as the handler becomes
real.

## Before you ship

```bash
opennvr-adapter dev             # watch the model answer
opennvr-adapter validate .      # the conformance run
opennvr-adapter spec            # the OpenAPI document you will publish
```

Full example:
[`07_testing_and_conformance.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/07_testing_and_conformance.py).
