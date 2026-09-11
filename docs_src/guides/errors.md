# Failing well

The error category is not paperwork. KAI-C routes on it: a
`transport_error` is never retried, a `model_error` counts against the
adapter, an `overloaded` makes KAI-C back off and come back. Getting it
wrong turns one bad frame into a retry storm, or a busy adapter into one
the operator is told is broken.

## What the facade classifies for you

| You raise | Becomes |
|---|---|
| `ValueError`, `KeyError`, `TypeError` | 400 `transport_error`, not retried |
| `Overloaded(retry_after_ms=…)` | 503 `overloaded`, retried after the hint |
| anything else | 500 `model_error` |
| `ServiceError(...)` | passed through exactly as you wrote it |

So the common cases need no error handling at all:

```python
@adapter.on_image()
def detect(call):
    if not call.image:
        raise ValueError("a frame is required")     # → 400
    if _queue_depth() > 32:
        raise Overloaded(retry_after_ms=250)        # → 503
    return _run(call.model, call.image)             # anything else → 500
```

## When to be explicit

Two categories the facade cannot infer, because only you know them:

```python
--8<-- "cookbook/04_errors_and_backpressure.py:44:56"
```

`provider_error` is the other: an upstream your adapter fronts is down.
The adapter is fine, its dependency is not, and it is transient — so
KAI-C should retry.

## Backpressure is honest

Silently queueing turns a slow model into growing latency nobody can
see. `Overloaded` says so, with a number the caller can act on.

Full example:
[`04_errors_and_backpressure.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/04_errors_and_backpressure.py).
