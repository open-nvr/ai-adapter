# Running the model

## The handler

```python
@adapter.on_image()
def detect(call):
    return [call.detection("fallen", 0.91, 0.1, 0.2, 0.3, 0.4)]
```

`call` is everything about one request:

| | |
|---|---|
| `call.image` / `.audio` / `.data` | the binary payload |
| `call.text` | the prompt, for a text adapter (checks `text`, `prompt`, `input`) |
| `call.params` | everything the caller sent besides the body |
| `call.param(name, default)` | one of them, with a default |
| `call.task` | which advertised task this call is for |
| `call.camera_id` | the camera, when the caller knows it |
| `call.model` | whatever `@adapter.load()` returned |
| `call.payload` | the raw dict `AdapterService.infer` would have received |

One adapter has one handler and one body shape. To serve several tasks,
branch on `call.task` inside it — that keeps `/capabilities` honest
about a single input shape.

## Returning an answer

```python
return [call.detection(...), ...]     # §5.1 detections
return {"caption": "a van at the gate"}   # your own shape, verbatim
return InferResponse(...)             # full control of the envelope
```

`call.detection` builds a contract-shaped item and **clamps the
coordinates to 0–1**, because normalized-versus-pixel is the mistake
that survives every test you write and only shows up as a box in the
wrong place on someone's screen.

## Follow a convention where one exists

```python
--8<-- "cookbook/03_result_conventions.py:38:50"
```

The convention types validate before the data reaches the wire, so a
malformed result fails in your process rather than in a consumer's.

Full example:
[`03_result_conventions.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/03_result_conventions.py).
