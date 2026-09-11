# Streaming inference

`POST /infer` is one HTTP round-trip per frame — fine at a frame every
few seconds, wasteful at twenty a second. A stream is one session, one
warm model, and one correlation id for the whole episode, which makes a
sequence of frames traceable as a single event rather than N unrelated
inferences.

It is also the only way an adapter can push back: `pause` and `resume`
are how a model that cannot keep up says so.

## Declaring it

```python
@adapter.on_stream()
async def stream(websocket):
    ...
```

Declaring the handler advertises streaming in `/capabilities` and
publishes the protocol at `/asyncapi.json`. An adapter without it
answers `/infer/stream` with a 501 carrying the §7 envelope, rather than
leaving the caller to guess.

## The protocol

```
client                          adapter
  │ ── handshake ─────────────────▶│   camera, task, transport
  │ ◀──────────── handshake_ack ──│   session id, correlation id
  │ ── frame + binary ────────────▶│
  │ ◀──────────────────── result ──│   echoes seq
  │ ◀───────────────────── pause ──│   backpressure
  │ ◀──────────────────── resume ──│
  │ ── close ─────────────────────▶│
```

Every message is a contract type in `opennvr_adapter_sdk.contract`, so
validation is a `model_validate_json` rather than hand-parsing, and the
AsyncAPI document is generated from the same classes.

Echo `seq` on every result: it is how the client pairs an answer with
the frame that produced it when several are in flight.

Full example:
[`05_streaming.py`](https://github.com/open-nvr/ai-adapter/blob/main/cookbook/05_streaming.py).
