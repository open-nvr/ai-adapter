# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Streaming inference — the §6 WebSocket protocol.

Demonstrates: `@Adapter.on_stream`, `HandshakeMessage`,
`HandshakeAckMessage`, `FrameMessage`, `ResultMessage`, `PauseMessage`,
`ResumeMessage`, `CloseMessage`, `StreamMessageType`, `FrameTransport`,
`StreamCloseCode`, and the AsyncAPI document it publishes.

`POST /infer` is one HTTP round-trip per frame — fine at a frame every
few seconds, wasteful at twenty a second. A stream is one session, one
warm model, and one correlation id for the whole episode, which is what
makes a sequence of frames traceable as one event rather than N
unrelated inferences. It also lets the adapter push back: `pause` and
`resume` are how a model that cannot keep up says so, instead of
silently queueing.

Declaring the handler advertises streaming in /capabilities and
publishes the protocol at /asyncapi.json.
"""
import json

from opennvr_adapter_sdk import Adapter
from opennvr_adapter_sdk.contract import (
    CloseMessage, FrameMessage, HandshakeAckMessage, HandshakeMessage,
    PauseMessage, ResultMessage, ResumeMessage, StreamCloseCode,
    StreamMessageType,
)

adapter = Adapter("demo-stream", tasks=["object_detection"])
MAX_QUEUE = 8


@adapter.load()
def load():
    return object()


@adapter.on_image()
def infer(call):
    """A streaming adapter still answers POST /infer — KAI-C falls back
    to it, and a caller that only needs one frame should not have to
    open a session."""
    return []


@adapter.on_stream()
async def stream(websocket) -> None:
    """One WebSocket = one camera's session.

    The shape below is the whole protocol: accept, handshake, then a
    frame/result loop with backpressure, and a close on the way out.
    """
    await websocket.accept()
    inflight = 0

    try:
        handshake = HandshakeMessage.model_validate_json(
            await websocket.receive_text())
    except Exception:
        # A malformed handshake is not a model failure; refuse the
        # session with a code the client can act on.
        await websocket.close(code=StreamCloseCode.POLICY_REFUSED,
                              reason="malformed handshake")
        return

    await websocket.send_text(HandshakeAckMessage(
        accepted=True,
        session_id=f"sess-{handshake.client_id}",
        # The session's correlation id threads every result in this
        # episode to the same causal chain in the audit trail.
        correlation_id=f"corr-{handshake.client_id}",
    ).model_dump_json())

    while True:
        message = json.loads(await websocket.receive_text())
        kind = message.get("type")

        if kind == StreamMessageType.CLOSE:
            break

        if kind == StreamMessageType.FRAME:
            frame_meta = FrameMessage.model_validate(message)
            # An inline frame is followed by ONE binary message.
            frame = await websocket.receive_bytes()

            if inflight >= MAX_QUEUE:
                # Backpressure, said out loud. Silently queueing turns
                # a slow model into growing latency nobody can see.
                await websocket.send_text(PauseMessage(
                    reason="queue full").model_dump_json())
                inflight = 0
                await websocket.send_text(ResumeMessage().model_dump_json())

            inflight += 1
            result = _run(frame)
            inflight -= 1

            await websocket.send_text(ResultMessage(
                seq=frame_meta.seq,          # echo it: the client pairs on seq
                ts_ms=frame_meta.ts_ms,
                inference_ms=7,
                result={"detections": result},
            ).model_dump_json())

    await websocket.send_text(CloseMessage(
        reason="client closed").model_dump_json())
    await websocket.close()


def _run(frame: bytes) -> list:
    return []


app = adapter.app
