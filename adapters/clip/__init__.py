"""
CLIP embedding adapter — reference implementation of the AI Adapter
Contract v1 for the ``embed`` task.

Wraps OpenCLIP ``ViT-B-32`` (LAION ``laion2b_s34b_b79k``, MIT-licensed
weights, 512 dimensions) and returns one unit-length vector per call.

What makes it different from every other vision adapter here: it embeds
IMAGES AND WORDS INTO THE SAME SPACE, so an operator's sentence can be
compared against a frame. OpenNVR's search fuses a keyword ranking with
that similarity ranking, which is only meaningful if both sides land
somewhere comparable — a vision-only embedder would let you find frames
similar to another frame, which is a much smaller feature.

That is also why this adapter is ``BodyShape.TEXT`` rather than
``BodyShape.IMAGE``: half its requests carry no image at all. See
``service.py`` for the full reasoning.

Self-contained like ``adapters/package_detection/``: it owns its model
and preprocessing, so the image contains the SDK and this package.

Run with:
    python -m uvicorn adapters.clip.main:app --host 0.0.0.0 --port 9011
"""
