# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Failing well — `ServiceError`, `ErrorCategory`, `Overloaded`.

Demonstrates: `ServiceError`, `ErrorCategory`, `FailureEnvelope`,
`ErrorDetail`, `Overloaded`, and what the facade classifies for you.

The category is not paperwork: KAI-C routes on it. A `transport_error`
is not retried, a `model_error` counts against the adapter, an
`overloaded` makes KAI-C back off and come back. Getting this wrong
turns one bad frame into a retry storm, or a busy adapter into one the
operator is told is broken.
"""
from opennvr_adapter_sdk import Adapter, Overloaded, ServiceError
from opennvr_adapter_sdk.contract import ErrorCategory

adapter = Adapter("demo-errors", tasks=["object_detection"])


@adapter.load()
def load():
    return object()


@adapter.on_image()
def infer(call):
    """What the facade does for you, and when to be explicit."""

    # ── Classified for you ─────────────────────────────────────────
    #
    #   ValueError / KeyError / TypeError → 400 transport_error, not
    #       retried. The caller sent something this model cannot use.
    #   anything else                     → 500 model_error.
    #   Overloaded                        → 503 with retry_after_ms.
    if not call.image:
        raise ValueError("a frame is required")

    if _queue_depth() > 32:
        raise Overloaded("inference queue is full", retry_after_ms=250)

    # ── Be explicit when the category is not obvious ───────────────

    if call.param("match_faces") and not _face_matching_permitted():
        # The operator's policy, not a model failure. PERMISSION_DENIED
        # tells KAI-C this will not succeed on retry either, and the
        # message reaches the operator's audit log.
        raise ServiceError(
            ErrorCategory.PERMISSION_DENIED,
            code="face_matching_refused",
            message="Site policy does not permit face matching on this camera.",
            transient=False,
            http_status=403,
        )

    try:
        return _run(call.model, call.image)
    except ConnectionError as exc:
        # An upstream this adapter fronts is down — the adapter is
        # fine, its provider is not. Transient, so KAI-C retries.
        raise ServiceError(
            ErrorCategory.PROVIDER_ERROR,
            code="upstream_unavailable",
            message=f"The inference provider is unreachable: {exc}",
            transient=True,
            http_status=502,
            retry_after_ms=5000,
        ) from exc


#: What each category means to the caller, in one place.
#:
#: ==================  ======  =========  ===================================
#: category            status  retried?   use it when
#: ==================  ======  =========  ===================================
#: transport_error     400     no         the request is malformed
#: permission_denied   403     no         policy refused this call
#: not_supported       501     no         this adapter does not do that
#: model_error         500     no         valid input, the model failed
#: provider_error      502     yes        an upstream dependency failed
#: overloaded          503     yes        backpressure — come back later
#: ==================  ======  =========  ===================================
CATEGORIES = tuple(ErrorCategory)


def _queue_depth() -> int:
    return 0


def _face_matching_permitted() -> bool:
    return True


def _run(model, frame):
    return []
