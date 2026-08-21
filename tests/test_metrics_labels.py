# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""Model/task labels on the SDK metrics (observability slice 1).

The registry answers "which model, on which task, how fast, how often
failing" — per adapter, from one scrape. Two guarantees under test:

* ``adapter_model_info`` carries the model identity (name/version/
  framework/fingerprint) as labels, refreshed via model_info().
* ``task`` is a CLOSED label set: only advertised tasks become series;
  anything else (client-controlled strings) folds into "other", so a
  hostile client cannot mint unbounded series (cardinality bomb).
"""
from __future__ import annotations

from opennvr_adapter_sdk.metrics import Metrics


def _lines(m: Metrics) -> list[str]:
    return m.render().splitlines()


def test_model_info_labels_rendered_and_escaped():
    m = Metrics()
    m.set_model_info(
        adapter="moondream", adapter_version="1.2.0",
        model='moon"dream\\2', model_version="2b",
        framework="transformers", fingerprint="sha256:abc",
    )
    (info_line,) = [l for l in _lines(m) if l.startswith("adapter_model_info{")]
    assert 'adapter="moondream"' in info_line
    assert 'model="moon\\"dream\\\\2"' in info_line     # quote + backslash escaped
    assert 'fingerprint="sha256:abc"' in info_line
    assert info_line.endswith("} 1")


def test_model_info_drops_empty_values_and_is_absent_until_set():
    m = Metrics()
    assert not any(l.startswith("adapter_model_info") for l in _lines(m))
    m.set_model_info(adapter="piper", fingerprint=None, model="")
    (info_line,) = [l for l in _lines(m) if l.startswith("adapter_model_info{")]
    assert "fingerprint" not in info_line and "model=" not in info_line


def test_task_label_on_counter_and_histogram():
    m = Metrics(known_tasks=("scene_caption", "visual_qa"))
    m.record_infer("ok", 0.2, task="scene_caption")
    m.record_infer("ok", 1.2, task="scene_caption")
    m.record_infer("model_error", 0.1, task="visual_qa")
    body = m.render()
    assert 'adapter_infer_total{outcome="ok",task="scene_caption"} 2' in body
    assert 'adapter_infer_total{outcome="model_error",task="visual_qa"} 1' in body
    assert 'adapter_infer_latency_seconds_count{task="scene_caption"} 2' in body
    assert 'adapter_infer_latency_seconds_count{task="visual_qa"} 1' in body


def test_unknown_tasks_fold_into_other():
    """Client-controlled task strings must not mint unbounded series."""
    m = Metrics(known_tasks=("scene_caption",))
    for i in range(50):
        m.record_infer("ok", 0.01, task=f"attack-{i}")
    body = m.render()
    assert 'adapter_infer_total{outcome="ok",task="other"} 50' in body
    assert "attack-" not in body


def test_untasked_calls_keep_the_preseeded_empty_task_series():
    m = Metrics()
    m.record_infer("ok", 0.5)                     # no task (stream / legacy)
    body = m.render()
    assert 'adapter_infer_total{outcome="ok",task=""} 1' in body
    # all outcomes pre-seeded at 0 on task="" so dashboards see the axes
    assert 'adapter_infer_total{outcome="refused",task=""} 0' in body
