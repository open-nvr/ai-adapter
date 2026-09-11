# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""The tools a model developer gets from `pip install`.

Before this they got none: the scaffold was a `scaffold.sh` that wrote
into this repository's `adapters/` directory, and the conformance kit —
the one thing that tells someone their adapter will be accepted — was
not in the wheel at all. These tests pin the packaged CLI, and that what
it scaffolds is a standalone project that passes its own conformance
run out of the box.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from opennvr_adapter_sdk import __version__, scaffold
from opennvr_adapter_sdk.discovery import NotAnAdapter, find_module, load
from opennvr_adapter_sdk.scaffold import generate, sdk_requirement

#: The conformance suite's /infer check posts multipart, because that is
#: what KAI-C sends — so the full run needs FastAPI's form extra. The
#: SDK itself does not depend on it (the base64 JSON route always
#: works), so these skip rather than fail where it is absent.
requires_multipart = pytest.mark.skipif(
    importlib.util.find_spec("multipart") is None,
    reason="python-multipart not installed; the /infer conformance check "
           "posts multipart")


@pytest.fixture
def project(tmp_path):
    return generate("fall-detection", tmp_path, vendor="ACME")


def read(project, name: str) -> str:
    return (project / name).read_text(encoding="utf-8")


# ── new ─────────────────────────────────────────────────────────────


def test_the_scaffold_is_a_standalone_project(project):
    """Not a directory inside this repository — a project a third party
    can build, test and publish on their own."""
    files = {str(p.relative_to(project)) for p in project.rglob("*") if p.is_file()}
    assert files == {"fall_detection.py", "pyproject.toml", "Dockerfile",
                     "README.md", "tests/test_smoke.py"}
    assert "from opennvr_adapter_sdk import Adapter" in read(project, "fall_detection.py")
    # Nothing reaches back into the SDK's own repository.
    assert "adapters." not in read(project, "fall_detection.py")


def test_no_placeholder_survives(project):
    import re

    for path in project.rglob("*"):
        if path.is_file():
            leftovers = re.findall(r"__[A-Z_]+__", path.read_text(encoding="utf-8"))
            assert not leftovers, f"{path.name}: {leftovers}"


def test_the_sdk_is_pinned_to_its_contract_major(project):
    """SDK 1.x targets contract v1; a v2 contract would ship SDK 2.x, so
    an adapter must not float across that boundary."""
    requirement = sdk_requirement(__version__)
    assert requirement in read(project, "pyproject.toml")
    assert requirement in read(project, "Dockerfile")
    assert sdk_requirement("1.2.0") == "opennvr-adapter-sdk>=1.2,<2.0"


def test_the_task_and_identity_reach_every_file(project):
    assert 'tasks=["object_detection"]' in read(project, "fall_detection.py")
    assert 'vendor="ACME"' in read(project, "fall_detection.py")
    assert "# Fall Detection" in read(project, "README.md")
    assert 'name = "fall-detection"' in read(project, "pyproject.toml")


def test_a_custom_task_and_port_are_honoured(tmp_path):
    project = generate("plate-reader", tmp_path, task="license_plate_recognition",
                       port=9123, license_name="MIT")
    assert 'tasks=["license_plate_recognition"]' in read(project, "plate_reader.py")
    assert "9123" in read(project, "Dockerfile")
    assert "SPDX-License-Identifier: MIT" in read(project, "plate_reader.py")


def test_an_id_the_platform_cannot_use_is_refused(tmp_path):
    for bad in ("Fall Detection", "fall_detection", "FallDetection"):
        with pytest.raises(ValueError, match="kebab-case"):
            generate(bad, tmp_path)


def test_it_refuses_to_overwrite(project, tmp_path):
    with pytest.raises(FileExistsError):
        generate("fall-detection", tmp_path)


def test_new_prints_what_to_do_next(tmp_path, capsys):
    assert scaffold.main(["new", "fall-detection", "--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "opennvr-adapter dev" in out and "opennvr-adapter validate ." in out
    # The thing a model developer most needs to understand.
    assert "An app asks for a TASK" in out


def test_new_reports_a_bad_id_without_a_traceback(tmp_path, capsys):
    assert scaffold.main(["new", "Bad Id", "--dest", str(tmp_path)]) == 2
    assert "kebab-case" in capsys.readouterr().err


# ── discovery ───────────────────────────────────────────────────────


def test_the_adapter_module_is_found_by_pyproject_then_by_content(project):
    assert find_module(project) == "fall_detection"
    (project / "pyproject.toml").unlink()
    assert find_module(project) == "fall_detection"


def test_a_directory_with_no_adapter_says_so(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(NotAnAdapter, match="no adapter module found"):
        load(tmp_path / "empty")


def test_an_import_failure_is_reported_not_raised(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "broken.py").write_text(
        "from opennvr_adapter_sdk import Adapter\nadapter = Adapter('x')\n1 / 0\n")
    with pytest.raises(NotAnAdapter, match="ZeroDivisionError"):
        load(broken)


def test_a_module_that_builds_nothing_says_what_to_expose(tmp_path):
    from opennvr_adapter_sdk.discovery import asgi_app

    nothing = tmp_path / "nothing"
    nothing.mkdir()
    (nothing / "nothing.py").write_text("# mentions AdapterApp( but builds none\n")
    module, _ = load(nothing)
    with pytest.raises(NotAnAdapter, match="app = adapter.app"):
        asgi_app(module)


# ── dev ─────────────────────────────────────────────────────────────


def test_dev_drives_every_contract_endpoint(project, capsys):
    assert scaffold.main(["dev", str(project)]) == 0
    out = capsys.readouterr().out
    assert "opennvr-adapter dev — fall-detection 1.0.0" in out
    for endpoint in ("/health", "/capabilities", "/hardware/evaluation",
                     "/metrics", "/infer"):
        assert endpoint in out
    assert "tasks: object_detection" in out
    assert "All green" in out


def test_dev_shows_what_the_model_answered(tmp_path, capsys):
    project = generate("detector", tmp_path)
    module_file = project / "detector.py"
    module_file.write_text(module_file.read_text().replace(
        "    return []",
        '    return [call.detection("person", 0.87, 0.1, 0.2, 0.3, 0.4)]'))
    assert scaffold.main(["dev", str(project)]) == 0
    out = capsys.readouterr().out
    assert "detections: 1" in out
    assert "person" in out and "0.87" in out


def test_dev_reports_a_failing_handler_with_the_contract_category(tmp_path, capsys):
    project = generate("detector", tmp_path)
    module_file = project / "detector.py"
    module_file.write_text(module_file.read_text().replace(
        "    return []", '    raise ValueError("frame is not a JPEG")'))
    assert scaffold.main(["dev", str(project)]) == 1
    out = capsys.readouterr().out
    assert "transport_error" in out and "frame is not a JPEG" in out
    assert "validate" in out


def test_dev_passes_params_through_with_their_types(tmp_path, capsys):
    project = generate("detector", tmp_path)
    module_file = project / "detector.py"
    module_file.write_text(module_file.read_text().replace(
        "    return []",
        '    return {"seen": [type(call.param("threshold")).__name__,\n'
        '                     type(call.param("debug")).__name__,\n'
        '                     call.param("label")]}'))
    assert scaffold.main([
        "dev", str(project), "--param", "threshold=0.8",
        "--param", "debug=true", "--param", "label=person"]) == 0
    out = capsys.readouterr().out
    assert '"float"' in out and '"bool"' in out and '"person"' in out


def test_dev_rejects_a_malformed_param(project, capsys):
    assert scaffold.main(["dev", str(project), "--param", "nope"]) == 2
    assert "expects K=V" in capsys.readouterr().err


def test_dev_reports_a_missing_adapter(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert scaffold.main(["dev", str(tmp_path / "empty")]) == 2
    assert "no adapter module found" in capsys.readouterr().err


# ── validate ────────────────────────────────────────────────────────


@requires_multipart
def test_a_freshly_scaffolded_adapter_passes_conformance(project, capsys):
    """The bar for the scaffold: what `opennvr-adapter new` writes is
    already an adapter KAI-C would accept."""
    assert scaffold.main(["validate", str(project)]) == 0
    out = capsys.readouterr().out
    assert "opennvr-adapter validate — fall-detection 1.0.0" in out
    for check in ("health", "capabilities", "hardware_evaluation", "metrics",
                  "infer"):
        assert check in out
    assert "KAI-C will accept this adapter" in out


@requires_multipart
def test_validate_reports_machine_readably(project, capsys):
    assert scaffold.main(["validate", str(project), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["adapter"] == {"name": "fall-detection", "version": "1.0.0"}
    assert report["ok"] is True
    assert report["summary"]["failed"] == 0
    assert {c["name"] for c in report["checks"]} >= {"health", "capabilities", "infer"}


def test_validate_fails_an_adapter_that_breaks_the_contract(tmp_path, capsys):
    """An adapter whose model never loads must not report itself
    healthy — that is the failure KAI-C most needs to be told about,
    because a green dot on a dead adapter routes real work into a hole."""
    project = generate("detector", tmp_path)
    module_file = project / "detector.py"
    module_file.write_text(module_file.read_text().replace(
        "    return None\n", '    raise RuntimeError("weights are missing")\n', 1))
    code = scaffold.main(["validate", str(project), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["ok"] is False
    failed = [c for c in report["checks"] if c["outcome"] == "FAIL"]
    assert failed, report["checks"]


# ── spec ────────────────────────────────────────────────────────────


def test_spec_prints_the_openapi_document(project, capsys):
    assert scaffold.main(["spec", str(project)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["openapi"] == "3.1.0"
    assert document["info"]["x-opennvr-tasks"] == ["object_detection"]
    assert "InferResponse" in document["components"]["schemas"]


def test_spec_prints_the_asyncapi_document(project, capsys):
    assert scaffold.main(["spec", str(project), "--format", "asyncapi"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["asyncapi"] == "3.0.0"
    # The scaffold does not stream, and the document says so rather than
    # advertising a protocol the adapter will answer 501 to.
    assert document["channels"] == {}


def test_spec_writes_a_file(project, tmp_path, capsys):
    out = tmp_path / "generated" / "openapi.json"
    assert scaffold.main(["spec", str(project), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["openapi"] == "3.1.0"
    assert "wrote openapi" in capsys.readouterr().err


def test_spec_can_emit_yaml(project, capsys):
    pytest.importorskip("yaml")
    import yaml

    assert scaffold.main(["spec", str(project), "--yaml"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["openapi"] == "3.1.0"


def test_spec_reports_a_missing_adapter(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert scaffold.main(["spec", str(tmp_path / "empty")]) == 2
    assert "no adapter module found" in capsys.readouterr().err
