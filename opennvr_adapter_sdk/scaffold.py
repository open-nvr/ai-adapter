# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
``opennvr-adapter`` — the SDK's command line.

The tools used to live in the ai-adapter repository: a ``scaffold.sh``
that wrote into ``adapters/`` and a conformance runner that was never
packaged. Both were unreachable by the people they are for — somebody
with a model and ``pip install opennvr-adapter-sdk``. They ship here
now, and the scaffold produces a STANDALONE adapter project rather than
a directory inside someone else's repository.

    opennvr-adapter new my-model      # a runnable adapter + tests
    opennvr-adapter dev               # drive it in-process
    opennvr-adapter validate .        # the full conformance run
    opennvr-adapter spec              # its OpenAPI 3.1 document
    opennvr-adapter conform URL       # check a running adapter
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from opennvr_adapter_sdk import __version__

TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates" / "adapter"

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Default task for a scaffolded adapter — the one convention almost
#: every OpenNVR deployment already routes work to.
DEFAULT_TASK = "object_detection"


def kebab_to_snake(adapter_id: str) -> str:
    return adapter_id.replace("-", "_")


def kebab_to_title(adapter_id: str) -> str:
    return " ".join(part.capitalize() for part in adapter_id.split("-"))


def sdk_requirement(version: str = __version__) -> str:
    """Pin the SDK to its contract-major series: SDK 1.x targets
    contract v1, so ``>=1.2,<2.0`` is the honest range."""
    major = version.split(".", 1)[0]
    minor = version.split(".")[1] if "." in version else "0"
    return f"opennvr-adapter-sdk>={major}.{minor},<{int(major) + 1}.0"


def tokens(adapter_id: str, *, task: str, vendor: str, license_name: str,
           port: int) -> dict[str, str]:
    return {
        "__ADAPTER_ID__": adapter_id,
        "__ADAPTER_MODULE__": kebab_to_snake(adapter_id),
        "__ADAPTER_NAME__": kebab_to_title(adapter_id),
        "__TASK__": task,
        "__VENDOR__": vendor,
        "__LICENSE__": license_name,
        "__PORT__": str(port),
        "__SDK_REQUIREMENT__": sdk_requirement(),
    }


def substitute(text: str, table: dict[str, str]) -> str:
    # Longest first, so __ADAPTER_MODULE__ never loses to __ADAPTER_ID__.
    for key in sorted(table, key=len, reverse=True):
        text = text.replace(key, table[key])
    return text


def generate(adapter_id: str, dest_dir: Path, *, task: str = DEFAULT_TASK,
             vendor: str = "", license_name: str = "Apache-2.0",
             port: int = 9000) -> Path:
    """Write a standalone adapter project. Returns its directory."""
    if not _ID_RE.match(adapter_id or ""):
        raise ValueError(
            f"adapter id {adapter_id!r} is not kebab-case — lowercase letters "
            f"and digits with single hyphens (e.g. 'fall-detection'). It "
            f"becomes the adapter's name in /capabilities, the module name "
            f"and the image name."
        )
    target = Path(dest_dir).expanduser().resolve() / adapter_id
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    if not TEMPLATE_ROOT.is_dir():  # pragma: no cover — packaging error
        raise FileNotFoundError(f"template missing at {TEMPLATE_ROOT}")

    table = tokens(adapter_id, task=task, vendor=vendor or adapter_id,
                   license_name=license_name, port=port)
    (target / "tests").mkdir(parents=True)
    for source in sorted(TEMPLATE_ROOT.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(TEMPLATE_ROOT)
        written = target / Path(substitute(str(relative), table))
        written.parent.mkdir(parents=True, exist_ok=True)
        try:
            written.write_text(
                substitute(source.read_text(encoding="utf-8"), table),
                encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover — binary template asset
            shutil.copy2(source, written)
    return target


def print_next_steps(adapter_id: str, target: Path, *, task: str) -> None:
    module = kebab_to_snake(adapter_id)
    try:
        shown = target.relative_to(Path.cwd())
    except ValueError:
        shown = target
    print(f"\nScaffolded {adapter_id!r} at {target}\n")
    print("Next steps:")
    print(f"  cd {shown}")
    print("  uv sync --extra dev       # or: pip install -e '.[dev]'")
    print("  uv run pytest -q          # the smoke test — should be GREEN")
    print("  uv run opennvr-adapter dev            # drive it in-process")
    print("  uv run opennvr-adapter validate .     # the conformance run")
    print(f"  # open {module}.py: fill in load() and the handler — that's the model")
    print(f"\nIt advertises the {task!r} task. An app asks for a TASK, not for")
    print("your adapter by name, so matching an existing convention is what")
    print("makes your model a drop-in for someone else's.")


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opennvr-adapter",
        description=f"OpenNVR Adapter SDK {__version__} — scaffold, run and "
                    f"check AI adapters.")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="scaffold a runnable adapter project")
    new.add_argument("adapter_id", help="kebab-case id, e.g. 'fall-detection'")
    new.add_argument("--task", default=DEFAULT_TASK,
                     help=f"task convention to advertise (default: {DEFAULT_TASK})")
    new.add_argument("--vendor", default="", help="your name or organisation")
    new.add_argument("--license", dest="license_name", default="Apache-2.0",
                     help="SPDX licence for the generated project")
    new.add_argument("--port", type=int, default=9000, help="serving port")
    new.add_argument("--dest", default=None,
                     help="parent directory (default: the current one)")

    dev = sub.add_parser(
        "dev", help="load the adapter and drive its endpoints — no stack needed")
    dev.add_argument("path", nargs="?", default=".", help="the adapter directory")
    dev.add_argument("--image", default=None,
                     help="send this file instead of the built-in 1x1 frame")
    dev.add_argument("--task", default=None, help="task to send")
    dev.add_argument("--camera-id", default="cam-1")
    dev.add_argument("--param", action="append", default=[], metavar="K=V",
                     help="an extra caller param; repeatable")
    dev.add_argument("--repeat", type=int, default=1,
                     help="send the frame N times (warm-up, latency)")

    val = sub.add_parser(
        "validate", help="run the conformance suite against the adapter, in-process")
    val.add_argument("path", nargs="?", default=".", help="the adapter directory")
    val.add_argument("--json", action="store_true", help="machine-readable report")

    conform = sub.add_parser(
        "conform", help="run the conformance suite against a RUNNING adapter")
    conform.add_argument("url", help="base URL, e.g. http://localhost:9001")
    conform.add_argument("--token", default=None, help="bearer token")
    conform.add_argument("--json", action="store_true")
    conform.add_argument("--no-colour", action="store_true")

    spec = sub.add_parser("spec", help="print the adapter's OpenAPI / AsyncAPI")
    spec.add_argument("path", nargs="?", default=".", help="the adapter directory")
    spec.add_argument("--format", choices=("openapi", "asyncapi"),
                      default="openapi")
    spec.add_argument("--yaml", action="store_true", help="YAML instead of JSON")
    spec.add_argument("-o", "--output", default=None, help="write to this file")

    args = parser.parse_args(argv)

    if args.command == "new":
        dest = Path(args.dest).expanduser() if args.dest else Path.cwd()
        try:
            target = generate(args.adapter_id, dest, task=args.task,
                              vendor=args.vendor, license_name=args.license_name,
                              port=args.port)
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print_next_steps(args.adapter_id, target, task=args.task)
        return 0

    if args.command == "dev":
        from opennvr_adapter_sdk.dev import run_dev

        params: dict[str, object] = {}
        for item in args.param:
            key, _, value = item.partition("=")
            if not key or not _:
                print(f"error: --param expects K=V, got {item!r}", file=sys.stderr)
                return 2
            params[key] = _coerce(value)
        return run_dev(Path(args.path), image=args.image, params=params,
                       task=args.task, camera_id=args.camera_id,
                       repeat=args.repeat)

    if args.command == "validate":
        from opennvr_adapter_sdk.validate import run_validate

        return run_validate(Path(args.path), as_json=args.json)

    if args.command == "conform":
        from opennvr_adapter_sdk.conformance.__main__ import main as conform_main

        forwarded = [args.url]
        if args.token:
            forwarded += ["--token", args.token]
        if args.json:
            forwarded.append("--json")
        if args.no_colour:
            forwarded.append("--no-colour")
        return conform_main(forwarded)

    if args.command == "spec":
        from opennvr_adapter_sdk.speccmd import run_spec

        return run_spec(Path(args.path), fmt=args.format, as_yaml=args.yaml,
                        output=args.output)

    return 2  # pragma: no cover — argparse requires a subcommand


def _coerce(value: str) -> object:
    """`--param threshold=0.8` should arrive as a number, not "0.8"."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
