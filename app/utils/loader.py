import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

from app.interfaces.adapter import BaseAdapter
from app.interfaces.task import BaseTask

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Maps optional-dep packages → install hint shown when an adapter is skipped
_INSTALL_HINTS: dict[str, str] = {
    "insightface":        "uv sync --extra face",
    "onnxruntime":        "uv sync --extra yolo",
    "ultralytics":        "uv sync --extra yolo11",
    "transformers":       "uv sync --extra blip",
    "torch":              "uv sync --extra cpu  (or --extra gpu)",
    "torchvision":        "uv sync --extra cpu  (or --extra gpu)",
    "huggingface_hub":    "uv sync --extra huggingface",
    "scipy":              "uv sync --extra face",
    "faster_whisper":     "uv sync --extra stt",
    "ctranslate2":        "uv sync --extra stt",
    "piper":              "uv sync --extra tts",
}


@dataclass
class _DiscoverySummary:
    """Tracks what was found / skipped during a discovery run."""
    registered: list[str] = field(default_factory=list)
    skipped_missing_dep: list[tuple[str, str]] = field(default_factory=list)  # (module, hint)
    skipped_error: list[tuple[str, str]] = field(default_factory=list)  # (module, error)


class PluginManager:
    """
    Discovers and registers task/adapter plugin classes for lazy instantiation.

    Discovery algorithm
    -------------------
    1. Walk every .py file (non-private, non-dunder) under app.adapters and
       app.pipelines using pkgutil.walk_packages.
    2. Import each module. On ImportError (optional dep not installed) the
       module is **skipped gracefully** — not a crash. A helpful install hint
       is logged so developers know exactly what to install.
    3. For each class defined in the module that is:
         • a concrete (non-abstract) subclass of BaseAdapter or BaseTask
         • defined in *this* module (not imported from another)
       register it by its ``name`` class attribute (or class name as fallback).

    Registry contents
    -----------------
    Only CLASS REFERENCES are stored — no instances, no model loading.
    Instances (and hence models) are created lazily by ModelRouter on the
    first request that requires them.
    """

    TASK_REGISTRY: dict[str, type[BaseTask]] = {}
    ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}
    _discovered = False

    @classmethod
    def discover_plugins(cls, force_reload: bool = False) -> None:
        if cls._discovered and not force_reload:
            return

        cls.TASK_REGISTRY = {}
        cls.ADAPTER_REGISTRY = {}

        task_summary = _DiscoverySummary()
        adapter_summary = _DiscoverySummary()

        cls._scan_package("app.pipelines", BaseTask, cls.TASK_REGISTRY, task_summary)
        cls._scan_package("app.adapters", BaseAdapter, cls.ADAPTER_REGISTRY, adapter_summary)
        cls._discovered = True

        cls._log_summary("tasks",    task_summary,    cls.TASK_REGISTRY)
        cls._log_summary("adapters", adapter_summary, cls.ADAPTER_REGISTRY)

    @classmethod
    def _scan_package(
        cls,
        package_name: str,
        base_class: Type[T],
        registry: dict[str, type[T]],
        summary: _DiscoverySummary,
    ) -> None:
        package = importlib.import_module(package_name)
        package_paths = getattr(package, "__path__", None)

        if not package_paths:
            return

        package_root = Path(next(iter(package_paths)))
        if not package_root.exists():
            return

        for module_info in pkgutil.walk_packages(package_paths, prefix=f"{package_name}."):
            module_name = module_info.name
            # Skip private/dunder modules (e.g. __init__, _helpers)
            if module_name.rsplit(".", 1)[-1].startswith("_"):
                continue
            cls._register_module_classes(module_name, base_class, registry, summary)

    @classmethod
    def _register_module_classes(
        cls,
        module_name: str,
        base_class: Type[T],
        registry: dict[str, type[T]],
        summary: _DiscoverySummary,
    ) -> None:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            # ── Expected path ──────────────────────────────────────────────
            # Optional dependency not installed. This is NOT a bug — it means
            # this deployment profile doesn't need this adapter. Log at INFO
            # (not WARNING) and suggest the right extra to install.
            missing_pkg = _extract_missing_package(exc)
            hint = _INSTALL_HINTS.get(missing_pkg, f"uv sync --extra <profile>")
            logger.info(
                "Skipping '%s': optional dependency '%s' not installed. "
                "Install with: %s",
                module_name, missing_pkg, hint,
            )
            summary.skipped_missing_dep.append((module_name, hint))
            return
        except Exception as exc:
            # ── Unexpected path ────────────────────────────────────────────
            # This is a real bug (syntax error, bad import, etc.). Warn loudly.
            logger.warning(
                "Skipping module '%s' due to unexpected error: %s",
                module_name, exc,
            )
            summary.skipped_error.append((module_name, str(exc)))
            return

        for _, discovered_class in inspect.getmembers(module, inspect.isclass):
            if discovered_class.__module__ != module.__name__:
                continue
            if discovered_class is base_class or not issubclass(discovered_class, base_class):
                continue
            if inspect.isabstract(discovered_class):
                continue

            plugin_name = cls._resolve_plugin_name(discovered_class)
            registry[plugin_name] = discovered_class
            summary.registered.append(plugin_name)

    @staticmethod
    def _resolve_plugin_name(discovered_class: type) -> str:
        name = getattr(discovered_class, "name", None)
        if isinstance(name, str) and name.strip():
            return name
        return discovered_class.__name__

    @staticmethod
    def _log_summary(
        kind: str,
        summary: _DiscoverySummary,
        registry: dict,
    ) -> None:
        """Emit a clean single-line summary of discovery results."""
        registered_count = len(summary.registered)
        skipped_dep = len(summary.skipped_missing_dep)
        skipped_err = len(summary.skipped_error)

        if registered_count:
            logger.info(
                "Discovered %d %s: [%s]",
                registered_count, kind, ", ".join(sorted(registry.keys())),
            )
        else:
            logger.info("No %s discovered.", kind)

        if skipped_dep:
            logger.info(
                "%d %s module(s) skipped (optional dep not installed). "
                "Run 'uv sync --extra all' to enable everything.",
                skipped_dep, kind,
            )

        if skipped_err:
            logger.warning(
                "%d %s module(s) skipped due to errors. Check logs above.",
                skipped_err, kind,
            )


def _extract_missing_package(exc: ImportError) -> str:
    """
    Best-effort extraction of the missing package name from an ImportError.

    ImportError.name contains the top-level module that could not be found.
    We split on '.' to get just the root package name (e.g. 'insightface'
    from 'insightface.app').
    """
    name = getattr(exc, "name", None) or str(exc)
    # str(exc) looks like: "No module named 'insightface'"
    if name.startswith("No module named "):
        name = name.removeprefix("No module named ").strip("'\"")
    return name.split(".")[0]
