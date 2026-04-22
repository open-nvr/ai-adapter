# Audio adapter package.
#
# WHY THIS FILE IS INTENTIONALLY EMPTY:
# Adapters are auto-discovered by PluginManager (app/utils/loader.py) which
# walks this package via pkgutil.walk_packages and imports each .py file
# individually. Importing adapter classes here would eagerly load their
# top-level dependencies (faster-whisper, torchaudio, ctranslate2…) for ALL
# audio adapters at server startup — even adapters that are disabled or whose
# optional dependencies are not installed.
#
# To add a new audio adapter:
#   1. Create app/adapters/audio/my_adapter.py
#   2. Define a class that extends BaseAdapter with name="my_adapter", type="audio"
#   3. Register it in app/config/config.py under CONFIG["adapters"]
#   4. Map its task(s) in TASK_ADAPTER_MAP
#   → No import needed here. Discovery is automatic.
