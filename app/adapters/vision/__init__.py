# Vision adapter package.
#
# WHY THIS FILE IS INTENTIONALLY EMPTY:
# Adapters are auto-discovered by PluginManager (app/utils/loader.py) which
# walks this package via pkgutil.walk_packages and imports each .py file
# individually. Importing adapter classes here would eagerly load their
# top-level dependencies (cv2, numpy, onnxruntime, insightface, ultralytics…)
# for ALL adapters at server startup — even adapters that are disabled or
# whose optional dependencies are not installed.
#
# To add a new vision adapter:
#   1. Create app/adapters/vision/my_adapter.py
#   2. Define a class that extends BaseAdapter with name/type set
#   3. Register it in app/config/config.py
#   → No import needed here. Discovery is automatic.

