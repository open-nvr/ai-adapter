# LLM adapter package.
#
# WHY THIS FILE IS INTENTIONALLY EMPTY:
# Adapters are auto-discovered by PluginManager (app/utils/loader.py).
# Importing classes here would eagerly load transformers, huggingface_hub,
# and torch at server startup — even for deployments that only use local
# vision models. See app/adapters/vision/__init__.py for full explanation.
#
# To add a new LLM adapter:
#   1. Create app/adapters/llm/my_adapter.py
#   2. Extend BaseAdapter with name/type set
#   3. Register it in app/config/config.py
#   → No import needed here. Discovery is automatic.

