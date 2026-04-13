import pytest
from unittest.mock import patch
from app.utils.loader import PluginManager

@pytest.fixture(autouse=True)
def mock_plugin_discovery():
    """Prevent heavy ML libraries from loading during tests unless required"""
    with patch.object(PluginManager, "discover_plugins", return_value=None):
        yield

