"""Re-export shim — ``python -m conformance`` still works."""
from opennvr_adapter_sdk.conformance.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
