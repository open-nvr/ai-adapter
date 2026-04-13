#!/usr/bin/env python3
import uvicorn
import sys

def main():
    print("Starting OpenNVR AI-Adapter server...")
    try:
        # Run the FastAPI app via uvicorn programmatically
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=9100, 
            reload=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
