#!/usr/bin/env python3
"""CVD - Computer Vision Dashboard

Single entry point: starts the FastAPI backend and opens the frontend in a browser.
"""

import subprocess
import sys
import time
import webbrowser
import os

BACKEND_HOST = os.environ.get("CVD_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("CVD_BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("CVD_FRONTEND_PORT", "5173"))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root, "backend")
    frontend_dir = os.path.join(root, "frontend")

    print("=" * 50)
    print("  CVD - Computer Vision Dashboard")
    print("=" * 50)
    print()

    # Start backend
    print("[1/2] Starting backend server...")
    backend = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--reload",
        ],
        cwd=backend_dir,
    )

    # Start frontend
    print("[2/2] Starting frontend dev server...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
    )

    time.sleep(3)
    url = f"http://localhost:{FRONTEND_PORT}"
    print(f"\n  Dashboard: {url}")
    print(f"  API:       http://localhost:{BACKEND_PORT}/docs")
    print()
    print("  Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Done.")


if __name__ == "__main__":
    main()
