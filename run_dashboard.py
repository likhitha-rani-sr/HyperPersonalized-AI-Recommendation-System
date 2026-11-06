import subprocess
import sys
import time
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PYTHON = sys.executable

API_URL = "http://127.0.0.1:8000/health"

DASHBOARD_URL = "http://127.0.0.1:8502"
API_STARTUP_TIMEOUT = 120


def wait_for_api(api_process, timeout=API_STARTUP_TIMEOUT):
    print("Waiting for FastAPI to start...")

    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        process_exit_code = api_process.poll()

        if process_exit_code is not None:
            print(
                "\nERROR: FastAPI exited before becoming ready "
                f"(exit code {process_exit_code})."
            )
            return False

        try:
            response = requests.get(API_URL, timeout=(1, 2))

            if response.status_code == 200:
                print("\nFastAPI started successfully.")
                print("API:     http://127.0.0.1:8000")
                print("Health:  http://127.0.0.1:8000/health")

                try:
                    health_data = response.json()
                except requests.exceptions.JSONDecodeError as exc:
                    print(f"WARNING: Could not parse health response: {exc}")
                else:
                    print(
                        "Products:",
                        health_data.get("products_loaded", "Unknown"),
                    )

                return True

            last_error = f"health endpoint returned HTTP {response.status_code}"

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)

        time.sleep(min(1, max(0, deadline - time.monotonic())))

    print("\nERROR: FastAPI did not start within the startup timeout.")
    if last_error:
        print(f"Last health check result: {last_error}")
    print("Check the FastAPI startup messages above for details.")
    return False


def stop_process(process):
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    api_process = None
    dashboard_process = None

    try:
        print("=" * 60)
        print("Hyper-Personalized AI Recommendation System")
        print("=" * 60)

        print("\nStarting FastAPI...")

        api_process = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=str(BASE_DIR),
        )

        if not wait_for_api(api_process):
            return 1

        print("\nStarting Streamlit dashboard...")

        dashboard_process = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "streamlit",
                "run",
                "src/dashboard.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8502",
                "--server.headless",
                "true",
                "--server.fileWatcherType",
                "none",
            ],
            cwd=str(BASE_DIR),
        )

        print("\n" + "=" * 60)
        print("SYSTEM STARTED")
        print("=" * 60)

        print("\nFastAPI:")
        print("  http://127.0.0.1:8000")

        print("\nHealth:")
        print("  http://127.0.0.1:8000/health")

        print("\nDashboard:")
        print("  http://127.0.0.1:8502")

        print("\nKeep this terminal open.")
        print("Press Ctrl+C to stop the system.")

        while True:
            if api_process.poll() is not None:
                print("\nFastAPI process stopped.")
                return 1

            if dashboard_process.poll() is not None:
                print("\nStreamlit process stopped.")
                return 1

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nStopping system...")
        return 0

    finally:
        stop_process(dashboard_process)
        stop_process(api_process)
        print("System stopped.")


if __name__ == "__main__":
    sys.exit(main())