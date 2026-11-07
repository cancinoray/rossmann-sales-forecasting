"""Container healthcheck for either service in the image.

The default CMD is the FastAPI app on 9696; overriding CMD with Streamlit
serves the dashboard on 8501. The API gets a real ``/health`` request, the
dashboard a TCP connect check.
"""

from __future__ import annotations

import socket
import urllib.request

API_HEALTH = "http://127.0.0.1:9696/health"
DASHBOARD_PORT = 8501


def main() -> int:
    try:
        with urllib.request.urlopen(API_HEALTH, timeout=3) as response:
            if response.status == 200:
                return 0
    except OSError:
        pass

    try:
        with socket.create_connection(("127.0.0.1", DASHBOARD_PORT), timeout=3):
            return 0
    except OSError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
