from __future__ import annotations

import os
import sys
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    url = os.getenv("AGENT_HEALTHCHECK_URL", "http://127.0.0.1:8501/_stcore/health")
    try:
        with urlopen(url, timeout=3) as response:  # noqa: S310 - URL 来自受控部署配置
            return 0 if response.status == 200 else 1
    except (OSError, URLError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
