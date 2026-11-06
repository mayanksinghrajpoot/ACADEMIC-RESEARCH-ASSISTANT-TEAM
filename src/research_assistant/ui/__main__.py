"""Open the local research page."""

from __future__ import annotations

import argparse
from pathlib import Path

from research_assistant.ui.server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open the local research assistant page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port, Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
