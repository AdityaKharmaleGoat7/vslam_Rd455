#!/usr/bin/env python3
"""Thin CLI entry point for the D455 scanner (kept for backwards
compatibility — the implementation lives in the ``scanner`` package).

Equivalent invocations:

    ./venv/bin/python realtime_pointcloud.py [options]
    ./venv/bin/python -m scanner [options]

Run with --help for all options; see README.md for usage and tuning.
"""

from scanner.cli import main

if __name__ == "__main__":
    main()
