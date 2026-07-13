#!/usr/bin/env python3
"""whiscribe — voice recording + transcription via whisper.cpp (CLI entry point).

Thin shim so the script stays runnable via a PATH symlink; the implementation
lives in cli.py (terminal front end) and backend.py (shared logic)."""

import pathlib
import sys

# Ensure sibling modules import even when invoked through a symlink.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from cli import main

if __name__ == "__main__":
    main()
