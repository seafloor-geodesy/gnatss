#!/usr/bin/env -S uv run --script

from __future__ import annotations

from pathlib import Path

DIR = Path(__file__).parent.resolve()

if not (DIR / "data" / "2022").exists():
    print("Test data doesn't exist. Downloading...")
    from gnatss.utilities.testing import download_test_data
    download_test_data(unzip=True)