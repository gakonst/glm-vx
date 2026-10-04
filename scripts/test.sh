#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" -m pytest tests kernels/test_backend.py -q
"$PYTHON" kernels/test_kernels.py
