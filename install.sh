#!/bin/sh
# Linux / NVIDIA / Python 3.10-3.12. Creates an isolated environment.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BASE_PYTHON=${BASE_PYTHON:-python3}
"$BASE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12"'
"$BASE_PYTHON" -m venv .venv
PYTHON=.venv/bin/python
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$PYTHON" -m pip install pyg-lib==0.4.0+pt27cu126 torch-scatter==2.1.2+pt27cu126 torch-sparse==0.6.18+pt27cu126 --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html
"$PYTHON" -m pip install -r requirements.txt
