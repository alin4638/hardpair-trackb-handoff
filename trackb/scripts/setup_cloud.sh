#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Linux Cloud, Python 3.11/3.12. CPU torch is for tiny random-model resume tests only.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cloud.txt
python scripts/validate_cpu.py
