#!/usr/bin/env bash
set -euo pipefail

uv run --extra docs quartodoc build --config nbs/_quarto.yml
uv run --extra docs quarto render nbs --no-execute
