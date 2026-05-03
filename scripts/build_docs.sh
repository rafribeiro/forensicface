#!/usr/bin/env bash
set -euo pipefail

uv run --extra docs quartodoc build --config _quartodoc.yml
uv run --extra docs quarto render nbs
