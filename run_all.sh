#!/usr/bin/env bash
mkdir -p "$(dirname "$0")/reports"
python -m src.run_all
