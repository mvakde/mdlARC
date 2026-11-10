#!/bin/zsh
set -eu
cd -- "${0:A:h:h}"
exec venv/bin/python scripts/colab_monitor.py --open
