#!/bin/zsh
TASK_REPO_ROOT="${0:A:h:h:h}"
cd "$TASK_REPO_ROOT" || exit 1
"$TASK_REPO_ROOT/venv/bin/python" "$TASK_REPO_ROOT/workflows/vast/choose.py"
TASK_RUN_EXIT=$?
if (( TASK_RUN_EXIT != 0 )); then
  read -r '?Press Return to close this window.'
fi
exit "$TASK_RUN_EXIT"
