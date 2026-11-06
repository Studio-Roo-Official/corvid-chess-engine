#!/usr/bin/env bash
# Start the training pipeline unless it is already running. Safe to run any time.
cd "$(dirname "$0")"
W=${CORVID_WORK:-/home/user/corvid-work}
if pgrep -f run_pipeline.sh >/dev/null; then
  echo "already running"
elif grep -q "pipeline finished" "$W/pipeline.log" 2>/dev/null; then
  echo "pipeline already finished"
else
  nohup bash -c ". /home/user/rookery-train/.venv/bin/activate; ./run_pipeline.sh" >> "$W/pipeline.log" 2>&1 &
  echo "restarted"
fi
