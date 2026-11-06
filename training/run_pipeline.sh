#!/usr/bin/env bash
# Full training run, in order:
#   1. Chess.com GM games (main training)
#   2. Lichess broadcast games (fine-tuning, lower learning rate)
#   3. ONNX export
# Every step is resumable: re-run this script after an interruption.
set -euo pipefail
cd "$(dirname "$0")"
W=${CORVID_WORK:-/home/user/corvid-work}
CK=$W/checkpoints

echo "== phase 1: Chess.com"
python train.py --data "$W/data" --ckpt "$CK/corvid.pt" --positions 25e6 --save-every 120

echo "== preparing Lichess broadcast data"
if [ ! -f "$W/data_broadcast/.done" ]; then
  rm -rf "$W/data_broadcast"
  python prepare_broadcast.py "$W"/broadcast/*.pgn.zst --out "$W/data_broadcast"
  touch "$W/data_broadcast/.done"
fi

echo "== phase 2: Lichess broadcast fine-tune"
python train.py --data "$W/data_broadcast" --ckpt "$CK/corvid_broadcast.pt" \
  --init-from "$CK/corvid.pt" --extra-val "$W/data" --epochs 2 --lr 3e-4 --warmup 300 --eval-every 1000 --save-every 120

echo "== export"
python export_onnx.py --ckpt "$CK/corvid_broadcast.pt" --out "$W/corvid.onnx"
echo "== pipeline finished"
