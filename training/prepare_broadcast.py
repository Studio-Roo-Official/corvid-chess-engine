"""Turn Lichess broadcast PGN dumps (over-the-board tournament games) into
training positions in the same format as prepare_data.py.

A position is kept when the player to move is rated >= --min-elo (FIDE rating
from the PGN headers). Both sides of a game can contribute. About 1% of games
(by hash of their headers) go to the validation split. Games are de-duplicated,
since the same game can be broadcast more than once.

Usage: python prepare_broadcast.py broadcast/*.pgn.zst --out data_broadcast
"""
import argparse, hashlib, io, os, sys, time
from multiprocessing import Pool

import chess
import numpy as np
import zstandard

from prepare_data import encode, movetext

KEYS = ("board.u8", "flags.u8", "move.u16", "result.i8")


def games(path):
    """Yield raw PGN text of each game in a .pgn.zst file."""
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
        buf, in_moves = [], False
        for line in text:
            if line.startswith("[Event ") and in_moves:
                yield "".join(buf)
                buf, in_moves = [], False
            if line.strip() and not line.startswith("["):
                in_moves = True
            buf.append(line)
        if buf:
            yield "".join(buf)


def headers(pgn):
    h = {}
    for line in pgn.splitlines():
        if not line.startswith("["):
            if line.strip():
                break
            continue
        k, _, v = line[1:-1].partition(" ")
        h[k] = v.strip('"')
    return h


def elo(h, k):
    v = h.get(k, "")
    return int(v) if v.isdigit() else 0


def process(job):
    pgn, white_ok, black_ok, result = job
    try:
        if "[FEN " in pgn or ("[Variant " in pgn and '[Variant "Standard"]' not in pgn):
            return None
        board = chess.Board()
        boards, flags, moves, res = [], [], [], []
        for t in movetext(pgn):
            mv = board.parse_san(t)
            w = board.turn == chess.WHITE
            if (white_ok if w else black_ok):
                a, c, ep = encode(board, w)
                f, to = (mv.from_square, mv.to_square) if w else (mv.from_square ^ 56, mv.to_square ^ 56)
                boards.append(a); flags.append((c, ep)); moves.append(f * 64 + to)
                res.append(result if w else -result)
            board.push(mv)
    except Exception:
        return None
    if not moves:
        return None
    return (np.stack(boards), np.array(flags, np.uint8), np.array(moves, np.uint16), np.array(res, np.int8))


def jobs(paths, a, stats, splits):
    seen = set()
    for p in paths:
        for pgn in games(p):
            stats["games"] += 1
            h = headers(pgn)
            r = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(h.get("Result"))
            if r is None:
                continue
            wo, bo = elo(h, "WhiteElo") >= a.min_elo, elo(h, "BlackElo") >= a.min_elo
            if not (wo or bo):
                continue
            body = pgn[pgn.find("\n\n"):]
            key = hashlib.md5((h.get("White", "") + h.get("Black", "") + h.get("Date", "")
                               + " ".join(movetext(body)[:40])).encode()).digest()
            if key in seen:
                stats["dupes"] += 1
                continue
            seen.add(key)
            stats["kept"] += 1
            splits.append("val" if key[0] < 3 else "train")
            yield pgn, wo, bo, r
        print(f"  read {os.path.basename(p)}  {stats}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data_broadcast")
    ap.add_argument("--min-elo", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {s: {k: open(os.path.join(a.out, f"{s}_{k}"), "wb") for k in KEYS} for s in ("train", "val")}
    stats = {"games": 0, "kept": 0, "dupes": 0, "bad": 0}
    counts = {"train": 0, "val": 0}
    splits = []
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap(process, jobs(sorted(a.files), a, stats, splits), chunksize=64)):
            split = splits[i]; splits[i] = None
            if res is None:
                stats["bad"] += 1
                continue
            for arr, k in zip(res, KEYS):
                out[split][k].write(arr.tobytes())
            counts[split] += len(res[2])
    for s in out.values():
        for f in s.values():
            f.close()
    print("done", stats, counts, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
