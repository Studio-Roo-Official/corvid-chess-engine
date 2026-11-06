"""Turn the Kaggle "GM games (chess.com)" CSV into compact training positions.

Keeps only positions where the grandmaster is to move, in games where the
grandmaster was rated >= --min-elo and the base time was >= --min-base
seconds (default: no bullet). A random --sample fraction of eligible games is
kept, and about 1% of games (chosen by URL hash) go to the validation split.

Output per split, in --out:
  {split}_board.u8   N x 64  (0 empty, 1-6 own P..K, 7-12 opponent P..K;
                               board flipped so the side to move is White)
  {split}_flags.u8   N x 2   (castling bits own K/Q, opp K/Q; ep square+1)
  {split}_move.u16   N       (from*64 + to, in the flipped frame)
  {split}_result.i8  N       (+1 GM won, 0 draw, -1 GM lost)
"""
import argparse, csv, hashlib, os, random, re, sys, time
from multiprocessing import Pool

import chess
import numpy as np

COMMENT = re.compile(r"\{[^}]*\}")
MOVENUM = re.compile(r"^\d+\.+$")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def movetext(pgn):
    i = pgn.find("\n\n")
    body = pgn[i + 2:] if i >= 0 else pgn
    body = COMMENT.sub(" ", body)
    out = []
    for t in body.split():
        if MOVENUM.match(t) or t in RESULTS:
            continue
        t = re.sub(r"^\d+\.+", "", t)
        if t:
            out.append(t)
    return out


def encode(board, gm_white):
    """Board from the GM's point of view (GM always 'White')."""
    arr = np.zeros(64, np.uint8)
    for sq, pc in board.piece_map().items():
        own = pc.color == (chess.WHITE if gm_white else chess.BLACK)
        s = sq if gm_white else sq ^ 56
        arr[s] = pc.piece_type + (0 if own else 6)
    us = chess.WHITE if gm_white else chess.BLACK
    them = not us
    c = (board.has_kingside_castling_rights(us) | board.has_queenside_castling_rights(us) << 1
         | board.has_kingside_castling_rights(them) << 2 | board.has_queenside_castling_rights(them) << 3)
    ep = 0
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = (board.ep_square if gm_white else board.ep_square ^ 56) + 1
    return arr, c, ep


def process(job):
    pgn, gm_white, result = job
    try:
        toks = movetext(pgn)
        board = chess.Board()
        boards, flags, moves = [], [], []
        for t in toks:
            mv = board.parse_san(t)
            if board.turn == (chess.WHITE if gm_white else chess.BLACK):
                a, c, ep = encode(board, gm_white)
                f, to = (mv.from_square, mv.to_square) if gm_white else (mv.from_square ^ 56, mv.to_square ^ 56)
                boards.append(a); flags.append((c, ep)); moves.append(f * 64 + to)
            board.push(mv)
    except Exception:
        return None
    if not moves:
        return None
    n = len(moves)
    return (np.stack(boards), np.array(flags, np.uint8), np.array(moves, np.uint16),
            np.full(n, result, np.int8))


def jobs(path, a, rng, stats):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stats["rows"] += 1
            p = row["player"].lower()
            if row["White"].lower() == p:
                gm_white, elo = True, row["WhiteElo"]
            elif row["Black"].lower() == p:
                gm_white, elo = False, row["BlackElo"]
            else:
                continue
            if not elo.isdigit() or int(elo) < a.min_elo:
                continue
            tc = row["TimeControl"]
            if "/" in tc or not tc.split("+")[0].isdigit() or int(tc.split("+")[0]) < a.min_base:
                continue
            res = row["Result"]
            if res == "1-0":
                r = 1 if gm_white else -1
            elif res == "0-1":
                r = -1 if gm_white else 1
            elif res == "1/2-1/2":
                r = 0
            else:
                continue
            stats["eligible"] += 1
            if rng.random() >= a.sample:
                continue
            split = "val" if hashlib.md5(row["url"].encode()).digest()[0] < 3 else "train"
            stats["sent"] += 1
            yield split, (row["pgn"], gm_white, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-elo", type=int, default=2500)
    ap.add_argument("--min-base", type=int, default=180)
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = {s: {k: open(os.path.join(a.out, f"{s}_{k}"), "wb")
                 for k in ("board.u8", "flags.u8", "move.u16", "result.i8")} for s in ("train", "val")}
    counts = {"train": 0, "val": 0}
    stats = {"rows": 0, "eligible": 0, "sent": 0, "bad": 0}
    rng = random.Random(a.seed)
    gen = jobs(a.csv, a, rng, stats)
    splits = []

    def tagged():
        for split, job in gen:
            splits.append(split)
            yield job

    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap(process, tagged(), chunksize=64)):
            split = splits[i]
            splits[i] = None
            if res is None:
                stats["bad"] += 1
                continue
            for arr, k in zip(res, ("board.u8", "flags.u8", "move.u16", "result.i8")):
                files[split][k].write(arr.tobytes())
            counts[split] += len(res[2])
            if i % 20000 == 0:
                print(f"{time.time()-t0:7.0f}s rows={stats['rows']} eligible={stats['eligible']} "
                      f"games={stats['sent']} positions={counts}", flush=True)
            if a.max_positions and counts["train"] >= a.max_positions:
                break
        pool.terminate()
    for s in files.values():
        for f in s.values():
            f.close()
    print("done", stats, counts, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
