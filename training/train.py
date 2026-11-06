"""Train CorvidNet to predict grandmaster moves (policy) and game results (value).

Resumable: re-running with the same --ckpt continues from the last checkpoint.
"""
import argparse, math, os, time

import numpy as np
import torch
import torch.nn.functional as F

from model import CorvidNet, planes_from_packed


def load(d, split):
    b = np.fromfile(os.path.join(d, f"{split}_board.u8"), np.uint8).reshape(-1, 64)
    f = np.fromfile(os.path.join(d, f"{split}_flags.u8"), np.uint8).reshape(-1, 2)
    m = np.fromfile(os.path.join(d, f"{split}_move.u16"), np.uint16).astype(np.int64)
    r = np.fromfile(os.path.join(d, f"{split}_result.i8"), np.int8).astype(np.float32)
    n = min(len(b), len(f), len(m), len(r))
    return (torch.from_numpy(b[:n]), torch.from_numpy(f[:n]), torch.from_numpy(m[:n]), torch.from_numpy(r[:n]))


@torch.no_grad()
def evaluate(model, val, n=20000, bs=1024):
    model.eval()
    b, f, m, r = val
    n = min(n, len(m))
    correct = top5 = 0
    ploss = vloss = 0.0
    for i in range(0, n, bs):
        p, v = model(planes_from_packed(b[i:i + bs], f[i:i + bs]))
        y = m[i:i + bs]
        # only moves of own pieces to non-own squares can be legal
        own = (b[i:i + bs] >= 1) & (b[i:i + bs] <= 6)
        mask = own.unsqueeze(2) & ~own.unsqueeze(1)          # from x to
        p = p.masked_fill(~mask.flatten(1), -1e9)
        ploss += F.cross_entropy(p, y, reduction="sum").item()
        vloss += F.mse_loss(v, r[i:i + bs], reduction="sum").item()
        correct += (p.argmax(1) == y).sum().item()
        top5 += (p.topk(5, 1).indices == y.unsqueeze(1)).any(1).sum().item()
    model.train()
    return dict(acc=correct / n, top5=top5 / n, ploss=ploss / n, vloss=vloss / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/corvid.pt")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--positions", type=float, default=25e6, help="total training positions")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=300, help="seconds")
    ap.add_argument("--eval-every", type=int, default=2000, help="steps")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)

    train, val = load(a.data, "train"), load(a.data, "val")
    N = len(train[2])
    total_steps = int(a.positions // a.bs)
    print(f"train positions {N:,}  val {len(val[2]):,}  steps {total_steps:,}", flush=True)

    model = CorvidNet(a.blocks, a.channels)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    step, perm, pos = 0, None, 0
    if os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, perm, pos = ck["step"], ck["perm"], ck["pos"]
        print(f"resumed at step {step:,}", flush=True)
    g = torch.Generator().manual_seed(step)
    if perm is None:
        perm = torch.randperm(N, generator=g)

    def lr_at(s):
        warm = 1000
        if s < warm:
            return a.lr * s / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))

    def save():
        tmp = a.ckpt + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, perm=perm, pos=pos,
                        blocks=a.blocks, channels=a.channels), tmp)
        os.replace(tmp, a.ckpt)

    t0 = last_save = time.time()
    run_loss = run_acc = 0.0
    while step < total_steps:
        if pos + a.bs > N:
            perm, pos = torch.randperm(N, generator=g), 0
        idx = perm[pos:pos + a.bs]; pos += a.bs
        b, f, m, r = (t[idx] for t in train)
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        p, v = model(planes_from_packed(b, f))
        lp = F.cross_entropy(p, m)
        loss = lp + a.value_weight * F.mse_loss(v, r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        run_loss = 0.98 * run_loss + 0.02 * lp.item() if step > 1 else lp.item()
        run_acc = 0.98 * run_acc + 0.02 * (p.argmax(1) == m).float().mean().item()
        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step:,}/{total_steps:,}  loss {run_loss:.3f}  acc {run_acc:.3f}  "
                  f"lr {lr_at(step):.2e}  {el/3600:.2f}h", flush=True)
        if step % a.eval_every == 0 or step == total_steps:
            print("  val", {k: round(v, 4) for k, v in evaluate(model, val).items()}, flush=True)
        if time.time() - last_save > a.save_every:
            save(); last_save = time.time()
    save()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
