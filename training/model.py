"""Corvid policy/value network.

Positions are always seen from the side to move (the board is flipped for
Black), so the network only ever plays "White".

Input planes (18 x 8 x 8):
  0-5   own pawn, knight, bishop, rook, queen, king
  6-11  opponent pawn ... king
  12-15 castling rights: own K-side, own Q-side, opp K-side, opp Q-side
  16    en-passant target square
  17    constant 1 (lets padded convolutions see the board edge)

Outputs:
  policy: 4096 logits, index = from_square * 64 + to_square
          (promotions are always to a queen)
  value:  expected game result for the side to move, in [-1, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PLANES = 18


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class CorvidNet(nn.Module):
    def __init__(self, blocks=6, ch=64):
        super().__init__()
        self.blocks_n, self.ch = blocks, ch
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # policy: per-square "from" features -> 64 "to" logits per square
        self.p_conv = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.p_out = nn.Conv2d(ch, 64, 1)
        # value
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Flatten(), nn.Linear(8 * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.stem(x))
        p = self.p_out(self.p_conv(h))            # B x 64(to) x 8 x 8(from)
        p = p.flatten(2).transpose(1, 2).flatten(1)  # B x (from*64 + to)
        v = self.v_fc(self.v_conv(h)).squeeze(1)
        return p, v


def planes_from_packed(board, flags):
    """board: B x 64 uint8 (0 empty, 1-6 own P..K, 7-12 opp P..K), square a1=0.
    flags: B x 2 uint8 -> [castling bits (own K,own Q,opp K,opp Q), ep square+1 or 0]."""
    B, dev = board.shape[0], board.device
    b = board.long()
    x = torch.zeros(B, 13, 64, device=dev)
    x.scatter_(1, b.unsqueeze(1), 1.0)
    x = x[:, 1:]                                   # drop "empty"
    cast = flags[:, 0].long()
    cp = torch.stack([((cast >> i) & 1).float() for i in range(4)], 1)  # B x 4
    cp = cp.unsqueeze(2).expand(B, 4, 64)
    ep = torch.zeros(B, 1, 65, device=dev)
    ep.scatter_(2, flags[:, 1].long().view(B, 1, 1), 1.0)
    ep = ep[:, :, 1:]
    one = torch.ones(B, 1, 64, device=dev)
    return torch.cat([x, cp, ep, one], 1).view(B, N_PLANES, 8, 8)
