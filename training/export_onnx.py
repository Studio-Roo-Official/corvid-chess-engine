"""Export a trained checkpoint to ONNX for ONNX Runtime Web.

The exported graph takes the 18x8x8 input planes (see model.py) and returns
policy logits (4096) and value (1).
"""
import argparse

import torch

from model import CorvidNet, N_PLANES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/corvid.pt")
    ap.add_argument("--out", default="corvid.onnx")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    m = CorvidNet(ck["blocks"], ck["channels"])
    m.load_state_dict(ck["model"])
    m.eval()
    x = torch.zeros(1, N_PLANES, 8, 8)
    torch.onnx.export(m, (x,), a.out, input_names=["planes"], output_names=["policy", "value"],
                      dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
