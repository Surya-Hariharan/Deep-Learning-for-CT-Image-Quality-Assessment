"""Lightweight smoke test: import -> construct -> synthetic forward pass.

Does not train anything and does not require the PNG dataset. Completes in
seconds on CPU.

    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from quality_aware_lung_ct.ngpnet import NGPNet


def main():
    print("1/5 import quality_aware_lung_ct.ngpnet.NGPNet ... ok")

    model = NGPNet(img_size=64, feat_dim=8)
    print("2/5 construct NGPNet(img_size=64, feat_dim=8) ... ok")

    n_params = sum(p.nelement() for p in model.parameters())
    print(f"3/5 parameter count: {n_params:,}")
    if n_params != 1_984_821:
        raise SystemExit(f"Expected 1,984,821 parameters, got {n_params:,}")

    batch = 1
    im0 = torch.randn(batch, 1, 64, 64, 64)
    im1 = torch.randn(batch, 1, 64, 64, 64)
    tm0 = torch.randint(0, 64, (batch,))
    tm1 = torch.randint(0, 64, (batch,))
    print("4/5 synthetic input constructed ... ok")

    model.eval()
    with torch.no_grad():
        image, mask = model(im0, im1, tm0, tm1)
    print(f"5/5 forward pass ok: image {tuple(image.shape)}, mask {tuple(mask.shape)}")

    print("SMOKE TEST PASSED (CPU)")

    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        with torch.no_grad():
            image, mask = model_cuda(im0.cuda(), im1.cuda(), tm0.cuda(), tm1.cuda())
        print(f"CUDA forward pass ok: image {tuple(image.shape)}, mask {tuple(mask.shape)}")
    else:
        print("CUDA not available; CPU-only run.")


if __name__ == "__main__":
    main()
