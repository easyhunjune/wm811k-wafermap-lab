from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Run scripts/setup.ps1 and install PyTorch before the smoke test.")
        return 2

    from wm811k.model import WaferCNN
    from wm811k.preprocessing import encode_wafer_map

    wafer = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 2, 1],
            [1, 2, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    encoded = encode_wafer_map(wafer, output_size=64, mode="two_channel_masks")
    assert encoded.shape == (2, 64, 64)
    assert set(torch.unique(encoded).tolist()).issubset({0.0, 1.0})
    assert torch.all(encoded[1] <= encoded[0])

    model = WaferCNN(input_channels=2, num_classes=9)
    output = model(encoded.unsqueeze(0))
    assert output.shape == (1, 9)
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
