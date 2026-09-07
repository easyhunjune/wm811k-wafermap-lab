import torch

from wm811k.model import WaferCNN


def test_model_output_shape():
    model = WaferCNN(input_channels=2, num_classes=9)
    output = model(torch.zeros(3, 2, 64, 64))
    assert output.shape == (3, 9)
