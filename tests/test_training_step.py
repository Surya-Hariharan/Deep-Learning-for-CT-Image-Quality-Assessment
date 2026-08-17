import copy

import torch

from ct_iqa.models.ohashi_resnet50 import INPUT_SIZE, OhashiResNet50
from ct_iqa.training.trainer import train_one_step


def test_one_training_step_changes_weights_and_produces_finite_loss():
    torch.manual_seed(0)
    model = OhashiResNet50(dropout_p=0.5, in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    before = copy.deepcopy(model.fc.weight.data)

    images = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
    raw_scores = torch.tensor([0.0, 1.0, 2.0, 4.0])
    loss = train_one_step(model, (images, raw_scores), optimizer, criterion, device="cpu")

    assert loss == loss  # not NaN
    assert loss >= 0.0
    after = model.fc.weight.data
    assert not torch.equal(before, after)
