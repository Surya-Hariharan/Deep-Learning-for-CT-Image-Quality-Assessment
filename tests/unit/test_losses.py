"""Unit tests for the MSE, ranking and Gaussian-NLL losses. All inputs are SYNTHETIC tensors."""

from __future__ import annotations

import math

import pytest
import torch

from ct_iqa.data.sampling import ordered_pair_mask
from ct_iqa.losses.gaussian_nll import GaussianNLLLoss
from ct_iqa.losses.mse import MSELoss
from ct_iqa.losses.ranking import MSERankingLoss, pairwise_margin_ranking
from ct_iqa.models.heads import LOGVAR_MAX, LOGVAR_MIN
from ct_iqa.models.registry import ModelOutput


def test_pair_mask_direction_ties_and_near_ties():
    y = torch.tensor([0.0, 1.0, 1.0, 1.1, 3.0])
    m = ordered_pair_mask(y, min_score_diff=0.25)
    assert not m.diagonal().any()
    assert m[4, 0] and m[1, 0] and m[3, 0] and not m[0, 4]     # higher-first only
    assert not m[1, 2] and not m[2, 1]                          # exact tie: no constraint
    assert not m[3, 1] and not m[3, 2]                          # near-tie (0.1 < 0.25): no constraint
    assert not (m & m.T).any()                                  # each unordered pair at most once
    assert int(ordered_pair_mask(y, 0.0).sum()) == int(ordered_pair_mask(y, 1e-9).sum())  # 0 still excludes ties


def test_hinge_hand_computed_values():
    pred = torch.tensor([0.9, 0.5, 0.1])                         # ordered consistently with y
    y = torch.tensor([4.0, 2.0, 0.0])
    loss, n = pairwise_margin_ranking(pred, y, margin=0.3, min_score_diff=0.25)
    # pairs (0,1): 0.4>=0.3 ->0 ; (0,2): 0.8 ->0 ; (1,2): 0.4 ->0
    assert n == 3 and loss.item() == 0.0
    pred_bad = torch.tensor([0.1, 0.5, 0.9])                     # reversed
    loss, n = pairwise_margin_ranking(pred_bad, y, margin=0.3, min_score_diff=0.25)
    # (0,1): 0.3-(-0.4)=0.7 ; (0,2): 0.3-(-0.8)=1.1 ; (1,2): 0.7 -> mean 0.8333
    assert n == 3 and loss.item() == pytest.approx((0.7 + 1.1 + 0.7) / 3, rel=1e-6)


def test_no_valid_pairs_is_finite_zero_with_zero_gradient():
    pred = torch.tensor([0.2, 0.3], requires_grad=True)
    loss, n = pairwise_margin_ranking(pred, torch.tensor([1.0, 1.0]), 0.1, 0.25)
    loss.backward()
    assert n == 0 and loss.item() == 0.0 and torch.equal(pred.grad, torch.zeros(2))


def test_ranking_gradients_finite_and_push_in_correct_direction():
    pred = torch.tensor([0.2, 0.8], requires_grad=True)          # y_0 > y_1 but f_0 < f_1: violated
    loss, _ = pairwise_margin_ranking(pred, torch.tensor([3.0, 1.0]), margin=0.1, min_score_diff=0.25)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0] < 0 and pred.grad[1] > 0                 # descent raises f_0 and lowers f_1


def test_lambda_zero_equals_mse_value_and_gradient():
    torch.manual_seed(0)
    mean = torch.rand(16, requires_grad=True)
    raw = torch.rand(16) * 4
    target = raw / 4
    a = MSELoss()(ModelOutput(mean), target, raw)
    ga, = torch.autograd.grad(a.total, mean)
    b = MSERankingLoss(lambda_rank=0.0, margin=0.1, min_score_diff=0.25)(ModelOutput(mean), target, raw)
    gb, = torch.autograd.grad(b.total, mean)
    assert torch.equal(a.total, b.total) and torch.equal(ga, gb)


def test_ranking_total_and_diagnostics():
    mean = torch.tensor([0.1, 0.9])
    raw = torch.tensor([4.0, 0.0])
    r = MSERankingLoss(lambda_rank=2.0, margin=0.1, min_score_diff=0.25)(ModelOutput(mean), raw / 4, raw)
    mse = ((0.1 - 1.0) ** 2 + (0.9 - 0.0) ** 2) / 2
    hinge = 0.1 - (0.1 - 0.9)
    assert r.total.item() == pytest.approx(mse + 2.0 * hinge, rel=1e-6)
    assert r.parts["n_valid_pairs"] == 1 and r.parts["n_possible_pairs"] == 1


def test_gaussian_nll_matches_closed_form_and_torch_reference():
    torch.manual_seed(1)
    mu, s = torch.rand(10), torch.randn(10)
    y = torch.rand(10)
    ours = GaussianNLLLoss()(ModelOutput(mu, s), y, y * 4).total
    manual = 0.5 * sum(math.exp(-si) * (yi - mi) ** 2 + si for mi, si, yi in zip(mu.tolist(), s.tolist(), y.tolist())) / 10
    assert ours.item() == pytest.approx(manual, rel=1e-5)
    ref = torch.nn.GaussianNLLLoss(full=False, eps=0.0, reduction="mean")(mu, y, torch.exp(s))
    assert ours.item() == pytest.approx(ref.item(), rel=1e-5)


def test_gaussian_nll_finite_at_logvar_bounds_and_gradients():
    for s_val in (LOGVAR_MIN, LOGVAR_MAX):
        mu = torch.full((4,), 0.5, requires_grad=True)
        s = torch.full((4,), s_val, requires_grad=True)
        r = GaussianNLLLoss()(ModelOutput(mu, s), torch.tensor([0.0, 1.0, 0.5, 0.2]), torch.zeros(4))
        r.total.backward()
        assert torch.isfinite(r.total) and torch.isfinite(mu.grad).all() and torch.isfinite(s.grad).all()


def test_gaussian_nll_requires_log_var():
    with pytest.raises(ValueError):
        GaussianNLLLoss()(ModelOutput(torch.rand(2)), torch.rand(2), torch.rand(2))
