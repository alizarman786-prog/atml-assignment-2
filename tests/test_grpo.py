import torch
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def test_group_advantages_are_per_group():
    r = torch.tensor([0., 2., 10., 14., 5., 5.])
    g = torch.tensor([0, 0, 1, 1, 2, 2])
    a = group_relative_advantages(r, g)
    assert torch.allclose(a[:4], torch.tensor([-1., 1., -1., 1.]), atol=1e-4)
    assert torch.allclose(a[4:], torch.zeros(2), atol=1e-8)


def _loss(loss_type):
    new = torch.log(torch.full((2, 4), 1.5))
    old = torch.zeros(2, 4)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    return grpo_policy_loss(new, old, torch.tensor([1., 1.]), mask, new, 0.2, 0.0, loss_type, 4)[0].item()


def test_grpo_divides_by_each_length():
    assert abs(_loss("grpo") + 1.2) < 1e-5


def test_dr_grpo_divides_by_constant():
    assert abs(_loss("dr_grpo") + 0.9) < 1e-5


def test_clip_negative_advantage():
    new = torch.log(torch.full((1, 1), 0.5))
    mask = torch.ones(1, 1)
    loss = grpo_policy_loss(new, torch.zeros(1, 1), torch.tensor([-1.]), mask, new, 0.2, 0.0, "grpo")[0]
    assert abs(loss.item() - 0.8) < 1e-5


def test_truncated_masked():
    m = mask_truncated_sequences(torch.ones(2, 3), [True, False])
    assert m[0].sum() == 0 and m[1].sum() == 3
