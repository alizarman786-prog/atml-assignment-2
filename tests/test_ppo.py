import torch
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards

def T(*x): return torch.tensor([list(map(float, x))])

def test_clip_positive_advantage():
    # ratio 1.5, A=+1, eps=0.2 -> min(1.5, 1.2) = 1.2 -> loss -1.2 (the maximum bug gives -1.5)
    loss, _, cf = ppo_policy_loss(torch.log(T(1.5)), torch.zeros(1, 1), T(1.0), T(1.0), eps=0.2)
    assert abs(loss.item() + 1.2) < 1e-5 and cf.item() == 1.0

def test_clip_negative_advantage():
    # ratio 0.5, A=-1 -> min(-0.5, -0.8) = -0.8 -> loss +0.8 (the maximum bug gives +0.5)
    loss, _, _ = ppo_policy_loss(torch.log(T(0.5)), torch.zeros(1, 1), T(-1.0), T(1.0), eps=0.2)
    assert abs(loss.item() - 0.8) < 1e-5

def test_mask_excludes_tokens():
    loss, _, _ = ppo_policy_loss(torch.zeros(1, 2), torch.zeros(1, 2), T(1.0, 99.0), T(1.0, 0.0), eps=0.2)
    assert abs(loss.item() + 1.0) < 1e-6

def test_gae_hand_computed():
    adv, ret = compute_gae(T(0, 1), T(0.2, 0.5), T(1, 1), gamma=1.0, lam=1.0)
    assert torch.allclose(adv, T(0.8, 0.5), atol=1e-6) and torch.allclose(ret, T(1.0, 1.0), atol=1e-6)

def test_shaped_rewards():
    r = shaped_rewards(torch.tensor([2.0]), T(-1, -2), T(-1.5, -1.0), T(1, 1), beta_kl=0.1)
    assert torch.allclose(r, T(-0.05, 2.1), atol=1e-6)
