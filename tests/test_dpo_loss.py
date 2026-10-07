import math, torch
from task1_dpo.dpo import dpo_loss

def t(x): return torch.tensor([float(x)])

def test_hand_computed():
    # policy margin = -10-(-12) = 2, ref margin = -10-(-14) = 4
    # correct logit = 0.1*(2-4) = -0.2, loss = ln(1+e^0.2) = 0.7981
    # buggy version would give 0.1*(2+4) = 0.6, loss = 0.4375
    loss, d = dpo_loss(t(-10), t(-12), t(-10), t(-14), beta=0.1)
    assert abs(loss.item() - math.log(1 + math.exp(0.2))) < 1e-5
    assert d["preference_accuracy"].item() == 0.0

def test_policy_equals_reference():
    # at initialisation the logit must be 0 and the loss ln 2
    loss, d = dpo_loss(t(-10), t(-12), t(-10), t(-12), beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert abs(d["logit_mean"].item()) < 1e-6
