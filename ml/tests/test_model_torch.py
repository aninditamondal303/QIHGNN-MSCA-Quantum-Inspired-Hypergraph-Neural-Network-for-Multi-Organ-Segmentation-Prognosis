"""PyTorch tests: parameter drift, shapes, gradients, agreement with the NumPy reference.

Skipped automatically when torch is not installed. CI installs the CPU wheel and runs them.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qihgnn import reference as R  # noqa: E402
from qihgnn.config import LABELS, ModelConfig  # noqa: E402
from qihgnn.losses import CompositeObjective, cox_loss, exclusion_loss, horizon_loss, marginal_loss  # noqa: E402
from qihgnn.model import QIHGNNMSCA  # noqa: E402
from qihgnn.model.stages import HypergraphModule, MPOChannelMixer, MultiScaleCrossAttention  # noqa: E402
from qihgnn.param_count import parameter_budget  # noqa: E402

SMALL = ModelConfig(input_shape=(32, 32, 16))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return QIHGNNMSCA(ModelConfig()).eval()


def test_parameter_count_matches_analytical_counter(model):
    assert model.stage_parameters() == parameter_budget(ModelConfig())
    assert sum(p.numel() for p in model.parameters()) == 8_347_071


@pytest.mark.parametrize("flags", [{"use_mpo": False}, {"use_hypergraph": False}, {"use_msca": False},
                                   {"use_mpo": False, "use_hypergraph": False, "use_msca": False}])
def test_ablation_parameter_counts(flags):
    cfg = ModelConfig(**flags)
    assert QIHGNNMSCA(cfg).stage_parameters() == parameter_budget(cfg)


def test_forward_shapes_small_volume():
    m = QIHGNNMSCA(SMALL).eval()
    x = torch.randn(2, 4, 32, 32, 16)
    with torch.no_grad():
        out = m(x, torch.zeros(2, 5))
    assert out["seg_logits"].shape == (2, len(LABELS), 32, 32, 16)
    assert out["log_risk"].shape == (2,)
    assert out["horizon_logits"].shape == (2, 3)
    assert out["log_variance"].shape == (2,)
    assert out["shared"].shape == (2, 512)
    assert out["incidence"].shape == (2, 256, 5)
    assert torch.all((out["incidence"] >= 0) & (out["incidence"] <= 1))


def test_forward_full_resolution_shapes(model):
    x = torch.randn(1, 4, 128, 128, 64)
    with torch.no_grad():
        out = model(x)
    assert out["seg_logits"].shape == (1, 6, 128, 128, 64)


def test_mpo_dense_matrix_equals_core_contraction():
    torch.manual_seed(1)
    m = MPOChannelMixer((4, 4), (4, 4), bond=3)
    g1, g2 = m.cores[0].detach()[0], m.cores[1].detach()[..., 0]  # (i1,o1,r), (r,i2,o2)
    full = torch.einsum("abr,rcd->acbd", g1, g2).reshape(16, 16)  # rows (i1,i2), cols (o1,o2)
    assert torch.allclose(m.dense_matrix().detach(), full, atol=1e-6)


def test_mpo_preserves_shape():
    m = MPOChannelMixer((16, 16), (16, 16), 32)
    x = torch.randn(1, 256, 4, 4, 2)
    assert m(x).shape == x.shape


def test_hypergraph_and_msca_preserve_shape():
    x = torch.randn(2, 256, 16, 16, 8)
    hg = HypergraphModule(ModelConfig())
    y, h = hg(x)
    assert y.shape == x.shape and h.shape == (2, 256, 5)
    msca = MultiScaleCrossAttention(ModelConfig())
    assert msca(x).shape == x.shape
    assert msca.token_budgets() == [256, 32, 4, 1]


def test_msca_token_budget_independent_of_resolution():
    msca = MultiScaleCrossAttention(ModelConfig())
    for shape in [(8, 8, 4), (16, 16, 8), (32, 32, 16)]:
        assert msca(torch.randn(1, 256, *shape)).shape[2:] == shape


def test_marginal_loss_gives_zero_gradient_to_unsupervised_logit_directions():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (1, 4, 4, 4)) * LABELS.index("liver")
    sup = torch.tensor([[True, False, True, False, False, False]])
    loss = marginal_loss(logits, target, sup)
    loss.backward()
    probs = torch.softmax(logits.detach(), 1)
    # dL/dp_c is equal for all channels merged into background, so moving mass between
    # unsupervised channels is a zero-gradient direction: check with autograd on probabilities.
    p = probs.clone().requires_grad_(True)
    from qihgnn.losses import merge_unsupervised
    merged = merge_unsupervised(p, sup)
    g = torch.nn.functional.one_hot(target.view(1, -1), 6).permute(0, 2, 1).float()
    inter = (merged.view(1, 6, -1) * g).sum(-1)
    dice = (2 * inter + 1e-5) / (merged.view(1, 6, -1).sum(-1) + g.sum(-1) + 1e-5)
    (1 - dice[:, [0, 2]].mean()).backward()
    for c in (1, 3, 4, 5):
        assert torch.allclose(p.grad[:, c], p.grad[:, 0])


def test_torch_losses_match_numpy_reference():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(1, 6, 3, 3, 3))
    y = rng.choice([0, 2], size=(1, 3, 3, 3))
    sup = np.array([[True, False, True, False, False, False]])
    t = marginal_loss(torch.tensor(logits), torch.tensor(y), torch.tensor(sup)).item()
    ref = R.marginal_loss(R.softmax(logits[0].reshape(6, -1), 0), y.ravel(), sup[0])
    assert abs(t - ref) < 1e-5
    eta, tt, e = rng.normal(size=8), rng.integers(10, 900, 8).astype(float), rng.integers(0, 2, 8)
    e[0] = 1
    assert abs(cox_loss(torch.tensor(eta), torch.tensor(tt), torch.tensor(e)).item() - R.cox_loss(eta, tt, e)) < 1e-5
    z = rng.normal(size=(8, 3))
    hz = [365.25, 1095.75, 1826.25]
    assert abs(horizon_loss(torch.tensor(z), torch.tensor(tt * 3), torch.tensor(e), hz).item() - R.horizon_loss(z, tt * 3, e, hz)) < 1e-5


def test_cox_torch_shift_and_censoring():
    eta = torch.randn(6)
    t = torch.tensor([5.0, 3, 8, 1, 9, 2])
    e = torch.tensor([1, 0, 1, 1, 0, 1])
    assert torch.allclose(cox_loss(eta + 37.5, t, e), cox_loss(eta, t, e), atol=1e-5)
    assert cox_loss(eta, t, torch.zeros(6)).item() == 0.0


def test_exclusion_zero_under_softmax_positive_under_sigmoid():
    logits = torch.randn(2, 6, 4, 4, 4) * 4
    assert exclusion_loss(logits).item() == 0.0
    assert exclusion_loss(logits, "sigmoid").item() > 0.0


def test_one_training_step_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    m = QIHGNNMSCA(SMALL)
    obj = CompositeObjective([365.25, 1095.75, 1826.25])
    opt = torch.optim.AdamW(list(m.parameters()) + list(obj.parameters()), lr=1e-3)
    x = torch.randn(2, 4, 32, 32, 16)
    label = torch.zeros(2, 32, 32, 16, dtype=torch.long)
    label[:, 8:20, 8:20, 4:12] = LABELS.index("liver")
    batch = {"label": label, "supervised": torch.tensor([[True, False, True, False, False, False]] * 2),
             "has_survival": torch.tensor([True, True]), "event": torch.tensor([1, 1]),
             "time": torch.tensor([300.0, 900.0]), "covariates": torch.zeros(2, 5)}
    losses = []
    for _ in range(8):
        out = m(x, batch["covariates"])
        terms = obj(out, batch)
        opt.zero_grad()
        terms["marginal"].backward()
        opt.step()
        losses.append(terms["marginal"].item())
    assert losses[-1] < losses[0]
