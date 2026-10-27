"""Every property the report claims, checked on the NumPy reference implementation."""
import numpy as np

from qihgnn import reference as R
from qihgnn.config import LABELS, REPORT_PARAMETER_BUDGET, REPORT_TOTAL_PARAMETERS, ModelConfig
from qihgnn.param_count import attention_cost, mpo_compression_table, parameter_budget, total_parameters
from qihgnn.verify import build_report, shift_to_channel, synthetic_case, verify_marginal, verify_unlabelled_prediction


def test_parameter_budget_matches_report_per_stage():
    assert parameter_budget() == REPORT_PARAMETER_BUDGET


def test_parameter_total_and_memory_match_report():
    total = total_parameters()
    assert total == REPORT_TOTAL_PARAMETERS == 8_347_071
    assert round(total * 4 / 2 ** 20, 2) == 31.84


def test_ablations_remove_exactly_their_stage():
    full = parameter_budget()
    for flag, stage in [("use_mpo", "mpo"), ("use_hypergraph", "hypergraph"), ("use_msca", "msca")]:
        b = parameter_budget(ModelConfig(**{flag: False}))
        assert b[stage] == 0
        assert {k: v for k, v in b.items() if k != stage} == {k: v for k, v in full.items() if k != stage}


def test_mpo_compression_table_matches_report():
    expected = {2: 64.0, 4: 32.0, 8: 16.0, 16: 8.0, 32: 4.0, 48: 65536 / 24576, 64: 2.0, 128: 1.0}
    for row in mpo_compression_table():
        assert abs(row["compression"] - expected[row["bond"]]) < 1e-12
    assert [r["parameters"] for r in mpo_compression_table() if r["bond"] == 32] == [16_384]


def test_attention_cost_matches_report_and_is_resolution_independent():
    c = attention_cost()
    assert c["multi_scale"] == 66_577
    assert c["naive"] == 1_099_511_627_776
    assert abs(c["reduction"] - 1.651e7) / 1.651e7 < 1e-3
    for shape in [(64, 64, 32), (256, 256, 128)]:
        assert attention_cost(shape=shape)["multi_scale"] == 66_577


def test_marginal_loss_is_invariant_and_naive_loss_is_not():
    v = verify_marginal()
    assert v["max_marginal_change"] < 1e-12
    assert v["min_naive_change"] > 1e-3
    assert v["passed"]


def test_marginal_loss_numerical_gradient_is_zero_for_unsupervised_channels():
    rng = np.random.default_rng(0)
    p = R.softmax(rng.normal(size=(len(LABELS), 64)), axis=0)
    y = rng.choice([0, 2], size=64)
    sup = np.array([True, False, True, False, False, False])
    base = R.marginal_loss(p, y, sup)
    # Moving mass between two unsupervised channels is the only perturbation that keeps p a
    # distribution while touching unsupervised channels alone; the loss must not change.
    q = p.copy()
    delta = 0.5 * q[3]
    q[3] -= delta
    q[5] += delta
    assert abs(R.marginal_loss(q, y, sup) - base) < 1e-12


def test_cox_shift_invariance_and_censoring():
    eta = np.array([0.3, -1.2, 2.0, 0.1, 0.7])
    t = np.array([100, 300, 200, 300, 50.0])
    e = np.array([1, 0, 1, 1, 1])
    assert abs(R.cox_loss(eta + 37.5, t, e) - R.cox_loss(eta, t, e)) < 1e-9
    assert R.cox_loss(eta, t, np.zeros(5)) == 0.0


def test_cox_matches_hand_computation():
    eta = np.array([1.0, 0.0])
    t = np.array([1.0, 2.0])
    e = np.array([1, 1])
    # i=0: risk set {0,1}; i=1: risk set {1}
    expected = -((1.0 - np.log(np.e + 1.0)) + (0.0 - 0.0)) / 2
    assert abs(R.cox_loss(eta, t, e) - expected) < 1e-12


def test_horizon_labels_skip_undetermined_pairs():
    y, m = R.horizon_labels([730.5, 400.0, 2000.0], [0, 1, 1], [365.25, 1095.75, 1826.25])
    assert m.tolist() == [[True, False, False], [True, True, True], [True, True, True]]
    assert y.tolist() == [[1, 0, 0], [1, 0, 0], [1, 1, 1]]


def test_uncertainty_weighting_optimum_is_log_loss():
    for L in [0.1, 1.0, 3.7]:
        s = np.linspace(np.log(L) - 1, np.log(L) + 1, 2001)
        vals = [R.uncertainty_weighted([L], [si]) for si in s]
        assert abs(s[int(np.argmin(vals))] - np.log(L)) < 2e-3


def test_dice_closed_forms():
    a = np.zeros(100, bool); a[:40] = True
    b = np.zeros(100, bool); b[20:60] = True
    assert R.dice_coefficient(a, a) == 1.0
    assert R.dice_coefficient(a, ~a) == 0.0
    assert abs(R.dice_coefficient(a, b) - 0.5) < 1e-12
    assert R.dice_coefficient(np.zeros(5, bool), np.zeros(5, bool)) == 1.0


def test_hd95_on_shifted_cube():
    a = np.zeros((20, 20, 20), bool); a[5:10, 5:10, 5:10] = True
    b = np.roll(a, 2, axis=0)
    assert R.hd95(a, a) == 0.0
    assert abs(R.hd95(a, b) - 2.0) < 1e-9
    assert np.isnan(R.hd95(a, np.zeros_like(a)))


def test_concordance_and_td_auc():
    t = np.array([100, 200, 300, 400.0]); e = np.ones(4)
    assert R.concordance_index(-t, t, e) == 1.0
    assert R.concordance_index(np.zeros(4), t, e) == 0.5
    assert R.time_dependent_auc(-t, t, e, 250.0) == 1.0


def test_exclusion_term_is_zero_under_softmax():
    p = R.softmax(np.random.default_rng(1).normal(0, 4, (6, 100)), axis=0)
    assert R.exclusion_loss(p) == 0.0


def test_full_report_passes():
    rep = build_report()
    assert rep["all_passed"], rep["checks"]


def test_naive_loss_punishes_correct_unlabelled_predictions_marginal_does_not():
    v = verify_unlabelled_prediction()
    assert v["passed"]
    p, y, sup = synthetic_case()
    prev = R.naive_loss(p, y)
    for t in (0.2, 0.4, 0.6, 0.8):
        q = shift_to_channel(p, y, LABELS.index("kidney"), t)
        assert abs(R.marginal_loss(q, y, sup) - R.marginal_loss(p, y, sup)) < 1e-12
        cur = R.naive_loss(q, y)
        assert cur > prev  # strictly increasing: every step toward the right answer is penalised
        prev = cur
