"""The Laplace logistic GLMM (robustness check 6) recovers planted effects on simulated data."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from rhinoc.glmm import fit_logit_glmm, wald  # noqa: E402

NAMES = ["(Intercept)", "armR0", "armR1", "pays", "armR0:pays", "armR1:pays"]
TRUE = np.array([-1.0, 0.3, -0.2, 1.2, -0.5, 0.6])


def _simulate(seed, sd_scenario=1.0, sd_run=0.3, n=10):
    rng = np.random.default_rng(seed)
    us, ur = rng.normal(0, sd_scenario, 50), rng.normal(0, sd_run, 6)
    y, ns, X, scen, run = [], [], [], [], []
    for r in range(6):
        arm = r // 2
        for s in range(50):
            for pays in (0, 1):
                x = [1, arm == 0, arm == 1, pays, (arm == 0) * pays, (arm == 1) * pays]
                eta = TRUE @ np.array(x, float) + us[s] + ur[r]
                y.append(rng.binomial(n, 1 / (1 + np.exp(-eta))))
                ns.append(n), X.append(x), scen.append(s), run.append(r)
    return y, ns, np.array(X, float), {"scenario": (scen, None), "run": (run, None)}


def test_recovers_planted_fixed_effects_and_scenario_sd():
    fits = []
    for seed in range(30):  # 30 replicates: the mean's sampling error is ~0.03 on the pays terms
        y, n, X, terms = _simulate(seed)
        fits.append(fit_logit_glmm(y, n, X, NAMES, terms))
    assert all(f["converged"] for f in fits)
    est = np.array([[f["fixed"][k]["estimate"] for k in NAMES] for f in fits])
    se = np.array([[f["fixed"][k]["se"] for k in NAMES] for f in fits])
    # the pays terms are estimated within run, so they are precise; arm main effects are not
    assert np.all(np.abs(est.mean(0) - TRUE)[3:] < 0.08)
    assert np.all(np.abs(se.mean(0)[3:] / est.std(0)[3:] - 1) < 0.5)
    assert np.mean([f["random_sd"]["scenario"] for f in fits]) == pytest.approx(1.0, abs=0.15)


def test_no_random_effect_variance_matches_plain_logistic_regression():
    # one observation per row, no clustering: the scenario SD goes to 0 and the fit is the GLM
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    y = (rng.random(400) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(float)
    fit = fit_logit_glmm(y, np.ones(400), np.column_stack([np.ones(400), x]), ["a", "b"],
                         {"scenario": (np.arange(400) % 40, None)})
    beta = np.zeros(2)
    X = np.column_stack([np.ones(400), x])
    for _ in range(50):  # plain IRLS
        mu = 1 / (1 + np.exp(-X @ beta))
        beta += np.linalg.solve(X.T @ (X * (mu * (1 - mu))[:, None]), X.T @ (y - mu))
    assert fit["random_sd"]["scenario"] < 0.3
    assert [fit["fixed"][k]["estimate"] for k in "ab"] == pytest.approx(beta, abs=0.05)


def test_wald_interval_and_one_sided_p():
    w = wald(1.0, 0.5, 0.90, +1)
    assert w["ci"] == pytest.approx((1 - 1.6449 * 0.5, 1 + 1.6449 * 0.5), abs=1e-3)
    assert w["p_one_sided"] == pytest.approx(0.02275, abs=1e-4)
