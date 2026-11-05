"""Logistic mixed-effects regression with independent Gaussian random effects, fitted by the
Laplace approximation (the objective lme4's `glmer` uses by default, nAGQ = 1).

Robustness check 6 of preregistration/hypotheses.md needs one model,
exploit ~ arm x pays + (1 | scenario) + (1 | run). R and statsmodels are not in this project,
so this is a small, self-contained fitter on numpy and scipy (installed with the pipeline).
The core package stays stdlib-only: import this module only where the model is fitted.

Model, for row i with y_i exploits out of n_i samples:
    logit P(exploit) = X_i beta + sum_k sd_k * u_k[level_k(i)] * m_k(i),   u ~ N(0, 1)
where each random-effect term k has a level index and a multiplier m_k (1 for an intercept,
the covariate for a slope). Fitting follows lme4:
    1. for given sds, maximise the penalised log-likelihood over (beta, u) by Newton steps;
       optimise the sds on the Laplace deviance with beta profiled out (lme4's nAGQ = 0);
    2. starting there, optimise (sds, beta) jointly on the Laplace deviance, with u at its
       conditional mode (nAGQ = 1).
Standard errors of beta come from the penalised Hessian at the optimum, as `vcov` in lme4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import optimize, special, stats

MAX_SD = 10.0


class _Model:
    def __init__(self, y, n, X, terms):
        self.y = np.asarray(y, float)
        self.n = np.asarray(n, float)
        self.X = np.asarray(X, float)
        cols, owner = [], []
        for k, (levels, mult) in enumerate(terms):
            levels = np.asarray(levels)
            mult = np.ones(len(levels)) if mult is None else np.asarray(mult, float)
            for lv in np.unique(levels):
                cols.append(np.where(levels == lv, mult, 0.0))
                owner.append(k)
        self.Z = np.column_stack(cols)
        self.owner = np.array(owner)
        self.n_terms = len(terms)
        self.const = float(np.sum(special.gammaln(self.n + 1) - special.gammaln(self.y + 1)
                                  - special.gammaln(self.n - self.y + 1)))

    def loglik(self, eta):
        return float(np.sum(self.y * eta - self.n * np.logaddexp(0.0, eta))) + self.const

    def _newton(self, sd, beta, u, joint):
        """Conditional mode of u (and beta if `joint`) by damped Newton steps."""
        lam = sd[self.owner]
        ZL = self.Z * lam
        A = np.hstack([self.X, ZL]) if joint else ZL
        p = self.X.shape[1] if joint else 0
        theta = np.concatenate([beta, u]) if joint else u.copy()

        def eta_of(t):
            b, uu = (t[:p], t[p:]) if joint else (beta, t)
            return self.X @ b + ZL @ uu

        def objective(t):
            return self.loglik(eta_of(t)) - 0.5 * float(t[p:] @ t[p:])

        current = objective(theta)
        for _ in range(100):
            eta = eta_of(theta)
            mu = special.expit(eta)
            w = self.n * mu * (1 - mu)
            grad = A.T @ (self.y - self.n * mu)
            grad[p:] -= theta[p:]
            H = A.T @ (A * w[:, None])
            H[p:, p:] += np.eye(len(u))
            step = np.linalg.solve(H, grad)
            t = 1.0
            while True:
                cand = theta + t * step
                value = objective(cand)
                if value >= current - 1e-12:
                    break
                t /= 2
                if t < 1e-8:  # no ascent along the Newton direction: at the mode
                    cand, value = theta, current
                    break
            theta, previous, current = cand, current, value
            if abs(current - previous) < 1e-10 and np.max(np.abs(t * step)) < 1e-8:
                break
        b, uu = (theta[:p], theta[p:]) if joint else (beta, theta)
        return b, uu

    def deviance(self, sd, beta, u):
        """Laplace deviance at (sd, beta) with u at its conditional mode."""
        lam = sd[self.owner]
        eta = self.X @ beta + self.Z @ (lam * u)
        mu = special.expit(eta)
        w = self.n * mu * (1 - mu)
        ZL = self.Z * lam
        Huu = ZL.T @ (ZL * w[:, None]) + np.eye(len(u))
        if not np.all(np.isfinite(Huu)):
            return np.inf
        try:  # Huu = Lambda Z'WZ Lambda + I is symmetric positive definite
            logdet = 2 * float(np.sum(np.log(np.diag(np.linalg.cholesky(Huu)))))
        except np.linalg.LinAlgError:
            return np.inf
        return -2 * (self.loglik(eta) - 0.5 * float(u @ u)) + logdet

    def vcov(self, sd, beta, u):
        lam = sd[self.owner]
        eta = self.X @ beta + self.Z @ (lam * u)
        mu = special.expit(eta)
        w = self.n * mu * (1 - mu)
        A = np.hstack([self.X, self.Z * lam])
        H = A.T @ (A * w[:, None])
        p = self.X.shape[1]
        H[p:, p:] += np.eye(len(u))
        return np.linalg.inv(H)[:p, :p]


def fit_logit_glmm(
    y: Sequence[float], n: Sequence[float], X: np.ndarray, names: Sequence[str],
    terms: Mapping[str, tuple[Sequence, Sequence[float] | None]],
) -> dict:
    """Fit the model on aggregated binomial rows (y successes of n trials). `terms` maps each
    random-effect name to (level of each row, multiplier per row or None for an intercept).
    Returns fixed effects (estimate, se, z), random-effect SDs, the Laplace deviance
    and whether the optimiser converged."""
    m = _Model(y, n, X, list(terms.values()))
    p, q = m.X.shape[1], m.Z.shape[1]
    state = {"beta": np.zeros(p), "u": np.zeros(q)}

    def stage1(sd):
        beta, u = m._newton(np.asarray(sd), state["beta"], state["u"], joint=True)
        state["beta"], state["u"] = beta, u
        return m.deviance(np.asarray(sd), beta, u)

    bounds = [(0.0, MAX_SD)] * m.n_terms
    r1 = optimize.minimize(stage1, np.full(m.n_terms, 0.5), method="L-BFGS-B", bounds=bounds)
    sd1 = r1.x
    stage1(sd1)

    def stage2(params):
        sd, beta = params[:m.n_terms], params[m.n_terms:]
        _, u = m._newton(sd, beta, state["u"], joint=False)
        state["u"] = u
        return m.deviance(sd, beta, u)

    start = np.concatenate([sd1, state["beta"]])
    r2 = optimize.minimize(stage2, start, method="L-BFGS-B", bounds=bounds + [(None, None)] * p)
    sd, beta = r2.x[:m.n_terms], r2.x[m.n_terms:]
    _, u = m._newton(sd, beta, state["u"], joint=False)
    V = m.vcov(sd, beta, u)
    d = np.diag(V)
    se = np.full(len(d), np.nan)  # a negative variance only comes out of a non-identified fit
    se[d > 0] = np.sqrt(d[d > 0])
    dev = float(m.deviance(sd, beta, u))
    return {
        "fixed": {nm: {"estimate": float(b), "se": float(s), "z": float(b / s)} for nm, b, s in zip(names, beta, se)},
        "vcov": V.tolist(),
        "random_sd": {nm: float(s) for nm, s in zip(terms, sd)},
        "deviance_laplace": dev,
        "loglik_laplace": -0.5 * dev,
        "converged": bool(r1.success and r2.success),
        "n_rows": int(len(m.y)),
        "n_samples": int(m.n.sum()),
    }


def wald(estimate: float, se: float, level: float, direction: int = 0) -> dict:
    """Wald interval at `level` and, for direction +1/-1, the one-sided p of that sign."""
    zc = stats.norm.ppf(0.5 + level / 2)
    out = {"ci": (estimate - zc * se, estimate + zc * se), "ci_level": level}
    if direction:
        out["p_one_sided"] = float(stats.norm.sf(direction * estimate / se))
    return out
