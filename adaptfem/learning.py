"""Phase 3: learned monitoring policy (supervised, trained from the simulations themselves).

The exact heuristic monitors elastic-anchored quiet elements with the trial yield check
and skips elements whose margin exceeds kappa x (max stress change per step) x age. The
learned policy replaces that skip rule by a predicted *horizon* (number of steps before
the element yields), trained on reference histories with labels computed a posteriori
(the solver is its own teacher). Deployment: an element monitored at step k with
predicted horizon h is not monitored again before step k + floor(safety * h).

Both policies see the same per-element information at monitoring time; the learned one
additionally uses hardening state, neighbourhood and load-direction features.
"""
import numpy as np
from .element import NGP
from .mesh import element_adjacency

HORIZON_CAP = 16
FEATURES = ["f", "df1", "df2", "alpha", "beta", "nb_plastic", "nb_fmax", "dlam_sign",
            "dlam_ratio", "rate", "h_heur", "since_plastic"]


def nb_mean(ptr, idx, v):
    deg = np.maximum(np.diff(ptr), 1)
    return np.add.reduceat(v[idx], ptr[:-1]) / deg


def nb_max(ptr, idx, v):
    return np.maximum.reduceat(v[idx], ptr[:-1])


def yield_margin_history(hist, mat, nel):
    """f_k per element (max over GPs), normalised by sig_y0. Uses equilibrium stress."""
    T = hist["stress"].shape[0]
    f = np.empty((T, nel))
    for k in range(T):
        f[k] = mat.yield_function(hist["stress"][k], hist["alpha"][k], hist["beta"][k]).reshape(nel, NGP).max(1)
    return f


def extract_dataset(hist, case, elems, mat, cap=HORIZON_CAP):
    """Samples: (element, step k) for elements with no plastic GP at step k, k >= 2.
    Label: number of steps until the element first has a plastic GP (capped)."""
    nel = elems.shape[0]
    T = hist["stress"].shape[0]
    ptr, idx = element_adjacency(elems)
    pl = hist["gp_plastic"] > 0                                  # (T,nel)
    f = yield_margin_history(hist, mat, nel)
    alpha = hist["alpha"].reshape(T, nel, NGP).max(2)
    beta = np.linalg.norm(hist["beta"], axis=2).reshape(T, nel, NGP).max(2) / mat.sig_y0
    lam = np.asarray(hist["lam"]); dlam = np.diff(lam, prepend=0.0)
    dsig = np.diff(hist["stress"][:, :, [0, 1, 3]], axis=0, prepend=hist["stress"][:1, :, [0, 1, 3]] * 0)
    rate = np.sqrt((dsig ** 2 * np.array([1, 1, 2])).sum(2)).max(1) / mat.sig_y0      # (T,)
    # steps-to-yield: for each (k, e): min j>k with pl[j,e], capped
    stp = np.full((T, nel), cap, dtype=int)
    nxt = np.full(nel, 10 ** 6)
    for k in range(T - 1, -1, -1):
        nxt = np.where(pl[k], k, nxt)
        if k >= 1:
            stp[k - 1] = np.minimum(nxt - (k - 1), cap)
    # last plastic step before k
    last_pl = np.full(nel, -10 ** 6); since = np.zeros((T, nel), int)
    for k in range(T):
        since[k] = np.minimum(k - last_pl, cap)
        last_pl = np.where(pl[k], k, last_pl)
    X, y, meta = [], [], []
    for k in range(2, T - 1):
        el = np.nonzero(~pl[k])[0]
        if el.size == 0:
            continue
        nbp = nb_mean(ptr, idx, pl[k].astype(float))[el]
        nbf = nb_max(ptr, idx, f[k])[el]
        margin = -f[k, el]
        h_heur = np.where(rate[k] > 0, margin / max(rate[k], 1e-12), cap)
        Xk = np.column_stack([f[k, el], f[k, el] - f[k - 1, el], f[k - 1, el] - f[k - 2, el],
                              alpha[k, el], beta[k, el], nbp, nbf, np.sign(dlam[k]) * np.ones(el.size),
                              (dlam[k] / dlam[k - 1] if dlam[k - 1] != 0 else 0.0) * np.ones(el.size),
                              rate[k] * np.ones(el.size), np.minimum(h_heur, cap), since[k, el]])
        X.append(Xk); y.append(stp[k, el]); meta.append(np.column_stack([np.full(el.size, k), el]))
    return np.vstack(X), np.concatenate(y), np.vstack(meta)


class HorizonModel:
    """Wrapper: predicts the number of steps before yielding (regression on log horizon)."""

    def __init__(self, kind="gbm", seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.kind = kind
        if kind == "gbm":
            self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                   random_state=seed, loss="absolute_error")
        elif kind == "linear":
            self.m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            raise ValueError(kind)

    def fit(self, X, y):
        self.m.fit(X, np.log1p(y))
        return self

    def predict(self, X):
        return np.clip(np.expm1(self.m.predict(X)), 0.0, HORIZON_CAP)


class OracleHorizon:
    """Upper bound: uses the true steps-to-yield from the reference history."""

    def __init__(self, hist, elems, mat):
        nel = elems.shape[0]; T = hist["stress"].shape[0]
        pl = hist["gp_plastic"] > 0
        self.stp = np.full((T, nel), HORIZON_CAP, dtype=int)
        nxt = np.full(nel, 10 ** 6)
        for k in range(T - 1, -1, -1):
            nxt = np.where(pl[k], k, nxt)
            if k >= 1:
                self.stp[k - 1] = np.minimum(nxt - (k - 1), HORIZON_CAP)

    def horizon(self, k, el):
        return self.stp[k, el].astype(float)


def online_features(orch, model, state_prev, el, f_now, k):
    """Features for elements `el` monitored at step k (fresh margin f_now), from
    orchestrator memory only (no Gauss-point work beyond the monitoring itself)."""
    cfg = orch.cfg
    nel = model.nel
    pl_el = state_prev.plastic.reshape(nel, NGP).any(1)
    f_prev = orch.f_hist[el]                     # (n,2): last two measured margins (aligned by age)
    age = np.maximum(orch.margin_age[el], 1)
    df1 = (f_now - f_prev[:, 0]) / age
    df2 = (f_prev[:, 0] - f_prev[:, 1]) / np.maximum(orch.age_hist[el], 1)
    alpha = state_prev.alpha.reshape(nel, NGP).max(1)[el]
    beta = np.linalg.norm(state_prev.beta, axis=1).reshape(nel, NGP).max(1)[el] / model.mat.sig_y0
    nbp = nb_mean(orch.adj_ptr, orch.adj_idx, pl_el.astype(float))[el]
    nbf = nb_max(orch.adj_ptr, orch.adj_idx, orch.f_last)[el]
    lam = model.lam_hist
    dl = lam[k] - lam[k - 1] if k >= 1 else lam[k]
    dl0 = lam[k - 1] - lam[k - 2] if k >= 2 else dl
    rate = max(orch.rate, 1e-12)
    h_heur = np.minimum(-f_now / rate, HORIZON_CAP)
    since = np.minimum(k - orch.last_plastic_step[el], HORIZON_CAP)
    return np.column_stack([f_now, df1, df2, alpha, beta, nbp, nbf, np.sign(dl) * np.ones(el.size),
                            (dl / dl0 if dl0 != 0 else 0.0) * np.ones(el.size), rate * np.ones(el.size),
                            h_heur, since])
