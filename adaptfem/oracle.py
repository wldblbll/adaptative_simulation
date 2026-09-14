"""Phase 0: a posteriori (oracle) analysis of a reference history.

For every element and step, decide whether the element could have been *extrapolated*
from its last integrated state instead of being integrated, under a greedy policy that
looks at the TRUE trajectory. This is an upper bound on what any online policy can do,
not a gain: online, the extrapolation error feeds back into the equilibrium and drifts
the solution.
"""
import numpy as np
from .element import NGP

_PS3 = np.array([0, 1, 3])
EXTRAPOLATORS = ("freeze", "tangent", "linear")


def _inplane_err(dsig, sig_y0):
    """Engineering norm of the in-plane stress error, normalised by the initial yield stress."""
    return np.sqrt(dsig[:, 0] ** 2 + dsig[:, 1] ** 2 + 2 * dsig[:, 2] ** 2) / sig_y0


def oracle_greedy(hist, mat, nel, tol, extrapolator="tangent", max_skip=None,
                  check_lazy=False, err_field="stress"):
    """Greedy per-element oracle.

    Returns dict with
      integrated (T,nel) bool: element integrated at step k (True) or extrapolated (False)
      err (T,nel): extrapolation error that was evaluated at step k (0 if integrated)
      lazy_sig_err / lazy_alpha_err (T,nel): error of the deferred integration at wake-up
    """
    eps, sig, epsp, alpha = hist["eps"], hist["stress"], hist["eps_p"], hist["alpha"]
    D = hist["D"]
    T, ngp = alpha.shape
    gp_el = np.repeat(np.arange(nel), NGP)
    k0 = np.zeros(nel, dtype=int) - 1        # last integrated step per element (-1: initial state)
    integrated = np.zeros((T, nel), bool)
    err = np.zeros((T, nel))
    lazy_sig = np.zeros((T, nel)); lazy_al = np.zeros((T, nel))
    # state at "step -1" = virgin state
    sig_m1 = np.zeros((ngp, 4)); eps_m1 = np.zeros((ngp, 3)); D_m1 = np.broadcast_to(mat.C3, (ngp, 3, 3))
    epsp_m1 = np.zeros((ngp, 4)); al_m1 = np.zeros(ngp); be_m1 = np.zeros((ngp, 4))
    beta = hist.get("beta", np.zeros_like(epsp))
    gpi = np.arange(ngp)

    def gather(arr, arr_m1, kk):
        mask = (kk >= 0).reshape((-1,) + (1,) * (arr.ndim - 2))
        return np.where(mask, arr[np.maximum(kk, 0), gpi], arr_m1)

    for k in range(T):
        kg = k0[gp_el]
        s0 = gather(sig, sig_m1, kg)
        if extrapolator == "freeze":
            shat = s0[:, _PS3]
        elif extrapolator == "tangent":
            e0 = gather(eps, eps_m1, kg)
            D0 = gather(D, D_m1, kg).astype(np.float64)
            shat = s0[:, _PS3] + np.einsum("nij,nj->ni", D0, eps[k] - e0)
        elif extrapolator == "linear":
            kprev = kg - 1
            s_prev = gather(sig, sig_m1, kprev)
            shat = s0[:, _PS3] + ((k - kg)[:, None]) * (s0[:, _PS3] - s_prev[:, _PS3])
            shat = np.where((kg >= 1)[:, None], shat, s0[:, _PS3])
        else:
            raise ValueError(extrapolator)
        if err_field == "stress":
            e_gp = _inplane_err(shat - sig[k][:, _PS3], mat.sig_y0)
        else:
            raise ValueError(err_field)
        e_el = e_gp.reshape(nel, NGP).max(1)
        err[k] = e_el
        ok = e_el <= tol
        if max_skip is not None:
            ok &= (k - k0) <= max_skip
        integrated[k] = ~ok
        if check_lazy:
            # deferred integration from the anchor state to the true strain at step k
            ep0 = gather(epsp, epsp_m1, kg); a0 = gather(alpha, al_m1, kg); b0 = gather(beta, be_m1, kg)
            s_l, ep_l, a_l, b_l, pl_l, _, _ = mat.integrate(eps[k], ep0, a0, b0, need_tangent=False)
            lazy_sig[k] = _inplane_err(s_l[:, _PS3] - sig[k][:, _PS3], mat.sig_y0).reshape(nel, NGP).max(1)
            lazy_al[k] = np.abs(a_l - alpha[k]).reshape(nel, NGP).max(1)
        k0[~ok] = k
    return dict(integrated=integrated, err=err, lazy_sig_err=lazy_sig, lazy_alpha_err=lazy_al)


def cost_weighted_potential(integrated, hist, costs):
    """Fraction of element-level cost saved, given per-element cost model `costs`
    (dict with active_el_elastic, active_el_plastic (per element and per iteration),
    quiet_el). Uses the reference iteration counts (assumption: unchanged online)."""
    n_iter = np.asarray(hist["n_iter"]) + 1          # integrations per step (iters + final check)
    gp_pl = hist["gp_plastic"].astype(float)          # (T,nel) plastic GPs per element
    c_act = costs["active_el_elastic"] + gp_pl / NGP * (costs["active_el_plastic"] - costs["active_el_elastic"])
    c_ref = (n_iter[:, None] * c_act).sum()
    c_orc = (n_iter[:, None] * np.where(integrated, c_act, costs["quiet_el"])).sum()
    return 1 - c_orc / c_ref, c_ref, c_orc


def unit_cost_model(rho, plastic_ratio):
    """Transferable abstract model: elastic active element = 1, plastic active element =
    plastic_ratio, quiet (extrapolated) element = rho."""
    return dict(active_el_elastic=1.0, active_el_plastic=plastic_ratio, quiet_el=rho)
