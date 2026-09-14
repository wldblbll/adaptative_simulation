import numpy as np
from adaptfem.material import J2Material

rng = np.random.default_rng(0)


def _mat():
    return J2Material(E=200e3, nu=0.3, sig_y0=250.0, H=1000.0, sig_inf=400.0, delta=20.0)


def test_scalar_vs_vectorised():
    m = _mat()
    n = 200
    eps = rng.normal(0, 1.2e-3, (n, 3))
    eps_p = rng.normal(0, 1e-3, (n, 4)); eps_p[:, 2] = -(eps_p[:, 0] + eps_p[:, 1])
    alpha = np.abs(rng.normal(0, 2e-3, n))
    sig, ep, al, pl, D, _ = m.integrate(eps, eps_p, alpha)
    assert pl.any() and (~pl).any()
    for i in range(n):
        s, e, a, p, d = m.integrate_scalar(eps[i], eps_p[i], alpha[i])
        assert np.allclose(s, sig[i], rtol=1e-8, atol=1e-8)
        assert np.allclose(e, ep[i], rtol=1e-8, atol=1e-12)
        assert abs(a - al[i]) < 1e-12
        assert p == pl[i]
        assert np.allclose(d, D[i], rtol=1e-8, atol=1e-4)


def test_consistent_tangent_finite_difference():
    m = _mat()
    n = 50
    eps = rng.normal(0, 4e-3, (n, 3))
    eps_p = np.zeros((n, 4))
    alpha = np.zeros(n)
    sig, ep, al, pl, D, _ = m.integrate(eps, eps_p, alpha)
    assert pl.sum() > 10
    h = 1e-7
    Dfd = np.zeros((n, 3, 3))
    for j in range(3):
        e1 = eps.copy(); e1[:, j] += h
        e2 = eps.copy(); e2[:, j] -= h
        s1 = m.integrate(e1, eps_p, alpha, need_tangent=False)[0]
        s2 = m.integrate(e2, eps_p, alpha, need_tangent=False)[0]
        Dfd[:, :, j] = (s1 - s2)[:, [0, 1, 3]] / (2 * h)
    err = np.abs(D - Dfd).max() / np.abs(D).max()
    assert err < 1e-5, err


def test_yield_surface_consistency():
    m = _mat()
    eps = rng.normal(0, 5e-3, (100, 3))
    sig, ep, al, pl, D, _ = m.integrate(eps, np.zeros((100, 4)), np.zeros(100))
    p = sig[:, :3].sum(1) / 3
    s = sig.copy(); s[:, :3] -= p[:, None]
    q = np.sqrt(1.5 * (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + 2 * s[:, 3] ** 2))
    assert np.allclose(q[pl], m.sig_y(al[pl]), rtol=1e-9)
    assert np.all(q[~pl] <= m.sig_y(al[~pl]) + 1e-9)
    # plastic incompressibility
    assert np.allclose(ep[:, :3].sum(1), 0, atol=1e-14)
