import numpy as np
from adaptfem import cases
from adaptfem.mesh import rect_mesh
from adaptfem.element import Q4Elements
from adaptfem.material import J2Material
from adaptfem.solver import FEModel


def test_patch_test_linear_field():
    """Distorted mesh, prescribed linear displacement on all boundary nodes -> exact
    uniform strain/stress inside (elastic)."""
    rng = np.random.default_rng(1)
    nodes, elems = rect_mesh(2.0, 1.0, 6, 3)
    inner = (nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2 - 1e-9) & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1 - 1e-9)
    nodes[inner] += rng.uniform(-0.08, 0.08, (inner.sum(), 2))
    el = Q4Elements(nodes, elems)
    assert el.min_detJ > 0
    a = np.array([[1e-3, 4e-4], [-2e-4, 5e-4]])
    u_exact = (nodes @ a.T).ravel()
    bnd = np.nonzero(~inner)[0]
    fixed = np.concatenate([2 * bnd, 2 * bnd + 1])
    m = J2Material(E=1000.0, nu=0.3, sig_y0=1e9)
    model = FEModel(el, m, fixed, u_exact, [1.0])
    u, state = model.run()
    assert np.allclose(u, u_exact, atol=1e-12)
    eps = el.strains(u).reshape(-1, 3)
    assert np.allclose(eps[:, 0], a[0, 0]) and np.allclose(eps[:, 1], a[1, 1])
    assert np.allclose(eps[:, 2], a[0, 1] + a[1, 0])


def test_shear_closed_form():
    """Homogeneous simple shear, linear hardening: tau(gamma) = (gamma + sqrt3*sy/H)/(1/mu + 3/H)."""
    c = cases.shear_block(n=3, gamma_max=0.02, n_steps=10)
    model = c.model(record_history=True)
    u, state = model.run()
    m = c.material
    gam = c.meta["gamma_max"] * c.lam_hist
    tau_el = m.mu * gam
    tau_y = m.sig_y0 / np.sqrt(3)
    tau_pl = (gam + np.sqrt(3) * m.sig_y0 / m.H) / (1 / m.mu + 3 / m.H)
    tau_exact = np.where(tau_el <= tau_y, tau_el, tau_pl)
    h = model.stacked_history()
    tau_fe = h["stress"][:, :, 3].mean(1)
    assert np.allclose(tau_fe, tau_exact, rtol=1e-10)
    assert np.allclose(h["stress"][:, :, 3].std(1), 0, atol=1e-8)


def test_newton_quadratic_convergence():
    c = cases.notched_plate(h=1.0, n_steps=6, u_max=0.12)
    model = c.model(record_history=True, tol=1e-12)
    model.run()
    quad = 0
    for rec in model.records:
        r = np.array(rec.residuals)
        if len(r) >= 4 and int(model.history["gp_plastic"][rec.step].sum()) > 0:
            # last two ratios: r_{k+1}/r_k^2 should stay bounded while r_{k+1}/r_k -> 0
            ratios = r[1:] / r[:-1]
            if ratios[-1] < 0.05 * ratios[-2] or ratios[-1] < 1e-3:
                quad += 1
    assert quad >= 2, [rec.residuals for rec in model.records]


def test_orchestrated_full_equals_reference():
    """Orchestrator that activates everything must reproduce the reference bit-for-bit."""
    class AllActive:
        def select(self, k, u, state, model):
            return np.arange(model.nel)
        def after_step(self, *a):
            pass
    c = cases.cantilever(h=2.0, n_steps=5)
    m1 = c.model(); u1, s1 = m1.run()
    m2 = c.model(); u2, s2 = m2.run(orchestrator=AllActive())
    assert np.array_equal(u1, u2) and np.array_equal(s1.alpha, s2.alpha)
