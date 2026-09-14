"""Test cases. Units: mm, N, MPa."""
from dataclasses import dataclass
import numpy as np
from .mesh import rect_mesh, notched_plate_mesh, nodes_on
from .element import Q4Elements
from .material import J2Material
from .solver import FEModel

STEEL = dict(E=200e3, nu=0.3, sig_y0=250.0, H=1000.0, sig_inf=400.0, delta=20.0)
# combined hardening for cyclic cases: Bauschinger effect makes reverse yielding frequent
STEEL_KIN = dict(E=200e3, nu=0.3, sig_y0=250.0, H=500.0, sig_inf=320.0, delta=20.0, Hk=4000.0)


@dataclass
class Case:
    name: str
    nodes: np.ndarray
    elems: np.ndarray
    material: J2Material
    fixed_dofs: np.ndarray
    ubar: np.ndarray
    lam_hist: np.ndarray
    reaction_dofs: np.ndarray
    meta: dict

    def model(self, **kw):
        el = Q4Elements(self.nodes, self.elems)
        m = FEModel(el, self.material, self.fixed_dofs, self.ubar, self.lam_hist, **kw)
        m.reaction_dofs = self.reaction_dofs
        return m


def _ramp(n, a=0.0, b=1.0):
    return np.linspace(a, b, n + 1)[1:]


def notched_plate(h=0.5, L=40.0, H=20.0, R=4.0, u_max=0.06, n_steps=60, lam_hist=None,
                  mat=None, name="notched_plate"):
    nodes, elems = notched_plate_mesh(L, H, R, h)
    left = nodes_on(nodes, 0, 0.0)
    right = nodes_on(nodes, 0, L)
    fixed = np.concatenate([2 * left, 2 * left + 1, 2 * right, 2 * right + 1])
    ubar = np.zeros(2 * len(nodes))
    ubar[2 * right] = u_max
    lam = _ramp(n_steps) if lam_hist is None else np.asarray(lam_hist)
    m = J2Material(**(STEEL if mat is None else mat))
    return Case(name, nodes, elems, m, fixed, ubar, lam, 2 * right,
                dict(L=L, H=H, R=R, h=h, u_max=u_max, notch=(L / 2, 0.0)))


def cantilever(h=0.5, L=50.0, H=10.0, v_max=1.0, n_steps=50, lam_hist=None, mat=None,
               name="cantilever"):
    nodes, elems = rect_mesh(L, H, int(round(L / h)), int(round(H / h)))
    left = nodes_on(nodes, 0, 0.0)
    right = nodes_on(nodes, 0, L)
    fixed = np.concatenate([2 * left, 2 * left + 1, 2 * right + 1])
    ubar = np.zeros(2 * len(nodes))
    ubar[2 * right + 1] = v_max
    lam = _ramp(n_steps) if lam_hist is None else np.asarray(lam_hist)
    m = J2Material(**(STEEL if mat is None else mat))
    return Case(name, nodes, elems, m, fixed, ubar, lam, 2 * right + 1,
                dict(L=L, H=H, h=h, v_max=v_max))


def cyclic_notched_plate(h=0.5, n_up=30, n_down=48, n_re=48, lam_min=-0.6, **kw):
    """Load / unload+reverse / reload with a uniform lambda increment."""
    d = 1.0 / n_up
    lam = np.concatenate([np.linspace(d, 1.0, n_up),
                          1.0 - d * np.arange(1, n_down + 1),
                          ])
    lam = lam[lam >= lam_min - 1e-12]
    lam_low = lam[-1]
    lam = np.concatenate([lam, lam_low + d * np.arange(1, n_re + 1)])
    lam = lam[lam <= 1.0 + 1e-12]
    return notched_plate(h=h, lam_hist=lam, name="cyclic_notched", **kw)


def cyclic_cantilever(h=0.5, n_up=25, lam_min=-0.6, **kw):
    d = 1.0 / n_up
    lam = np.concatenate([np.linspace(d, 1.0, n_up), 1.0 - d * np.arange(1, 200)])
    lam = lam[lam >= lam_min - 1e-12]
    lam = np.concatenate([lam, lam[-1] + d * np.arange(1, 200)])
    lam = lam[lam <= 1.0 + 1e-12]
    return cantilever(h=h, lam_hist=lam, name="cyclic_cantilever", **kw)


def shear_block(n=4, gamma_max=0.02, n_steps=20, mat=None):
    """Homogeneous simple shear of a unit square: u_x = gamma*y, u_y = 0 on all nodes.
    Used for validation against the closed-form J2 shear response."""
    nodes, elems = rect_mesh(1.0, 1.0, n, n)
    ndof = 2 * len(nodes)
    fixed = np.arange(ndof)
    ubar = np.zeros(ndof)
    ubar[0::2] = gamma_max * nodes[:, 1]
    lam = _ramp(n_steps)
    m = J2Material(**(dict(E=200e3, nu=0.3, sig_y0=250.0, H=2000.0) if mat is None else mat))
    top = nodes_on(nodes, 1, 1.0)
    return Case("shear_block", nodes, elems, m, fixed, ubar, lam, 2 * top, dict(gamma_max=gamma_max))


def notched_plate_overload(h=0.5, **kw):
    """Unfavourable variant: loaded to general yield of the ligament."""
    return notched_plate(h=h, u_max=0.12, name="notched_overload", **kw)


def cyclic_notched_kin(h=0.5, **kw):
    c = cyclic_notched_plate(h=h, mat=STEEL_KIN, **kw)
    c.name = "cyclic_notched_kin"
    return c


def cyclic_cantilever_kin(h=0.5, **kw):
    c = cyclic_cantilever(h=h, mat=STEEL_KIN, **kw)
    c.name = "cyclic_cantilever_kin"
    return c


def notched_family(R=4.0, H=20.0, L=40.0, h=0.5, u_max=None, **kw):
    """Parametrised family: notch radius R, plate height H (same nominal strain at u_max)."""
    u = 0.06 * L / 40.0 if u_max is None else u_max
    c = notched_plate(h=h, L=L, H=H, R=R, u_max=u, **kw)
    c.name = f"notched_R{R:g}_H{H:g}_L{L:g}"
    return c


def cantilever_family(H=10.0, L=50.0, h=0.5, v_max=None, **kw):
    v = 1.0 * (L / 50.0) ** 2 / (H / 10.0) if v_max is None else v_max
    c = cantilever(h=h, L=L, H=H, v_max=v, **kw)
    c.name = f"cantilever_H{H:g}_L{L:g}"
    return c


CASES = {"notched_plate": notched_plate, "notched_overload": notched_plate_overload,
         "cantilever": cantilever, "cyclic_notched_kin": cyclic_notched_kin,
         "cyclic_cantilever_kin": cyclic_cantilever_kin,
         "cyclic_notched": cyclic_notched_plate, "cyclic_cantilever": cyclic_cantilever}
