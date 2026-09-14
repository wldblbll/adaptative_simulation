"""J2 (von Mises) plasticity, plane strain, combined isotropic (linear + Voce) and linear
kinematic (Prager) hardening.

State at a Gauss point: stress (4,) Voigt [xx, yy, zz, xy]; plastic strain (4,) tensorial
(eps_p_xy, not gamma); accumulated plastic strain alpha; back-stress beta (4,) deviatoric.
Strain input is plane-strain engineering Voigt [exx, eyy, gxy]; eps_zz = 0.

Radial return following Simo & Hughes (Computational Inelasticity, Box 3.2), consistent
tangent returned as a 3x3 plane-strain matrix (engineering shear).
"""
from dataclasses import dataclass
import math
import numpy as np

SQ23 = np.sqrt(2.0 / 3.0)
_M = np.array([1.0, 1.0, 1.0, 0.0])
_P = np.array([[2 / 3, -1 / 3, -1 / 3, 0],
               [-1 / 3, 2 / 3, -1 / 3, 0],
               [-1 / 3, -1 / 3, 2 / 3, 0],
               [0, 0, 0, 0.5]])
_PS = np.array([0, 1, 3])  # plane-strain rows/cols of the 4x4 Voigt operator


def dev_norm(s):
    """||dev(s)|| (tensor norm) for stress-like (n,4) Voigt arrays."""
    p = (s[:, 0] + s[:, 1] + s[:, 2]) / 3
    d0, d1, d2 = s[:, 0] - p, s[:, 1] - p, s[:, 2] - p
    return np.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2 + 2 * s[:, 3] ** 2), np.stack([d0, d1, d2, s[:, 3]], 1)


@dataclass
class J2Material:
    E: float
    nu: float
    sig_y0: float
    H: float = 0.0          # linear isotropic hardening modulus
    sig_inf: float = None   # Voce saturation stress (None -> no Voce term)
    delta: float = 0.0      # Voce exponent
    Hk: float = 0.0         # linear kinematic hardening modulus (Prager)
    local_tol: float = 1e-10
    local_maxit: int = 50

    def __post_init__(self):
        E, nu = self.E, self.nu
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = self.lam + 2 * self.mu / 3
        self.C4 = self.kappa * np.outer(_M, _M) + 2 * self.mu * _P
        self.C3 = self.C4[np.ix_(_PS, _PS)]
        if self.sig_inf is None:
            self.sig_inf = self.sig_y0

    # hardening law -------------------------------------------------------
    def sig_y(self, a):
        return self.sig_y0 + self.H * a + (self.sig_inf - self.sig_y0) * (1 - np.exp(-self.delta * a))

    def dsig_y(self, a):
        return self.H + (self.sig_inf - self.sig_y0) * self.delta * np.exp(-self.delta * a)

    # cheap monitor ---------------------------------------------------------
    def yield_function(self, sig, alpha, beta):
        """f = ||dev(sig) - beta|| - sqrt(2/3) sig_y(alpha), normalised by sig_y0.
        Used to *monitor* extrapolated (quiet) Gauss points without a return mapping."""
        _, d = dev_norm(sig)
        xi = d - beta
        xn = np.sqrt(xi[:, 0] ** 2 + xi[:, 1] ** 2 + xi[:, 2] ** 2 + 2 * xi[:, 3] ** 2)
        return (xn - SQ23 * self.sig_y(alpha)) / self.sig_y0

    # vectorised integration ---------------------------------------------
    def integrate(self, eps, eps_p, alpha, beta=None, need_tangent=True):
        """Return (stress(n,4), eps_p_new(n,4), alpha_new(n,), beta_new(n,4),
        plastic(n,) bool, D(n,3,3) or None, n_local_newton_iters)."""
        n = eps.shape[0]
        mu, lam, Hk = self.mu, self.lam, self.Hk
        if beta is None:
            beta = np.zeros((n, 4))
        ee = np.empty((n, 4))
        ee[:, 0] = eps[:, 0] - eps_p[:, 0]
        ee[:, 1] = eps[:, 1] - eps_p[:, 1]
        ee[:, 2] = -eps_p[:, 2]
        ee[:, 3] = 0.5 * eps[:, 2] - eps_p[:, 3]
        tr = ee[:, 0] + ee[:, 1] + ee[:, 2]
        sig = 2 * mu * ee
        sig[:, :3] += lam * tr[:, None]
        p = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3
        xi = sig.copy()
        xi[:, :3] -= p[:, None]
        xi -= beta
        xnorm = np.sqrt(xi[:, 0] ** 2 + xi[:, 1] ** 2 + xi[:, 2] ** 2 + 2 * xi[:, 3] ** 2)
        f = xnorm - SQ23 * self.sig_y(alpha)
        plastic = f > 0
        dg = np.zeros(n)
        nit = 0
        idx = np.nonzero(plastic)[0]
        if idx.size:
            a0 = alpha[idx]
            xn = xnorm[idx]
            dgp = np.zeros(idx.size)
            for nit in range(1, self.local_maxit + 1):
                an = a0 + SQ23 * dgp
                r = xn - (2 * mu + 2 / 3 * Hk) * dgp - SQ23 * self.sig_y(an)
                if np.all(np.abs(r) < self.local_tol * self.sig_y0):
                    break
                dr = -(2 * mu + 2 / 3 * Hk) - (2 / 3) * self.dsig_y(an)
                dgp -= r / dr
            dg[idx] = dgp
        nvec = xi / np.maximum(xnorm, 1e-300)[:, None]
        sig_new = sig - (2 * mu * dg)[:, None] * nvec
        eps_p_new = eps_p + dg[:, None] * nvec
        alpha_new = alpha + SQ23 * dg
        beta_new = beta + (2 / 3 * Hk * dg)[:, None] * nvec
        D = None
        if need_tangent:
            D = np.broadcast_to(self.C3, (n, 3, 3)).copy()
            if idx.size:
                theta = 1 - 2 * mu * dg[idx] / xnorm[idx]
                Hp = self.dsig_y(alpha_new[idx])
                thetabar = 1 / (1 + (Hp + Hk) / (3 * mu)) - (1 - theta)
                nn = nvec[idx]
                D4 = (self.kappa * np.outer(_M, _M))[None] \
                    + (2 * mu * theta)[:, None, None] * _P[None] \
                    - (2 * mu * thetabar)[:, None, None] * nn[:, :, None] * nn[:, None, :]
                D[idx] = D4[:, _PS][:, :, _PS]
        return sig_new, eps_p_new, alpha_new, beta_new, plastic, D, nit

    # scalar reference implementation (for tests and cost benchmarking) --
    def integrate_scalar(self, eps, eps_p, alpha, beta=None):
        """Same algorithm written point by point in pure Python (no vectorisation).
        Returns (sig(4,), eps_p(4,), alpha, beta(4,), plastic, D(3,3))."""
        mu, lam, Hk = self.mu, self.lam, self.Hk
        if beta is None:
            beta = [0.0, 0.0, 0.0, 0.0]
        ee = [eps[0] - eps_p[0], eps[1] - eps_p[1], -eps_p[2], 0.5 * eps[2] - eps_p[3]]
        tr = ee[0] + ee[1] + ee[2]
        sig = [2 * mu * ee[0] + lam * tr, 2 * mu * ee[1] + lam * tr, 2 * mu * ee[2] + lam * tr, 2 * mu * ee[3]]
        p = (sig[0] + sig[1] + sig[2]) / 3
        xi = [sig[0] - p - beta[0], sig[1] - p - beta[1], sig[2] - p - beta[2], sig[3] - beta[3]]
        xnorm = math.sqrt(xi[0] ** 2 + xi[1] ** 2 + xi[2] ** 2 + 2 * xi[3] ** 2)
        f = xnorm - SQ23 * self.sig_y(alpha)
        if f <= 0:
            return np.array(sig), np.array(eps_p, float), alpha, np.array(beta, float), False, self.C3.copy()
        dg = 0.0
        for _ in range(self.local_maxit):
            an = alpha + SQ23 * dg
            r = xnorm - (2 * mu + 2 / 3 * Hk) * dg - SQ23 * self.sig_y(an)
            if abs(r) < self.local_tol * self.sig_y0:
                break
            dg -= r / (-(2 * mu + 2 / 3 * Hk) - (2 / 3) * self.dsig_y(an))
        nv = np.array(xi) / xnorm
        sig_new = np.array(sig) - 2 * mu * dg * nv
        eps_p_new = np.array(eps_p, float) + dg * nv
        alpha_new = alpha + SQ23 * dg
        beta_new = np.array(beta, float) + 2 / 3 * Hk * dg * nv
        theta = 1 - 2 * mu * dg / xnorm
        thetabar = 1 / (1 + (self.dsig_y(alpha_new) + Hk) / (3 * mu)) - (1 - theta)
        D4 = self.kappa * np.outer(_M, _M) + 2 * mu * theta * _P - 2 * mu * thetabar * np.outer(nv, nv)
        return sig_new, eps_p_new, alpha_new, beta_new, True, D4[np.ix_(_PS, _PS)]

    def yield_function_scalar(self, sig, alpha, beta):
        p = (sig[0] + sig[1] + sig[2]) / 3
        xi = [sig[0] - p - beta[0], sig[1] - p - beta[1], sig[2] - p - beta[2], sig[3] - beta[3]]
        return (math.sqrt(xi[0] ** 2 + xi[1] ** 2 + xi[2] ** 2 + 2 * xi[3] ** 2) - SQ23 * self.sig_y(alpha)) / self.sig_y0
