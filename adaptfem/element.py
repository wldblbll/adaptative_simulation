"""Bilinear quadrilateral (Q4), 2x2 Gauss, plane strain, optional B-bar (mean dilatation)."""
import numpy as np

_G = 1.0 / np.sqrt(3.0)
GAUSS = np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]])
NGP = 4


def _dN_dxi(xi, eta):
    return 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                            [(1 - eta), -(1 + xi)],
                            [(1 + eta), (1 + xi)],
                            [-(1 + eta), (1 - xi)]])


class Q4Elements:
    def __init__(self, nodes, elems, bbar=True):
        self.nodes = nodes
        self.elems = elems
        self.nel = elems.shape[0]
        self.ndof = 2 * nodes.shape[0]
        X = nodes[elems]  # (nel,4,2)
        B = np.zeros((self.nel, NGP, 3, 8))
        Bdil = np.zeros((self.nel, NGP, 3, 8))
        wdetJ = np.zeros((self.nel, NGP))
        for g in range(NGP):
            dN = _dN_dxi(*GAUSS[g])                       # (4,2)
            J = np.einsum("ai,naj->nij", dN, X)            # J_ij = dx_j/dxi_i
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            invJ = np.empty_like(J)
            invJ[:, 0, 0] = J[:, 1, 1]; invJ[:, 1, 1] = J[:, 0, 0]
            invJ[:, 0, 1] = -J[:, 0, 1]; invJ[:, 1, 0] = -J[:, 1, 0]
            invJ /= detJ[:, None, None]
            dNdx = np.einsum("nji,ai->naj", invJ, dN)      # (nel,4,2)
            for a in range(4):
                B[:, g, 0, 2 * a] = dNdx[:, a, 0]
                B[:, g, 1, 2 * a + 1] = dNdx[:, a, 1]
                B[:, g, 2, 2 * a] = dNdx[:, a, 1]
                B[:, g, 2, 2 * a + 1] = dNdx[:, a, 0]
                Bdil[:, g, 0, 2 * a] = dNdx[:, a, 0] / 3
                Bdil[:, g, 0, 2 * a + 1] = dNdx[:, a, 1] / 3
                Bdil[:, g, 1, 2 * a] = dNdx[:, a, 0] / 3
                Bdil[:, g, 1, 2 * a + 1] = dNdx[:, a, 1] / 3
            wdetJ[:, g] = detJ
        self.min_detJ = wdetJ.min()
        if bbar:
            Bdil_mean = (Bdil * wdetJ[:, :, None, None]).sum(1) / wdetJ.sum(1)[:, None, None]
            B = B - Bdil + Bdil_mean[:, None]
        self.B = B
        self.wdetJ = wdetJ
        self.area = wdetJ.sum(1)
        self.dofs = np.empty((self.nel, 8), dtype=int)
        self.dofs[:, 0::2] = 2 * elems
        self.dofs[:, 1::2] = 2 * elems + 1
        self.centroids = X.mean(1)
        self._build_pattern()

    def _build_pattern(self):
        rows = np.repeat(self.dofs, 8, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 8)).ravel()
        key = rows.astype(np.int64) * self.ndof + cols
        ukey, inv = np.unique(key, return_inverse=True)
        self.asm_map = inv
        urows = ukey // self.ndof
        ucols = ukey % self.ndof
        self.nnz = ukey.size
        self.indices = ucols.astype(np.int32)
        self.indptr = np.zeros(self.ndof + 1, dtype=np.int32)
        np.add.at(self.indptr, urows + 1, 1)
        self.indptr = np.cumsum(self.indptr).astype(np.int32)

    # kinematics -----------------------------------------------------------
    def strains(self, u, el=None):
        B = self.B if el is None else self.B[el]
        d = self.dofs if el is None else self.dofs[el]
        return np.einsum("ngij,nj->ngi", B, u[d])

    # element quantities ---------------------------------------------------
    def element_fint(self, sig3, el=None):
        """sig3: (n_el_sel, NGP, 3) -> element internal force (n_el_sel, 8)."""
        B = self.B if el is None else self.B[el]
        w = self.wdetJ if el is None else self.wdetJ[el]
        return np.einsum("ngij,ngi,ng->nj", B, sig3, w)

    def element_K(self, D, el=None):
        """D: (n_el_sel, NGP, 3, 3) -> element stiffness (n_el_sel, 8, 8)."""
        B = self.B if el is None else self.B[el]
        w = self.wdetJ if el is None else self.wdetJ[el]
        DB = np.einsum("ngkl,nglj->ngkj", D, B)
        return np.einsum("ngki,ngkj,ng->nij", B, DB, w)

    # assembly ---------------------------------------------------------------
    def assemble_fint(self, fe, el=None):
        f = np.zeros(self.ndof)
        d = self.dofs if el is None else self.dofs[el]
        np.add.at(f, d.ravel(), fe.ravel())
        return f

    def assemble_K_data(self, Ke, el=None):
        """Return CSR data array (length nnz) for the given element matrices.
        Elements not in `el` contribute zero."""
        m = self.asm_map if el is None else self.asm_map.reshape(self.nel, 64)[el].ravel()
        return np.bincount(m, weights=Ke.ravel(), minlength=self.nnz)

    def csr(self, data):
        from scipy.sparse import csr_matrix
        return csr_matrix((data, self.indices, self.indptr), shape=(self.ndof, self.ndof))
