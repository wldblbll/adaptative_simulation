"""Structured Q4 mesh generators. The mesh is an *input* of the method: it is never
refined or modified by the orchestrator."""
import numpy as np


def rect_mesh(Lx, Ly, nx, ny, x0=0.0, y0=0.0):
    xs = np.linspace(x0, x0 + Lx, nx + 1)
    ys = np.linspace(y0, y0 + Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    n0 = nid[:-1, :-1].ravel(); n1 = nid[:-1, 1:].ravel()
    n2 = nid[1:, 1:].ravel(); n3 = nid[1:, :-1].ravel()
    elems = np.column_stack([n0, n1, n2, n3])
    return nodes, elems


def remove_elements(nodes, elems, mask_remove):
    keep = elems[~mask_remove]
    used = np.unique(keep)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    return nodes[used].copy(), remap[keep]


def notched_plate_mesh(L, H, R, h, notch_x=None):
    """Rectangle L x H with a semicircular edge notch of radius R at the bottom edge.
    Elements whose centroid lies inside the notch are removed; remaining nodes inside the
    circle are projected radially onto it."""
    nx, ny = int(round(L / h)), int(round(H / h))
    nodes, elems = rect_mesh(L, H, nx, ny)
    cx = L / 2 if notch_x is None else notch_x
    cen = nodes[elems].mean(axis=1)
    inside_el = np.hypot(cen[:, 0] - cx, cen[:, 1]) < R
    nodes, elems = remove_elements(nodes, elems, inside_el)
    d = np.hypot(nodes[:, 0] - cx, nodes[:, 1])
    ins = d < R - 1e-12
    ang = np.arctan2(nodes[ins, 1], nodes[ins, 0] - cx)
    nodes[ins, 0] = cx + R * np.cos(ang)
    nodes[ins, 1] = R * np.sin(ang)
    return nodes, elems


def nodes_on(nodes, axis, value, tol=1e-9):
    return np.nonzero(np.abs(nodes[:, axis] - value) < tol)[0]


def element_adjacency(elems):
    """List of neighbouring elements (sharing at least one node) as CSR arrays."""
    nel = elems.shape[0]
    nn = elems.max() + 1
    node_to_el = [[] for _ in range(nn)]
    for e, en in enumerate(elems):
        for n in en:
            node_to_el[n].append(e)
    neigh = [set() for _ in range(nel)]
    for lst in node_to_el:
        for e in lst:
            neigh[e].update(lst)
    for e in range(nel):
        neigh[e].discard(e)
    indptr = np.zeros(nel + 1, dtype=int)
    indices = []
    for e in range(nel):
        s = sorted(neigh[e])
        indices.extend(s)
        indptr[e + 1] = indptr[e] + len(s)
    return indptr, np.array(indices, dtype=int)
