"""Spatial candidate networks with a single source and unit sinks."""
import numpy as np
from scipy.spatial import Delaunay, cKDTree

def make_domain(n_sinks=600, seed=3, kind="delaunay", knn=6):
    """Random points in the unit disc; node 0 is the source at the centre.
    kind='delaunay' gives a planar triangulation (some degree-3+ junctions);
    kind='knn' links each node to its knn nearest neighbours."""
    rng = np.random.default_rng(seed)
    u, t = rng.random(n_sinks), rng.random(n_sinks) * 2 * np.pi
    pts = np.vstack([[0.0, 0.0], np.c_[np.sqrt(u) * np.cos(t), np.sqrt(u) * np.sin(t)]])
    E = set()
    if kind == "delaunay":
        for s in Delaunay(pts).simplices:
            for a in range(3):
                i, j = int(s[a]), int(s[(a + 1) % 3]); E.add((min(i, j), max(i, j)))
    elif kind == "knn":
        _, nb = cKDTree(pts).query(pts, k=knn + 1)
        for i, row in enumerate(nb):
            for j in row[1:]: E.add((min(i, int(j)), max(i, int(j))))
    else:
        raise ValueError(kind)
    E = sorted(E)
    L = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in E])
    return pts, E, L
