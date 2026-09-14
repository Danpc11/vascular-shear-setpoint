"""Local shear set-point remodeling.

Rule per active edge:   d ln r_e / dt = kappa (tau_e / tau_set,e - 1),
tau_set,e = tau0 r_e^{b-1} l_e^{(b-1)/2}.  Its fixed points are exactly the
KKT points of  min D  s.t.  sum_e a_e w_e^alpha = C  (see flows.py).

Modes:  steady demand; growth (sinks added outward in stages, or the whole
domain scaled isotropically while adapting); fluctuating demand, where each
sink's current has relative std sigma and adaptation follows
tau_rms = <f_e^2>^{1/2} / r_e^3, computed exactly from the demand covariance
(dense pinvh, fine up to ~1000 nodes) or by Monte Carlo with sparse solves.
"""
import numpy as np, networkx as nx
from scipy.linalg import pinvh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

def _cov_demand(net, nodes, sigma):
    idx = np.flatnonzero(nodes); loc = np.full(net.n, -1); loc[idx] = np.arange(len(idx))
    sinks = nodes.copy(); sinks[net.src] = False; sl = loc[np.flatnonzero(sinks)]; s0 = loc[net.src]
    S = np.zeros((len(idx), len(idx))); S[sl, sl] = sigma ** 2
    S[s0, sl] = S[sl, s0] = -sigma ** 2; S[s0, s0] = sinks.sum() * sigma ** 2
    return idx, S

def adapt(net, b, r0, tau0, nodes=None, sigma=0.0, kappa=0.2, max_steps=5000, tol=1e-6,
          prune=1e-3, clip=0.15, mc_samples=0, seed=0, history=None):
    nodes = np.ones(net.n, bool) if nodes is None else nodes
    r = r0.copy(); active = (r > 0) & nodes[net.Ei] & nodes[net.Ej]
    mu = net.unit_demand(nodes); rng = np.random.default_rng(seed)
    idx = np.flatnonzero(nodes); mask = idx != net.src
    if sigma > 0 and mc_samples == 0: idx, S = _cov_demand(net, nodes, sigma)
    sinks = nodes.copy(); sinks[net.src] = False
    for step in range(max_steps):
        w = net.conductance(r, active); Bs = net.B[idx][:, active] * w[active]
        Lap = (net.B[idx] * w) @ net.B[idx].T
        if sigma > 0 and mc_samples == 0:
            Lp = pinvh(Lap); phi = Lp @ mu[idx]; fmean = Bs.T @ phi
            Gm = Bs.T @ Lp; f2 = fmean ** 2 + np.einsum("ij,jk,ik->i", Gm, S, Gm)
        elif sigma > 0:
            lu = splu(csr_matrix(Lap[np.ix_(mask, mask)]).tocsc()); f2 = np.zeros(active.sum())
            for _ in range(mc_samples):
                d = mu.copy(); noise = sigma * rng.standard_normal(net.n) * sinks
                d += noise; d[net.src] -= noise.sum()
                phi = np.zeros(len(idx)); phi[mask] = lu.solve(d[idx][mask]); f2 += (Bs.T @ phi) ** 2
            f2 /= mc_samples
        else:
            phi = np.zeros(len(idx))
            phi[mask] = np.linalg.solve(Lap[np.ix_(mask, mask)], mu[idx][mask]); f2 = (Bs.T @ phi) ** 2
        ra = r[active]; tau = np.sqrt(f2) / ra ** 3
        tset = tau0 * ra ** (b - 1) * net.L[active] ** ((b - 1) / 2)
        ratio = tau / tset
        r[active] = ra * np.exp(np.clip(kappa * (ratio - 1.0), -clip, clip))
        ka = np.flatnonzero(active); gone = ka[r[ka] < prune * r[ka].max()]
        if gone.size: active[gone] = False; r[gone] = 0.0
        err = float(np.abs(ratio - 1.0).max())
        if history is not None and step % 10 == 0: history.append((step, err, int(active.sum())))
        if err < tol: break
    return r, active, step + 1, err

def grow_peripheral(net, b, tau0, stages, r_seed, **kw):
    """sinks appear outward from the source in `stages` shells; adapt after each."""
    dist = np.linalg.norm(net.pts - net.pts[net.src], axis=1)
    order = np.argsort(dist); order = order[order != net.src]
    nodes = np.zeros(net.n, bool); nodes[net.src] = True; r = np.zeros(net.m)
    for s in range(stages):
        nodes[order[int(len(order) * s / stages):int(len(order) * (s + 1) / stages)]] = True
        fresh = (r == 0) & nodes[net.Ei] & nodes[net.Ej]
        r[fresh] = 0.5 * (r_seed if not (r > 0).any() else np.median(r[r > 0]))
        r, active, _, _ = adapt(net, b, r, tau0, nodes, **kw)
    return r, active

def grow_isotropic(net, b, tau0, stages, r_seed, g0=0.2, **kw):
    """all sinks present; the domain (all lengths) is scaled from g0 to 1 in stages."""
    L_final = net.L.copy(); r = np.full(net.m, 0.5 * r_seed); active = None
    for g in np.linspace(g0, 1.0, stages):
        net.L = L_final * g
        r, active, _, _ = adapt(net, b, r, tau0, **kw)
    net.L = L_final
    return r, active

def summarize(net, r, active, b):
    G = nx.Graph([net.E[k] for k in np.flatnonzero(active)])
    beta = int(active.sum() - G.number_of_nodes() + nx.number_connected_components(G))
    return dict(edges=int(active.sum()), beta=beta, D_norm=net.d_norm(r, active, b),
                C=net.cost(r, active, b))
