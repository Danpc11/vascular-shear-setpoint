"""Local shear set-point remodeling.

Rule per active edge, with x_e = ln r_e and z_e = |tau_e| / tau_set,e >= 0,

    dx_e/dt = kappa (z_e - 1),      tau_set,e = tau_0 r_e^{b-1} l_e^{(b-1)/2}.

Its rest points are the stationary points of  min D  s.t.  sum_e a_e w_e^alpha
= C, and on the penalized functional F = D + lambda C_b the rule is a strict
Lyapunov descent,

    dF/dt = -2 b lambda kappa sum_e c_e (z_e - 1)^2 (z_e + 1) <= 0,

with c_e = a_e w_e^alpha (verified numerically by checks.c7_lyapunov). The
identity assumes fixed geometry and loading, constant lambda and kappa,
continuous time and a fixed admissible support; the explicit Euler step and the
discrete pruning below are outside it.

Loadings
  steady        mu_e from unit demand at every node except the source.
  fluctuating   each sink draws an independent current of mean 1 and standard
                deviation sigma. The rule follows the root-mean-square shear,
                <f_e^2>^{1/2} / r_e^3, obtained from the second moment
                Q = mean(I) mean(I)^T + Cov(I); both terms are kept, so the
                sigma -> 0 limit reproduces the steady case exactly.
"""
import numpy as np
import networkx as nx
from scipy.linalg import pinvh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


def _cov_demand(net, nodes, sigma):
    """Cov(I) on the present nodes for independent sink currents of s.d. sigma.

    I = (sum_t d_t) e_s - sum_t d_t e_t with d_t independent of variance
    sigma^2, so Cov is sigma^2 on each sink diagonal, -sigma^2 between the
    source and each sink, and n_sinks sigma^2 on the source diagonal.
    """
    idx = np.flatnonzero(nodes)
    loc = np.full(net.n, -1)
    loc[idx] = np.arange(len(idx))
    sinks = nodes.copy()
    sinks[net.src] = False
    sl = loc[np.flatnonzero(sinks)]
    s0 = loc[net.src]
    S = np.zeros((len(idx), len(idx)))
    S[sl, sl] = sigma ** 2
    S[s0, sl] = S[sl, s0] = -sigma ** 2
    S[s0, s0] = sinks.sum() * sigma ** 2
    return idx, S


def adapt(net, b, r0, tau0, nodes=None, sigma=0.0, kappa=0.2, max_steps=5000,
          tol=1e-6, prune=1e-3, clip=0.02, mc_samples=0, seed=0, history=None):
    """Integrate the rule by explicit Euler in ln r.

    A vessel whose radius falls below `prune` times the largest is removed;
    removal is irreversible, which is the discrete step the Lyapunov identity
    does not cover. Convergence means max_e |z_e - 1| < tol over active edges.

    `clip` bounds |d ln r| per step. It is a discretisation parameter, not a
    robustness knob: the network the rule settles on depends on it systematically
    until the step is small enough, and coarser steps give worse networks. The
    default 0.02 is where checks.c14_step_convergence finds the remaining drift
    comparable to the spread between basins; 0.15 is not converged.

    mc_samples > 0 replaces the dense pseudo-inverse by Monte Carlo over demand
    patterns with sparse solves; use it above roughly 1000 nodes.
    """
    nodes = np.ones(net.n, bool) if nodes is None else nodes
    r = r0.copy()
    active = (r > 0) & nodes[net.Ei] & nodes[net.Ej]
    mu = net.unit_demand(nodes)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(nodes)
    if sigma > 0 and mc_samples == 0:
        idx, S = _cov_demand(net, nodes, sigma)
    mask = idx != net.src
    sinks = nodes.copy()
    sinks[net.src] = False
    steps = 0
    err = np.inf
    for steps in range(1, max_steps + 1):
        w = net.conductance(r, active)
        Bs = net.B[idx][:, active] * w[active]
        Lap = (net.B[idx] * w) @ net.B[idx].T
        if sigma > 0 and mc_samples == 0:
            Lp = pinvh(Lap)
            fmean = Bs.T @ (Lp @ mu[idx])
            G = Bs.T @ Lp
            f2 = fmean ** 2 + np.einsum("ij,jk,ik->i", G, S, G)
        elif sigma > 0:
            lu = splu(csr_matrix(Lap[np.ix_(mask, mask)]).tocsc())
            f2 = np.zeros(int(active.sum()))
            for _ in range(mc_samples):
                d = mu.copy()
                noise = sigma * rng.standard_normal(net.n) * sinks
                d += noise
                d[net.src] -= noise.sum()
                phi = np.zeros(len(idx))
                phi[mask] = lu.solve(d[idx][mask])
                f2 += (Bs.T @ phi) ** 2
            f2 /= mc_samples
        else:
            phi = np.zeros(len(idx))
            phi[mask] = np.linalg.solve(Lap[np.ix_(mask, mask)], mu[idx][mask])
            f2 = (Bs.T @ phi) ** 2
        ra = r[active]
        tau = np.sqrt(f2) / ra ** 3                       # |tau|, or its rms
        tset = tau0 * ra ** (b - 1) * net.L[active] ** ((b - 1) / 2)
        z = tau / tset
        r[active] = ra * np.exp(np.clip(kappa * (z - 1.0), -clip, clip))
        ka = np.flatnonzero(active)
        gone = ka[r[ka] < prune * r[ka].max()]
        if gone.size:
            active[gone] = False
            r[gone] = 0.0
        err = float(np.abs(z - 1.0).max())
        if history is not None and steps % 10 == 0:
            history.append((steps, err, int(active.sum())))
        if err < tol:
            break
    return r, active, steps, err


def grow_peripheral(net, b, tau0, stages, r_seed, **kw):
    """Sinks appear outward from the source in `stages` shells.

    The rule is run to convergence after each shell, and edges that become
    admissible are seeded at half the median radius of the current network.
    """
    dist = np.linalg.norm(net.pts - net.pts[net.src], axis=1)
    order = np.argsort(dist)
    order = order[order != net.src]
    nodes = np.zeros(net.n, bool)
    nodes[net.src] = True
    r = np.zeros(net.m)
    active = np.zeros(net.m, bool)
    for s in range(stages):
        lo = int(len(order) * s / stages)
        hi = int(len(order) * (s + 1) / stages)
        nodes[order[lo:hi]] = True
        fresh = (r == 0) & nodes[net.Ei] & nodes[net.Ej]
        r[fresh] = 0.5 * (r_seed if not (r > 0).any() else np.median(r[r > 0]))
        r, active, _, _ = adapt(net, b, r, tau0, nodes, **kw)
    return r, active


def grow_isotropic(net, b, tau0, stages, r_seed, g0=0.2, **kw):
    """All sinks present; the domain is dilated from g0 to 1 in `stages`.

    net.L is mutated during the sweep and restored in a finally block, so an
    interrupted run leaves the Net usable. Note that net.G still carries the
    unscaled lengths: only routines reading net.L (adapt, a_coef) see the
    dilation, which is all this function needs.
    """
    L_final = net.L.copy()
    r = np.full(net.m, 0.5 * r_seed)
    active = np.zeros(net.m, bool)
    try:
        for g in np.linspace(g0, 1.0, stages):
            net.L = L_final * g
            r, active, _, _ = adapt(net, b, r, tau0, **kw)
    finally:
        net.L = L_final
    return r, active


def summarize(net, r, active, b):
    """Support size, cycle rank, budget-invariant dissipation and budget.

    Evaluated on the full node set, so call it on a converged final network
    rather than on an intermediate growth stage.
    """
    G = nx.Graph([net.E[k] for k in np.flatnonzero(active)])
    beta = int(active.sum() - G.number_of_nodes() + nx.number_connected_components(G))
    return dict(edges=int(active.sum()), beta=beta,
                D_norm=net.d_norm(r, active, b), C=net.cost(r, active, b))
