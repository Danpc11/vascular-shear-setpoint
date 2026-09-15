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
from scipy.linalg import pinvh, cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


def _second_moment_flows(net, w, mu, idx, mask, active, sigma):
    """Mean and second moment of the edge flows under fluctuating sink demands.

    Cov(I) = sigma^2 sum_t v_t v_t^T with v_t = e_s - e_t, a sum of rank-one
    terms, one per sink. The variance of f_e is therefore sigma^2 times the sum
    over sinks of the squared unit source-to-sink flow through e, and with M the
    Laplacian inverse grounded at the source that flow is w_e (M[i,t] - M[j,t]).

    Written this way the step costs one Cholesky inverse of the grounded
    Laplacian plus an m x n array, instead of a dense pseudo-inverse and two
    m x n x n products. It agrees with the direct form to 5e-12 and is about ten
    times faster at 600 sinks, which matters because the fluctuating runs are
    thirty times slower per step than the steady ones and dominate a campaign.
    """
    Lap = (net.B[idx] * w) @ net.B[idx].T
    c = cho_factor(Lap[np.ix_(mask, mask)], lower=True)
    phi = np.zeros(len(idx))
    phi[mask] = cho_solve(c, mu[idx][mask])
    loc = np.full(net.n, -1)
    loc[idx] = np.arange(len(idx))
    ei, ej = loc[net.Ei[active]], loc[net.Ej[active]]
    wa = w[active]
    fmean = wa * (phi[ei] - phi[ej])
    if sigma <= 0:
        return fmean ** 2
    M = np.zeros((len(idx), len(idx)))
    M[np.ix_(mask, mask)] = cho_solve(c, np.eye(int(mask.sum())))
    H = wa[:, None] * (M[ei][:, mask] - M[ej][:, mask])
    return fmean ** 2 + sigma ** 2 * (H ** 2).sum(1)


def _cov_demand(net, nodes, sigma):
    """Cov(I) on the present nodes, kept for the Monte Carlo path and for tests."""
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
    mask = idx != net.src
    sinks = nodes.copy()
    sinks[net.src] = False
    steps, err = 0, np.inf
    for steps in range(1, max_steps + 1):
        w = net.conductance(r, active)
        if mc_samples == 0:
            f2 = _second_moment_flows(net, w, mu, idx, mask, active, sigma)
        else:
            Bs = net.B[idx][:, active] * w[active]
            Lap = (net.B[idx] * w) @ net.B[idx].T
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


def grow_peripheral(net, b, tau0, stages, r_seed, reactivate=True, seed_frac=0.5, **kw):
    """Sinks appear outward from the source in `stages` shells.

    reactivate=True  seeds every admissible edge that currently carries no
                     radius, including ones pruned in an earlier shell, so a
                     vessel lost early can be rebuilt as the tissue around it
                     grows. This is the protocol used in the campaign, and it
                     gives the rule a move that fixed-domain adaptation does not
                     have; read biologically it is angiogenesis into new tissue.
    reactivate=False seeds only edges never previously active, so a pruned
                     vessel stays pruned. This separates enlarging the domain
                     from allowing regrowth, and is the control against which
                     the default must be read.

    A newly admissible edge is seeded at `seed_frac` times the median radius of the
    current network, or times `r_seed` on the first shell when there is no network
    yet. Both are free parameters of the protocol and the answer depends on them:
    at 300 sinks, raising r_seed from 0.05 to 0.25 lowers the final dissipation by
    about 5 per cent. They must therefore be reported, and varied when the growth
    advantage is quoted.

    Returns radii, active mask and a per-shell diagnostic list.
    """
    dist = np.linalg.norm(net.pts - net.pts[net.src], axis=1)
    order = np.argsort(dist)
    order = order[order != net.src]
    nodes = np.zeros(net.n, bool)
    nodes[net.src] = True
    r = np.zeros(net.m)
    active = np.zeros(net.m, bool)
    # an edge counts as already given its chance once it has been seeded, not
    # only if it survived: otherwise every pruned edge is reseeded and the
    # control is vacuous.
    ever_seeded = np.zeros(net.m, bool)
    tol = kw.get("tol", 1e-6)
    diag = []
    for s in range(stages):
        lo = int(len(order) * s / stages)
        hi = int(len(order) * (s + 1) / stages)
        nodes[order[lo:hi]] = True
        fresh = (r == 0) & nodes[net.Ei] & nodes[net.Ej]
        if not reactivate:
            fresh &= ~ever_seeded
        ever_seeded |= fresh
        r[fresh] = seed_frac * (r_seed if not (r > 0).any() else np.median(r[r > 0]))
        r, active, st, e = adapt(net, b, r, tau0, nodes, **kw)
        ever_seeded |= active
        diag.append(dict(shell=s + 1, sinks=int(nodes.sum() - 1), steps=int(st),
                         final_err=float(e), converged=bool(e < tol),
                         edges=int(active.sum()), seeded=int(fresh.sum()),
                         seed_frac=float(seed_frac), r_seed=float(r_seed)))
    return r, active, diag


def grow_isotropic(net, b, tau0, stages, r_seed, g0=0.2, **kw):
    """All sinks present; the domain is dilated from g0 to 1 in `stages`.

    Scaling every length by a common factor is a symmetry of the dissipation and
    of the cost, so this should do nothing beyond adaptation on the fixed
    domain; it is the negative control for grow_peripheral. net.L is restored in
    a finally block and net.G keeps the unscaled lengths, so only routines
    reading net.L see the dilation.
    """
    L_final = net.L.copy()
    r = np.full(net.m, 0.5 * r_seed)
    active = np.zeros(net.m, bool)
    tol = kw.get("tol", 1e-6)
    diag = []
    try:
        for k, g in enumerate(np.linspace(g0, 1.0, stages)):
            net.L = L_final * g
            r, active, st, e = adapt(net, b, r, tau0, **kw)
            diag.append(dict(shell=k + 1, scale=float(g), steps=int(st),
                             final_err=float(e), converged=bool(e < tol),
                             edges=int(active.sum())))
    finally:
        net.L = L_final
    return r, active, diag


def adapt_to_budget(net, b, r0, tau0_guess, C_target, tol_C=2e-3, iters=6, **kw):
    """Adapt with tau_0 chosen so the rest point lands on a prescribed budget.

    The rule holds the set-point, which fixes the multiplier lambda and not the
    budget: raising sigma buys more material, so a sweep at fixed tau_0 compares
    networks of different cost and confounds loop count with how much was spent.

    At fixed topology stationarity gives w ∝ lambda^{-1/(alpha+1)} and hence
    C ∝ tau_0^{-2 alpha/(alpha+1)}, so one run fixes the constant and a single
    analytic correction lands on the target. The loop repeats that correction
    because the topology can change with tau_0, and each pass warm-starts from
    the previous radii, so this costs a few adapt calls rather than the twenty-odd
    a bisection would need.

    Returns (r, active, steps_total, err, tau0_used, C_reached, budget_ok) where
    tau0_used is the value that produced the radii returned, not the next
    correction, and budget_ok says whether |C/C_target - 1| < tol_C. A caller
    must require both err < tol and budget_ok before calling a run converged.
    """
    alpha = net.alpha(b)
    expo = (alpha + 1.0) / (2.0 * alpha)
    tau0 = float(tau0_guess)
    r_start = r0.copy()
    r = active = None
    steps_total, err = 0, np.inf
    tau0_used, C = tau0, np.nan
    for _ in range(iters):
        r, active, steps, err = adapt(net, b, r_start, tau0, **kw)
        steps_total += steps
        tau0_used = tau0                      # the value that produced these radii
        C = net.cost(r, active, b)
        if not np.isfinite(C) or C <= 0:
            break
        if abs(C / C_target - 1.0) < tol_C:
            break
        # C ∝ tau0^{-1/expo}: to scale C by C_target/C, scale tau0 by (C/C_target)^expo
        tau0 *= (C / C_target) ** expo
        r_start = r.copy()
    budget_ok = bool(np.isfinite(C) and C > 0 and abs(C / C_target - 1.0) < tol_C)
    return r, active, steps_total, err, float(tau0_used), float(C), budget_ok


def dissipation_under_load(net, r, active, sigma, nodes=None):
    """Expected dissipation under the fluctuating demand itself, tr(L^+ Q).

    summarize() reports D at the mean demand, which is not what a network
    adapted under fluctuations is minimizing. This returns the quantity that is,
    with Q = mean(I) mean(I)^T + Cov(I).
    """
    nodes = np.ones(net.n, bool) if nodes is None else nodes
    mu = net.unit_demand(nodes)
    w = net.conductance(r, active)
    if sigma <= 0:
        return net.dissipation(w, mu, nodes)[0]
    idx, S = _cov_demand(net, nodes, sigma)
    Lap = (net.B[idx] * w) @ net.B[idx].T
    Lp = pinvh(Lap)
    Q = np.outer(mu[idx], mu[idx]) + S
    return float(np.trace(Lp @ Q))


def summarize(net, r, active, b, sigma=0.0):
    """Support size, cycle rank, budget-invariant dissipation and budget.

    Evaluated on the full node set, so call it on a converged final network
    rather than on an intermediate growth stage.
    """
    G = nx.Graph([net.E[k] for k in np.flatnonzero(active)])
    beta = int(active.sum() - G.number_of_nodes() + nx.number_connected_components(G))
    C = net.cost(r, active, b)
    out = dict(edges=int(active.sum()), beta=beta,
               D_norm=net.d_norm(r, active, b), C=C)
    if sigma > 0:
        out["D_norm_load"] = (dissipation_under_load(net, r, active, sigma)
                              * C ** (1.0 / net.alpha(b)))
    return out
