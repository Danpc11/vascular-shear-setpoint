"""Controls, robustness tests and verifications for every claim in the Letter.

Each function returns a dict with the measured quantity, the value the theory
requires, and a pass/fail at a stated tolerance, so a run of run_checks.py is a
machine-readable audit of the manuscript.

Claims covered
  C1  gradient of the dissipation against central differences
  C2  the closed-form tree allocation is the constrained minimum on that tree
  C3  set-point constancy: tau/(r^{b-1} l^{(b-1)/2}) is one number on an optimal tree
  C4  the invariant D_e/c_e = lambda b/2 and D_e ∝ m_e^b
  C5  Murray recovery: b=1 gives uniform tau and uniform D/V; b<1 does not
  C6  joint regression recovers (b-1) and (b-1)/2; regression on r alone is biased
  C7  Lyapunov: measured dF/dt equals -2 b lambda kappa sum c (z-1)^2 (z+1)
  C8  D_norm is invariant under rescaling all radii (budget independence)
  C9  rest points: optimal tree and adapted tree both satisfy z=1 yet differ in D
  C10 exact certificate on small domains by enumerating all spanning trees
  C11 robustness to kappa, prune threshold, tolerance, integration step
  C12 robustness to domain seed and candidate-graph type
  C13 convergence of fluctuating-demand runs (err below tol, loop count stable)
"""
import itertools, numpy as np, networkx as nx
from domain import make_domain
from flows import Net, shear_stats
from treesearch import reweighted_spt
from remodel import adapt, summarize

def _ok(name, measured, required, tol, rel=True, extra=None):
    d = abs(measured - required)
    if rel and required != 0: d /= abs(required)
    r = dict(check=name, measured=measured, required=required, deviation=d,
             tol=tol, passed=bool(d <= tol))
    if extra: r.update(extra)
    return r

# ---------------------------------------------------------------- C1
def c1_gradient(net, b, h=1e-3, n_edges=25, seed=0, tol=1e-5):
    """Central differences of D against -Delta p^2, Richardson-extrapolated.

    A plain central difference is limited by cancellation: the per-edge
    sensitivity is a small fraction of D, so the usable step grows with the
    domain. Richardson extrapolation of two central differences,
    (4 cd(h/2) - cd(h))/3, has O(h^4) truncation error and stays below 1e-5 from
    150 to 900 sinks at h = 1e-3, which a single difference at a fixed step does
    not. This check tests the check as much as the gradient.
    """
    rng = np.random.default_rng(seed)
    tree = reweighted_spt(net, b, seeds=2, iters=8)["edges"]
    w = np.zeros(net.m)
    ks = np.array([net.idx[e] for e in tree])
    w[ks] = 1.0 + 0.5 * rng.random(ks.size)
    nodes = np.ones(net.n, bool)
    mu = net.unit_demand(nodes)
    f, phi = net.flows(w, mu, nodes)
    ana = -(net.B.T @ phi) ** 2

    def cd(k, step):
        wp = w.copy(); wp[k] *= (1 + step)
        wm = w.copy(); wm[k] *= (1 - step)
        return (net.dissipation(wp, mu, nodes)[0]
                - net.dissipation(wm, mu, nodes)[0]) / (2 * step * w[k])

    worst = 0.0
    for k in rng.choice(ks, size=min(n_edges, ks.size), replace=False):
        rich = (4 * cd(k, h / 2) - cd(k, h)) / 3
        worst = max(worst, abs(rich - ana[k]) / abs(ana[k]))
    return _ok("C1 dissipation gradient", worst, 0.0, tol, rel=False)


# ---------------------------------------------------------------- C2
def c2_allocation(net, b, seed=0, trials=200):
    """random feasible reallocations on the same tree must not lower D."""
    rng = np.random.default_rng(seed); alpha = net.alpha(b)
    sol = reweighted_spt(net, b); ks = np.array([net.idx[e] for e in sol["edges"]])
    a = net.a_coef(b)[ks]; C0 = float(np.sum(a * sol["w"] ** alpha))
    best_gain = 0.0
    for _ in range(trials):
        p = rng.dirichlet(np.full(ks.size, 60.0))
        w = (C0 * p / a) ** (1 / alpha)
        D = float(np.sum(sol["f"] ** 2 / w))
        best_gain = max(best_gain, (sol["D"] - D) / sol["D"])
    return _ok("C2 tree allocation is minimal", best_gain, 0.0, 1e-9, rel=False,
               extra=dict(trials=trials))

# ---------------------------------------------------------------- C3, C4, C5
def c3_c4_c5(net, b):
    sol = reweighted_spt(net, b); alpha = net.alpha(b)
    f, l, w, r = sol["f"], sol["l"], sol["w"], sol["r"]
    tau = f / r ** 3
    t0 = tau / (r ** (b - 1) * l ** ((b - 1) / 2))
    D = f ** 2 / w; c = (l ** (3 * alpha)) * w ** alpha; m = r ** 2 * l
    cv = lambda x: float(x.std() / x.mean())
    out = [_ok("C3 set-point constancy", cv(t0), 0.0, 1e-12, rel=False, extra=dict(b=b)),
           _ok("C4 invariant D/c uniform", cv(D / c), 0.0, 1e-12, rel=False, extra=dict(b=b)),
           _ok("C4 invariant D/m^b uniform", cv(D / m ** b), 0.0, 1e-12, rel=False, extra=dict(b=b))]
    if abs(b - 1) < 1e-12:
        out += [_ok("C5 b=1 uniform tau", cv(tau), 0.0, 1e-10, rel=False),
                _ok("C5 b=1 uniform D/V", cv(D / m), 0.0, 1e-10, rel=False)]
    else:
        out += [dict(check="C5 b<1 tau NOT uniform", measured=cv(tau), required=">0",
                     deviation=np.nan, tol=np.nan, passed=bool(cv(tau) > 0.05), b=b),
                dict(check="C5 b<1 D/V NOT uniform", measured=cv(D / m), required=">0",
                     deviation=np.nan, tol=np.nan, passed=bool(cv(D / m) > 0.05), b=b)]
    return out

# ---------------------------------------------------------------- C6
def c6_regression(net, b):
    sol = reweighted_spt(net, b); st = shear_stats(sol, b)
    return [_ok("C6 joint slope on ln r", st["slope_r_joint"], b - 1, 2e-3, extra=dict(b=b)),
            _ok("C6 joint slope on ln l", st["slope_l_joint"], (b - 1) / 2, 2e-3, extra=dict(b=b)),
            dict(check="C6 r-only slope is biased", measured=st["slope_r_only"], required=b - 1,
                 deviation=abs(st["slope_r_only"] - (b - 1)), tol=np.nan,
                 passed=bool(b == 1 or abs(st["slope_r_only"] - (b - 1)) > 1e-3), b=b)]

# ---------------------------------------------------------------- C7
def c7_lyapunov(net, b, lam=None, kappa=0.05, steps=12, seed=0):
    """measured dF/dt against the analytic identity, away from the rest point."""
    rng = np.random.default_rng(seed); alpha = net.alpha(b)
    sol = reweighted_spt(net, b); ks = np.array([net.idx[e] for e in sol["edges"]])
    a = net.a_coef(b); nodes = np.ones(net.n, bool); mu = net.unit_demand(nodes)
    # lambda consistent with this budget, read off the optimal tree
    w0 = np.zeros(net.m); w0[ks] = sol["w"]
    f0, phi0 = net.flows(w0, mu, nodes); dp0 = (net.B.T @ phi0)[ks]
    lam = float(np.median(dp0 ** 2 / (alpha * a[ks] * sol["w"] ** (alpha - 1)))) if lam is None else lam
    # tau0 is read off the optimal tree rather than from a unit-dependent constant:
    # with dropped Poiseuille constants Delta p = tau*l/r here, so tau0^2 = lam*b/2,
    # whereas the physical convention Delta p = 2*tau*l/r gives lam*b/8. Only tau0 ∝ sqrt(lam) is
    # convention-independent, so the code uses the measured value and the check below
    # confirms the two agree.
    tau0 = float(np.median((sol["f"] / sol["r"] ** 3) /
                           (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    r = np.zeros(net.m); r[ks] = sol["r"] * np.exp(0.25 * rng.standard_normal(ks.size))  # push off the rest point
    worst = 0.0
    for _ in range(steps):
        w = np.zeros(net.m); w[ks] = r[ks] ** 4 / net.L[ks]
        f, phi = net.flows(w, mu, nodes); dp = (net.B.T @ phi)[ks]
        F = float(np.sum(f[ks] ** 2 / w[ks]) + lam * np.sum(a[ks] * w[ks] ** alpha))
        tau = np.abs(f[ks]) / r[ks] ** 3
        tset = tau0 * r[ks] ** (b - 1) * net.L[ks] ** ((b - 1) / 2)
        z = tau / tset; c = a[ks] * w[ks] ** alpha
        analytic = -2 * b * lam * kappa * float(np.sum(c * (z - 1) ** 2 * (z + 1)))
        dt = 1e-6
        r2 = r.copy(); r2[ks] = r[ks] * np.exp(kappa * (z - 1) * dt)
        w2 = np.zeros(net.m); w2[ks] = r2[ks] ** 4 / net.L[ks]
        f2, _ = net.flows(w2, mu, nodes)
        F2 = float(np.sum(f2[ks] ** 2 / w2[ks]) + lam * np.sum(a[ks] * w2[ks] ** alpha))
        measured = (F2 - F) / dt
        worst = max(worst, abs(measured - analytic) / abs(analytic))
        r[ks] = r[ks] * np.exp(kappa * (z - 1) * 0.05)          # advance a little and retest
    ratio = tau0 / np.sqrt(lam * b / 2.0)
    return [_ok("C7 Lyapunov identity", worst, 0.0, 1e-4, rel=False, extra=dict(b=b)),
            _ok("C7 tau0 = sqrt(lambda b/2) in code units", ratio, 1.0, 1e-9, extra=dict(b=b))]

# ---------------------------------------------------------------- C8
def c8_budget_invariance(net, b, scale=3.7):
    sol = reweighted_spt(net, b)
    r = sol["r_full"]; act = sol["active"]
    d1 = net.d_norm(r, act, b); d2 = net.d_norm(r * scale, act, b)
    return _ok("C8 D_norm budget-invariant", d2, d1, 1e-10, extra=dict(b=b, scale=scale))

# ---------------------------------------------------------------- C9
def c9_rest_points(net, b, kappa=0.2, max_steps=20000, tol=1e-6, seed=0):
    """the optimal tree and the tree reached from a dense start are both rest points."""
    sol = reweighted_spt(net, b)
    t0 = float(np.median((sol["f"] / sol["r"] ** 3) /
                         (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    r_dense = np.full(net.m, float(sol["r"].mean()))
    r1, a1, s1, e1 = adapt(net, b, r_dense, t0, kappa=kappa, max_steps=max_steps, tol=tol, seed=seed)
    S1 = summarize(net, r1, a1, b)
    r_opt = sol["r_full"].copy()
    _, a2, s2, e2 = adapt(net, b, r_opt, t0, kappa=kappa, max_steps=50, tol=tol)
    ratio = S1["D_norm"] / sol["D_norm"]
    shared = len({net.E[k] for k in np.flatnonzero(a1)} & set(sol["edges"]))
    return [_ok("C9 optimal tree is a rest point", e2, 0.0, tol, rel=False, extra=dict(b=b, steps=s2)),
            _ok("C9 adapted tree is a rest point", e1, 0.0, tol, rel=False, extra=dict(b=b, steps=s1)),
            dict(check="C9 the two rest points differ", measured=ratio, required=">1",
                 deviation=ratio - 1, tol=np.nan, passed=bool(ratio > 1.01), b=b,
                 shared_edges=shared, adapted_edges=S1["edges"])]

# ---------------------------------------------------------------- C10
def c10_certificate(b, n_sinks=7, seed=1, kind="knn", knn=3, max_trees=200000):
    """Exact optimum on a small domain by enumerating every spanning tree.

    Gives a genuine certificate where one is affordable: if the heuristic and
    the dynamics both return the enumerated optimum, the 'best candidate'
    language is justified at that size.
    """
    pts, E, L = make_domain(n_sinks, seed, kind, knn); net = Net(pts, E, L)
    G = nx.Graph(); G.add_edges_from(net.E)
    if not nx.is_connected(G): return dict(check="C10 exhaustive certificate", passed=False,
                                           measured=np.nan, required=np.nan, deviation=np.nan,
                                           tol=np.nan, note="candidate graph disconnected")
    best = (np.inf, None); count = 0
    for T in nx.SpanningTreeIterator(G):
        count += 1
        if count > max_trees: break
        edges = sorted((min(u, v), max(u, v)) for u, v in T.edges())
        D = net.allocate(edges, b)["D_norm"]
        if D < best[0]: best = (D, edges)
    heur = reweighted_spt(net, b)["D_norm"]
    sol = net.allocate(best[1], b)
    t0 = float(np.median((sol["f"] / sol["r"] ** 3) / (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    rd = np.full(net.m, float(sol["r"].mean()))
    rr, aa, _, _ = adapt(net, b, rd, t0, max_steps=20000, tol=1e-7)
    dyn = summarize(net, rr, aa, b)["D_norm"]
    # reported, not asserted: the point of the enumeration is to measure how far the
    # heuristic and the dynamics sit from a certified optimum, which is the quantity the
    # manuscript cannot otherwise bound. "passed" only records that the enumeration was
    # exhaustive and that neither method beat the certified optimum.
    exhaustive = count <= max_trees
    return [dict(check="C10 enumeration exhaustive", measured=float(count), required=float(max_trees),
                 deviation=0.0 if exhaustive else 1.0, tol=0.5, passed=bool(exhaustive),
                 b=b, n_sinks=n_sinks, trees_enumerated=count),
            dict(check="C10 heuristic gap to certified optimum", measured=heur / best[0], required=1.0,
                 deviation=heur / best[0] - 1, tol=np.nan, passed=bool(heur >= best[0] * (1 - 1e-9)),
                 b=b, n_sinks=n_sinks, trees_enumerated=count),
            dict(check="C10 local-rule gap to certified optimum", measured=dyn / best[0], required=1.0,
                 deviation=dyn / best[0] - 1, tol=np.nan, passed=bool(dyn >= best[0] * (1 - 1e-9)),
                 b=b, n_sinks=n_sinks, trees_enumerated=count)]

# ---------------------------------------------------------------- C11
def c11_numerics(net, b, seed=0, clip=0.02):
    """Robustness to parameters that should NOT change the answer.

    kappa rescales time, and the pruning cut and the tolerance only decide when
    to stop, so all three must leave the network the rule settles on unchanged.
    The step cap is deliberately excluded: it is a discretisation parameter and
    is tested for convergence in c14 instead.
    """
    sol = reweighted_spt(net, b)
    t0 = float(np.median((sol["f"] / sol["r"] ** 3) /
                         (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    r0 = np.full(net.m, float(sol["r"].mean()))
    base = None
    out = []
    for tag, kw in [("kappa=0.2", dict(kappa=0.2)), ("kappa=0.05", dict(kappa=0.05)),
                    ("kappa=0.4", dict(kappa=0.4)), ("prune=1e-2", dict(prune=1e-2)),
                    ("prune=1e-4", dict(prune=1e-4)), ("tol=1e-8", dict(tol=1e-8))]:
        kw.setdefault("max_steps", 60000)
        kw.setdefault("tol", 1e-6)
        kw["clip"] = clip
        r, a, s_, e = adapt(net, b, r0.copy(), t0, seed=seed, **kw)
        S = summarize(net, r, a, b)
        if base is None:
            base = S["D_norm"]
        out.append(dict(check=f"C11 robustness {tag}", measured=S["D_norm"] / base, required=1.0,
                        deviation=abs(S["D_norm"] / base - 1), tol=5e-2,
                        passed=bool(abs(S["D_norm"] / base - 1) <= 5e-2),
                        b=b, clip=clip, edges=S["edges"], beta=S["beta"], steps=s_))
    return out


def c14_step_convergence(net, b, clips=(0.15, 0.05, 0.02, 0.01, 0.005), seed=0,
                         tol=3e-2):
    """Is the endpoint converged in the integration step?

    Reported, not asserted for every step: the pass flag is whether the two
    finest steps agree to `tol`, which is the statement that the campaign value
    is small enough. Coarser steps are recorded so the drift is visible.
    """
    sol = reweighted_spt(net, b)
    t0 = float(np.median((sol["f"] / sol["r"] ** 3) /
                         (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    r0 = np.full(net.m, float(sol["r"].mean()))
    vals, out = {}, []
    for c in clips:
        r, a, s_, e = adapt(net, b, r0.copy(), t0, clip=c, max_steps=80000, tol=1e-6, seed=seed)
        S = summarize(net, r, a, b)
        vals[c] = S["D_norm"] / sol["D_norm"]
        out.append(dict(check=f"C14 step clip={c}", measured=vals[c], required=np.nan,
                        deviation=np.nan, tol=np.nan, passed=True,
                        b=b, clip=c, steps=s_, edges=S["edges"]))
    fine = sorted(clips)[:2]
    drift = abs(vals[fine[0]] / vals[fine[1]] - 1)
    out.append(dict(check=f"C14 converged at clip={fine[1]}", measured=drift, required=0.0,
                    deviation=drift, tol=tol, passed=bool(drift <= tol), b=b,
                    coarse_to_fine=vals[max(clips)] / vals[min(clips)]))
    return out


# ---------------------------------------------------------------- C12
def c12_domains(b, n_sinks=200, seeds=(1, 2, 3), kinds=("delaunay", "knn")):
    out = []
    for kind in kinds:
        for sd in seeds:
            pts, E, L = make_domain(n_sinks, sd, kind); net = Net(pts, E, L)
            sol = reweighted_spt(net, b); st = shear_stats(sol, b)
            out.append(dict(check=f"C12 exponents on {kind} seed {sd}",
                            measured=st["slope_r_joint"], required=b - 1,
                            deviation=abs(st["slope_r_joint"] - (b - 1)), tol=5e-3,
                            passed=bool(abs(st["slope_r_joint"] - (b - 1)) <= 5e-3),
                            b=b, kind=kind, domain_seed=sd, edges=len(sol["edges"])))
    return out

# ---------------------------------------------------------------- C13
def c13_fluctuation(net, b, sigmas=(0.0, 1.0, 2.0), max_steps=20000, tol=1e-5, seed=0):
    sol = reweighted_spt(net, b)
    t0 = float(np.median((sol["f"] / sol["r"] ** 3) / (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))))
    out = []
    for sg in sigmas:
        r0 = np.full(net.m, float(sol["r"].mean()))
        r, a, s, e = adapt(net, b, r0, t0, sigma=sg, max_steps=max_steps, tol=tol, seed=seed)
        S = summarize(net, r, a, b)
        out.append(dict(check=f"C13 fluctuation sigma={sg}", measured=e, required=0.0,
                        deviation=e, tol=tol, passed=bool(e <= tol),
                        b=b, sigma=sg, beta=S["beta"], edges=S["edges"], steps=s))
    return out
