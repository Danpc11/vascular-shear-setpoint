"""Point-to-area transport with a dissipation objective: a hierarchical H-tree against a
free-topology optimizer, and the wall shear stress each state implies.

Domain: 7x7 lattice, 4-neighbour candidate edges (unit length), rectilinear like the construct.
Source at the centre. Sinks: the 16 leaves of an order-2 H-tree, unit demand.
Other lattice nodes are pass-through (Steiner) nodes with zero demand.
Objective: dissipation  D = I^T L^+ I  with I = 16 e_s - sum_t e_t  (rank-one Q).
Cost:  sum_e l_e w_e^alpha = C0, with alpha = b/2 for fixed-length Poiseuille tubes.
"""
import numpy as np, networkx as nx, pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------- domain
COORDS = [(x, y) for x in range(-3, 4) for y in range(-3, 4)]
IDX = {c: i for i, c in enumerate(COORDS)}
N = len(COORDS)
SRC = IDX[(0, 0)]
LEAVES = [IDX[(sx*a, sy*b)] for a in (1, 3) for b in (1, 3) for sx in (-1, 1) for sy in (-1, 1)]
LEAVES = sorted(set(LEAVES))                      # 16 leaves of the order-2 H-tree
DEMAND = np.zeros(N); DEMAND[LEAVES] = 1.0
I_VEC = -DEMAND.copy(); I_VEC[SRC] = DEMAND.sum()  # source injects, sinks withdraw

def king_edges():
    E = []
    for (x, y) in COORDS:
        for dx, dy in ((1, 0), (0, 1)):
            if (x+dx, y+dy) in IDX:
                i, j = IDX[(x, y)], IDX[(x+dx, y+dy)]
                E.append((min(i, j), max(i, j)))
    return sorted(set(E))
EDGES = king_edges(); M = len(EDGES)
LENGTH = np.array([np.hypot(*(np.subtract(COORDS[i], COORDS[j]))) for i, j in EDGES])
B = np.zeros((N, M))
for _k, (_i, _j) in enumerate(EDGES): B[_i, _k], B[_j, _k] = 1.0, -1.0

def h_tree_edges():
    """Order-2 H-tree rooted at the centre, expressed as unit lattice edges."""
    segs = []
    def bar(p, q):                       # straight bar between lattice points, unit steps
        (x0, y0), (x1, y1) = p, q
        steps = max(abs(x1-x0), abs(y1-y0)); dx = np.sign(x1-x0); dy = np.sign(y1-y0)
        for k in range(steps):
            a = (x0+dx*k, y0+dy*k); b = (x0+dx*(k+1), y0+dy*(k+1))
            segs.append((min(IDX[a], IDX[b]), max(IDX[a], IDX[b])))
    bar((0, 0), (-2, 0)); bar((0, 0), (2, 0))                   # root bar
    for sx in (-1, 1):
        bar((2*sx, 0), (2*sx, -2)); bar((2*sx, 0), (2*sx, 2))    # first-level verticals
        for sy in (-1, 1):
            c = (2*sx, 2*sy)
            bar(c, (2*sx-1, 2*sy)); bar(c, (2*sx+1, 2*sy))       # second-level bars
            for ex in (-1, 1):
                bar((2*sx+ex, 2*sy), (2*sx+ex, 2*sy-1)); bar((2*sx+ex, 2*sy), (2*sx+ex, 2*sy+1))
    return sorted(set(segs))
HTREE = h_tree_edges()

# ---------------------------------------------------------------- objective
REQUIRED = set(LEAVES) | {SRC}

def dissipation_grad(w, tol=1e-12):
    """D = I^T L^+ I on the component of the active support that carries the source
    and every sink. Pass-through nodes left outside the support cost nothing, so
    connectivity is required only for the demand-carrying nodes."""
    act = np.flatnonzero(w > tol)
    if act.size == 0: return 1e30, np.zeros_like(w)
    G = nx.Graph([EDGES[k] for k in act])
    if not REQUIRED <= set(G.nodes): return 1e30, np.zeros_like(w)
    comp = nx.node_connected_component(G, SRC)
    if not REQUIRED <= comp: return 1e30, np.zeros_like(w)
    nodes = sorted(comp); loc = {v: i for i, v in enumerate(nodes)}
    in_comp = np.array([EDGES[k][0] in comp and EDGES[k][1] in comp for k in range(M)])
    ks = np.flatnonzero(in_comp)
    Bs = np.zeros((len(nodes), ks.size))
    for c, k in enumerate(ks):
        i, j = EDGES[k]; Bs[loc[i], c], Bs[loc[j], c] = 1.0, -1.0
    L = (Bs * w[ks]) @ Bs.T
    vals, vecs = np.linalg.eigh(L)
    if vals[1] < 1e-11: return 1e30, np.zeros_like(w)
    Lp = (vecs[:, 1:] / vals[1:]) @ vecs[:, 1:].T
    Is = I_VEC[nodes]; phi = Lp @ Is
    g = np.zeros_like(w); g[ks] = -(Bs.T @ phi)**2
    return float(Is @ phi), g

def tree_flows(tree_edges):
    """Flow on each tree edge = downstream demand, oriented away from the source."""
    T = nx.Graph(tree_edges); assert nx.is_tree(T) and nx.is_connected(T)
    order = list(nx.dfs_preorder_nodes(T, SRC)); parent = dict(nx.dfs_predecessors(T, SRC))
    sub = {v: DEMAND[v] for v in T.nodes}
    for v in reversed(order):
        if v in parent: sub[parent[v]] += sub[v]
    return {e: sub[e[0] if parent.get(e[0]) == e[1] else e[1]] for e in tree_edges}

def tree_optimum(tree_edges, alpha, C0=1.0):
    """Closed-form allocation on a fixed tree: w_e ∝ (f_e^2/a_e)^{1/(alpha+1)}."""
    f = tree_flows(tree_edges); idx = {e: k for k, e in enumerate(EDGES)}
    w = np.zeros(M)
    for e, fe in f.items():
        if fe > 0: w[idx[e]] = (fe**2 / LENGTH[idx[e]])**(1.0/(alpha+1))
    scale = (C0 / np.sum(LENGTH * w**alpha))**(1.0/alpha); w *= scale
    D, _ = dissipation_grad(w)
    return w, D, f

# ---------------------------------------------------------------- optimizer
def solve_free(alpha, C0=1.0, n_random=24, seed=7, extra=()):
    rng = np.random.default_rng(seed)
    def unpack(z):
        z = np.clip(z - z.max(), -45, 0); p = np.exp(z); p /= p.sum()
        return p, (C0*p/LENGTH)**(1.0/alpha)
    def obj(z):
        p, w = unpack(z); D, gw = dissipation_grad(w)
        if D > 1e25: return 1e25, np.zeros_like(z)
        gp = gw * w/(alpha*np.maximum(p, 1e-300)); return D, p*(gp - gp@p)
    starts = [np.full(M, 1.0/M)]
    for tree in (HTREE, list(nx.bfs_tree(nx.Graph(EDGES), SRC).to_undirected().edges())):
        q = np.full(M, 1e-7)
        for e in tree: q[EDGES.index((min(e), max(e)))] = 1.0
        starts.append(q/q.sum())
    for _ in range(n_random): starts.append(rng.dirichlet(np.full(M, 0.35)))
    for q in extra: starts.append(np.asarray(q))
    # random shortest-path trees over the demand nodes: cheap, sparse, structured
    G0 = nx.Graph(); G0.add_edges_from(EDGES)
    for k in range(10):
        for (u, v) in G0.edges: G0[u][v]["w"] = LENGTH[EDGES.index((min(u,v),max(u,v)))]*(1+0.3*rng.random())
        paths = nx.single_source_dijkstra_path(G0, SRC, weight="w")
        tree = set()
        for t in LEAVES:
            p = paths[t]; tree |= {(min(a,b),max(a,b)) for a,b in zip(p[:-1],p[1:])}
        q = np.full(M, 1e-7)
        for e in tree: q[EDGES.index(e)] = 1.0
        starts.append(q/q.sum())
    def refine(keep):
        """optimise on a support, prune, repeat until the support is stable"""
        for _ in range(8):
            def obj_s(zs):
                ps = np.exp(zs - zs.max()); ps /= ps.sum(); ws = np.zeros(M)
                ws[keep] = (C0*ps/LENGTH[keep])**(1.0/alpha); D, gw = dissipation_grad(ws)
                if D > 1e25: return 1e25, np.zeros_like(zs)
                gp = gw[keep]*ws[keep]/(alpha*np.maximum(ps, 1e-300)); return D, ps*(gp - gp@ps)
            rs = minimize(obj_s, np.zeros(int(keep.sum())), jac=True, method="BFGS",
                          options={"maxiter": 3000, "gtol": 1e-10})
            ps = np.exp(rs.x - rs.x.max()); ps /= ps.sum()
            new_keep = keep.copy(); new_keep[np.flatnonzero(keep)[ps <= 1e-5]] = False
            if new_keep.sum() == keep.sum():
                w = np.zeros(M); w[keep] = (C0*ps/LENGTH[keep])**(1.0/alpha)
                return dissipation_grad(w)[0], w
            keep = new_keep
        w = np.zeros(M); w[keep] = (C0*ps/LENGTH[keep])**(1.0/alpha)
        return dissipation_grad(w)[0], w
    best = (np.inf, None)
    for q0 in starts:
        z0 = np.log(np.maximum(q0, 1e-15))
        r = minimize(obj, z0, jac=True, method="L-BFGS-B", bounds=[(-25, 5)]*M,
                     options={"maxiter": 2500, "ftol": 1e-12, "gtol": 1e-8})
        p, w = unpack(r.x); keep = p > 1e-5
        if dissipation_grad(np.where(keep, w, 0.0))[0] > 1e25: continue
        D, w = refine(keep)
        if D < best[0]: best = (D, w)
    return best

def describe(w):
    act = [e for e, x in zip(EDGES, w) if x > 0]
    G = nx.Graph(act); nodes = set(G.nodes)
    beta = len(act) - len(nodes) + nx.number_connected_components(G)
    return len(act), beta, act

def shear(w, active):
    """tau ∝ f/r^3 with r from Poiseuille w = r^4/l (units dropped); slope of log tau vs log r."""
    if not active: return None
    T = nx.Graph(active)
    if not nx.is_tree(T): return None
    f = tree_flows(active); idx = {e: k for k, e in enumerate(EDGES)}
    active = [e for e in active if f[e] > 0]
    r = np.array([(w[idx[e]]*LENGTH[idx[e]])**0.25 for e in active])
    fl = np.array([f[e] for e in active]); tau = fl / r**3
    slope = np.polyfit(np.log(r), np.log(tau), 1)[0]
    return dict(slope=slope, tau_ratio=tau.max()/tau.min(), r_range=r.max()/r.min(),
                r=r, tau=tau, l=np.array([LENGTH[idx[e]] for e in active]))

if __name__ == "__main__":
    print(f"lattice {N} nodes, {M} candidate edges, {len(LEAVES)} sinks, H-tree {len(HTREE)} edges")
    rows = []
    for b in (1.0, 0.75, 2/3):
        alpha = b/2
        wH, DH, _ = tree_optimum(HTREE, alpha)
        bfs = [(min(e), max(e)) for e in nx.bfs_tree(nx.Graph(EDGES), SRC).edges()]
        wB, DB, _ = tree_optimum(bfs, alpha)
        Dfree, wfree = solve_free(alpha)
        na, beta, act = describe(wfree)
        same = set(act) == set(HTREE)
        sH, sF = shear(wH, HTREE), shear(wfree, act)
        rows.append(dict(b=b, alpha=alpha, D_Htree=DH, D_bfs=DB, D_free=Dfree,
                         eta_Htree=Dfree/DH, eta_bfs=Dfree/DB, free_active=na, free_beta=beta,
                         free_is_Htree=same, shear_slope_pred=2*alpha-1,
                         shear_slope_H=sH["slope"], shear_slope_free=sF["slope"] if sF else np.nan,
                         tau_ratio_H=sH["tau_ratio"], r_range_H=sH["r_range"]))
        print(f"\nb={b:.3f} alpha={alpha:.3f}")
        print(f"  D  H-tree={DH:.5f}  BFS tree={DB:.5f}  free optimum={Dfree:.5f}"
              f"   -> eta_top(H-tree)={Dfree/DH:.4f}  eta_top(BFS)={Dfree/DB:.4f}")
        print(f"  free state: {na} active edges, beta={beta}, identical to H-tree: {same}")
        print(f"  shear slope d ln tau/d ln r: predicted {2*alpha-1:+.3f}, H-tree {sH['slope']:+.3f}, "
              f"free {sF['slope'] if sF else float('nan'):+.3f};  tau_max/tau_min on H-tree = {sH['tau_ratio']:.2f} "
              f"over r range {sH['r_range']:.2f}")
    pd.DataFrame(rows).to_csv("vascular_results.tsv", sep="\t", index=False)
    np.save("htree_edges.npy", np.array(HTREE))
