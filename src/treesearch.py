"""Search over trees with the exact closed-form allocation."""
import numpy as np, networkx as nx

def spt(net, weight):
    for (i, j), wt in zip(net.E, weight): net.G[i][j]["wt"] = wt
    paths = nx.single_source_dijkstra_path(net.G, net.src, weight="wt")
    tree = set()
    for p in paths.values():
        for a, b in zip(p[:-1], p[1:]): tree.add((min(a, b), max(a, b)))
    return sorted(tree)

def reweighted_spt(net, b, seeds=8, iters=30, seed=5, jitter=0.5):
    """Fixed-point heuristic for concave-cost network design: shortest-path trees
    under edge weights equal to the marginal concave flow cost
    a_e^{1/(alpha+1)} f_e^{2alpha/(alpha+1)-1}, iterated from jittered seeds."""
    rng = np.random.default_rng(seed); alpha = net.alpha(b); q = 2 * alpha / (alpha + 1)
    a = net.a_coef(b); best = None
    for _ in range(seeds):
        tree = spt(net, net.L * (1 + jitter * rng.random(net.m))); Dprev = np.inf
        for _ in range(iters):
            sol = net.allocate(tree, b)
            if sol["D"] > Dprev - 1e-9: break
            Dprev = sol["D"]
            f = np.full(net.m, 0.05)
            for e, fe in zip(sol["edges"], sol["f"]): f[net.idx[e]] = fe
            tree = spt(net, a ** (1 / (alpha + 1)) * f ** (q - 1))
        sol = net.allocate(tree, b)
        if best is None or sol["D"] < best["D"]: best = sol
    return best

def edge_swap(net, tree, b, sweeps=3, rng=None):
    """Descent by single edge exchange keeping a tree; expensive, use sparingly."""
    best = list(tree); Db = net.allocate(best, b)["D"]
    for _ in range(sweeps):
        improved = False
        for e in list(best):
            if e not in best: continue
            T = nx.Graph(best); T.remove_edge(*e); side = nx.node_connected_component(T, net.src)
            for (u, v) in net.E:
                if (u in side) != (v in side):
                    cand = [x for x in best if x != e] + [(min(u, v), max(u, v))]
                    Tc = nx.Graph(cand)
                    if not (nx.is_tree(Tc) and Tc.number_of_nodes() == net.n): continue
                    D = net.allocate(cand, b)["D"]
                    if D < Db - 1e-9: best, Db, improved = cand, D, True
        if not improved: break
    return net.allocate(best, b)
