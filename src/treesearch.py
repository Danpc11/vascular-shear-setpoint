"""Search over trees with the exact closed-form allocation.

For a single flow pattern and a concave cost the minimizer is supported on a
tree (Banavar et al., PRL 84, 4745), so the search is a search over trees and
each candidate is evaluated exactly through w_e ∝ (f_e^2/a_e)^{1/(alpha+1)}.

These are best trees found, not certified optima. checks.c10_certificate
enumerates every spanning tree on a small domain and measures how far this
heuristic sits from the true optimum.
"""
import numpy as np
import networkx as nx


def spt(net, weight):
    """Shortest-path tree from the source under the given edge weights.

    Every node carries demand, so the tree spans all of them.
    """
    for (i, j), wt in zip(net.E, weight):
        net.G[i][j]["wt"] = wt
    paths = nx.single_source_dijkstra_path(net.G, net.src, weight="wt")
    tree = set()
    for p in paths.values():
        for a, b in zip(p[:-1], p[1:]):
            tree.add((min(a, b), max(a, b)))
    return sorted(tree)


def reweighted_spt(net, b, seeds=8, iters=30, seed=5, jitter=0.5, f_floor=0.05,
                   tol=1e-9):
    """Fixed-point heuristic for concave-cost network design.

    Shortest-path trees under edge weights equal to the marginal concave flow
    cost a_e^{1/(alpha+1)} f_e^{2 alpha/(alpha+1) - 1}, iterated from jittered
    seeds. Edges absent from the current tree carry no flow and would get an
    infinite weight, so they are assigned a small positive flow `f_floor`;
    raising it makes the next tree more willing to adopt unused edges.

    The best tree seen at any iteration is kept. The iteration is not monotone
    in general, so returning the last tree could return a worse one; that was a
    latent defect of an earlier version of this function.
    """
    rng = np.random.default_rng(seed)
    alpha = net.alpha(b)
    q = 2 * alpha / (alpha + 1)
    a = net.a_coef(b)
    best = None
    for _ in range(seeds):
        tree = spt(net, net.L * (1 + jitter * rng.random(net.m)))
        prev = np.inf
        for _ in range(iters):
            sol = net.allocate(tree, b)
            if best is None or sol["D"] < best["D"]:
                best = sol                      # keep the best seen, not the last
            if sol["D"] > prev - tol:
                break                           # fixed point, or no longer improving
            prev = sol["D"]
            f = np.full(net.m, f_floor)
            for e, fe in zip(sol["edges"], sol["f"]):
                f[net.idx[e]] = fe
            tree = spt(net, a ** (1 / (alpha + 1)) * f ** (q - 1))
    return best


def edge_swap(net, tree, b, sweeps=3, tol=1e-9):
    """Descent by single edge exchange, keeping a spanning tree throughout.

    Expensive: each sweep is O(|tree| x |E|) allocations. Used to polish the
    best few seeds rather than as the main search.
    """
    best = list(tree)
    best_D = net.allocate(best, b)["D"]
    for _ in range(sweeps):
        improved = False
        for e in list(best):
            if e not in best:
                continue
            T = nx.Graph(best)
            T.remove_edge(*e)
            side = nx.node_connected_component(T, net.src)
            for (u, v) in net.E:
                if (u in side) == (v in side):
                    continue                    # does not reconnect the cut
                cand = [x for x in best if x != e] + [(min(u, v), max(u, v))]
                Tc = nx.Graph(cand)
                if not (nx.is_tree(Tc) and Tc.number_of_nodes() == net.n):
                    continue
                D = net.allocate(cand, b)["D"]
                if D < best_D - tol:
                    best, best_D, improved = cand, D, True
        if not improved:
            break
    return net.allocate(best, b)
