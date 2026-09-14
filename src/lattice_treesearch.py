"""Search over trees with the exact closed-form allocation (Banavar: rank-one
loading + concave cost => a tree is optimal). Each tree costs microseconds to
evaluate, so we can afford thousands of candidates plus local improvement."""
import numpy as np, networkx as nx, pandas as pd
import lattice_domain as V

def steiner_prune(tree_edges):
    """drop dangling pass-through nodes (no demand) until every leaf is a sink"""
    T = nx.Graph(tree_edges)
    while True:
        dead = [v for v in T.nodes if T.degree(v) == 1 and v not in V.REQUIRED]
        if not dead: break
        T.remove_nodes_from(dead)
    return sorted((min(u,v),max(u,v)) for u,v in T.edges())

def D_tree(tree_edges, alpha):
    return V.tree_optimum(tree_edges, alpha)[1]

def random_spt(rng, jitter=0.6):
    G = nx.Graph()
    for k,(u,v) in enumerate(V.EDGES): G.add_edge(u, v, w=V.LENGTH[k]*(1+jitter*rng.random()))
    paths = nx.single_source_dijkstra_path(G, V.SRC, weight="w")
    tree = set()
    for t in V.LEAVES:
        p = paths[t]; tree |= {(min(a,b),max(a,b)) for a,b in zip(p[:-1],p[1:])}
    return sorted(tree)

def local_improve(tree_edges, alpha, rng, sweeps=6):
    """edge-swap descent: remove a tree edge, reconnect the cut with the best
    candidate edge, accept if dissipation drops; repeat until no move helps"""
    best = steiner_prune(tree_edges); Db = D_tree(best, alpha)
    for _ in range(sweeps):
        improved = False
        for e in list(best):
            if e not in best: continue
            T = nx.Graph(best); T.remove_edge(*e)
            side = nx.node_connected_component(T, V.SRC)
            for (u,v) in V.EDGES:
                if (u in side) != (v in side):
                    cand = [x for x in best if x != e] + [(min(u,v),max(u,v))]
                    cand = steiner_prune(cand)
                    if not V.REQUIRED <= set(nx.Graph(cand).nodes): continue
                    Tc = nx.Graph(cand)
                    if not (nx.is_connected(Tc) and nx.is_tree(Tc)): continue
                    D = D_tree(cand, alpha)
                    if D < Db - 1e-9: best, Db, improved = cand, D, True
        if not improved: break
    return best, Db

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    bfs = steiner_prune([(min(e),max(e)) for e in nx.bfs_tree(nx.Graph(V.EDGES), V.SRC).edges()])
    rows = []
    for b in (1.0, 0.75, 2/3):
        alpha = b/2
        DH = D_tree(V.HTREE, alpha); DB = D_tree(bfs, alpha)
        pool = [V.HTREE, bfs] + [random_spt(rng) for _ in range(400)]
        pool.sort(key=lambda t: D_tree(t, alpha))
        best, Db = None, np.inf
        for t in pool[:12]:                       # polish the dozen best seeds
            tt, Dt = local_improve(t, alpha, rng)
            if Dt < Db: best, Db = tt, Dt
        # also polish the H-tree itself, to see where it flows
        Hpol, DHpol = local_improve(V.HTREE, alpha, rng)
        w, _, f = V.tree_optimum(best, alpha)
        sh = V.shear(w, best)
        rows.append(dict(b=b, alpha=alpha, D_H=DH, D_bfs=DB, D_best=Db, D_H_polished=DHpol,
                         eta_top_H=Db/DH, eta_top_bfs=Db/DB, best_edges=len(best),
                         best_nodes=len(set(sum(best,()))), shear_slope=sh["slope"],
                         tau_ratio=sh["tau_ratio"], root_flow=max(f.values())))
        print(f"\nb={b:.3f} alpha={alpha:.3f}")
        print(f"  D: H-tree {DH:.1f} | BFS {DB:.1f} | best tree found {Db:.1f} | H-tree after edge-swap descent {DHpol:.1f}")
        print(f"  eta_top(H-tree)={Db/DH:.3f}  eta_top(BFS)={Db/DB:.3f}  ->  H-tree costs {DH/Db:.2f}x the best tree")
        print(f"  best tree: {len(best)} edges, {len(set(sum(best,())))} nodes (16 sinks + source + "
              f"{len(set(sum(best,())))-17} pass-through)")
        print(f"  shear: slope {sh['slope']:+.3f} (predicted {2*alpha-1:+.3f}), tau_max/tau_min={sh['tau_ratio']:.3f}, "
              f"analytic 16^(1-3/gamma)={16**(3/(2*(alpha+1))-1):.3f}")
    pd.DataFrame(rows).to_csv("tree_search.tsv", sep="\t", index=False)
    import pickle; pickle.dump({r['b']:None for r in rows}, open('dummy.pkl','wb'))
