"""Poiseuille flows on a network, dissipation, and the exact tree allocation.

Conventions (units dropped): conductance w_e = r_e^4 / l_e; wall shear
tau_e = |f_e| / r_e^3; cost  C = sum_e a_e w_e^alpha  with  alpha = b/2 and
a_e = l_e^{3b/2}, so that C is proportional to sum_e (r_e^2 l_e)^b, the
metabolic cost of the vessel mass with exponent b.  b = 1 is Murray's volume
cost, 3/4 and 2/3 are the metabolic exponents of Kleiber and Rubner.  Lowering b
flattens the cost in w: as b -> 0, a_e w_e^alpha -> 1 and the budget counts
active channels instead of weighing them, so the problem tends to a Steiner-like
one.  Exponents below 2/3 are not metabolic rates of any measured tissue; they
are included to show that the trends are monotone in b and to approach that
limit.  The rank-one loading (one source, unit sinks) makes the dissipation
optimum a tree, on which the allocation is closed-form.
"""
import numpy as np, networkx as nx
from scipy.linalg import solve, pinvh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

class Net:
    def __init__(self, pts, E, L, src=0):
        self.pts, self.E, self.L, self.src = pts, list(E), np.asarray(L), src
        self.n, self.m = len(pts), len(E)
        self.idx = {e: k for k, e in enumerate(self.E)}
        self.Ei = np.array([e[0] for e in self.E]); self.Ej = np.array([e[1] for e in self.E])
        self.B = np.zeros((self.n, self.m))
        self.B[self.Ei, np.arange(self.m)] = 1.0; self.B[self.Ej, np.arange(self.m)] = -1.0
        self.G = nx.Graph(); self.G.add_nodes_from(range(self.n))
        for k, (i, j) in enumerate(self.E): self.G.add_edge(i, j, l=self.L[k], k=k)

    # ---- parameters of the resource law
    @staticmethod
    def alpha(b): return b / 2.0
    def a_coef(self, b): return self.L ** (1.5 * b)

    # ---- linear flow solve
    def conductance(self, r, active):
        w = np.zeros(self.m); w[active] = r[active] ** 4 / self.L[active]; return w
    def potentials(self, w, mu, nodes):
        """phi on present nodes for demand mu (sums to zero on present nodes)."""
        idx = np.flatnonzero(nodes); Lap = (self.B[idx] * w) @ self.B[idx].T
        mask = idx != self.src; phi = np.zeros(self.n)
        phi[idx[mask]] = solve(Lap[np.ix_(mask, mask)], mu[idx][mask], assume_a="pos")
        return phi
    def flows(self, w, mu, nodes):
        phi = self.potentials(w, mu, nodes); return w * (self.B.T @ phi), phi
    def dissipation(self, w, mu, nodes):
        f, phi = self.flows(w, mu, nodes); return float(mu @ phi), f
    def cost(self, r, active, b):
        w = self.conductance(r, active); a = self.a_coef(b)
        return float(np.sum(a[active] * w[active] ** self.alpha(b)))
    def d_norm(self, r, active, b, mu=None, nodes=None):
        """budget-invariant dissipation  D * C^{1/alpha}  (D scales as C^{-1/alpha})."""
        nodes = np.ones(self.n, bool) if nodes is None else nodes
        if mu is None: mu = self.unit_demand(nodes)
        w = self.conductance(r, active); D, _ = self.dissipation(w, mu, nodes)
        return D * self.cost(r, active, b) ** (1.0 / self.alpha(b))
    def unit_demand(self, nodes):
        mu = -nodes.astype(float); mu[self.src] = nodes.sum() - 1; return mu

    # ---- trees
    def tree_flows(self, tree, nodes=None):
        nodes = np.ones(self.n, bool) if nodes is None else nodes
        T = nx.Graph(tree); parent = dict(nx.dfs_predecessors(T, self.src))
        order = list(nx.dfs_preorder_nodes(T, self.src))
        sub = {v: (1.0 if (nodes[v] and v != self.src) else 0.0) for v in T.nodes}
        for v in reversed(order):
            if v in parent: sub[parent[v]] += sub[v]
        f = {e: sub[e[0] if parent.get(e[0]) == e[1] else e[1]] for e in tree}
        return f, parent
    def allocate(self, tree, b, C0=1.0):
        """exact optimum on a fixed tree:  w_e ∝ (f_e^2 / a_e)^{1/(alpha+1)}."""
        alpha = self.alpha(b); f, parent = self.tree_flows(tree)
        ks = np.array([self.idx[e] for e in tree]); fl = np.array([f[e] for e in tree])
        a = self.a_coef(b)[ks]; l = self.L[ks]
        w = np.where(fl > 0, (fl ** 2 / a) ** (1 / (alpha + 1)), 0.0)
        keep = w > 0; w *= (C0 / np.sum(a[keep] * w[keep] ** alpha)) ** (1 / alpha)
        D = float(np.sum(fl[keep] ** 2 / w[keep]))
        r_full = np.zeros(self.m); r_full[ks] = (w * l) ** 0.25
        act = np.zeros(self.m, bool); act[ks[keep]] = True
        return dict(D=D, D_norm=D * C0 ** (1 / alpha), edges=[e for e, k in zip(tree, keep) if k],
                    f=fl[keep], l=l[keep], w=w[keep], r=(w[keep] * l[keep]) ** 0.25,
                    r_full=r_full, active=act, parent=parent)

def shear_stats(sol, b):
    """joint fit  ln tau = s_r ln r + s_l ln l + c ; exact values are b-1 and (b-1)/2."""
    tau = sol["f"] / sol["r"] ** 3
    x, z, y = np.log(sol["r"]), np.log(sol["l"]), np.log(tau)
    A = np.c_[x, z, np.ones_like(x)]; coef = np.linalg.lstsq(A, y, rcond=None)[0]
    s1, c1 = np.polyfit(x, y, 1); resid = y - (s1 * x + c1)
    # The residual of the free one-variable fit equals |1-b|/2 times the spread of
    # ln l that survives once ln r is regressed out, not |1-b|/2 times sd(ln l):
    # along a tree the two covary, and using the raw spread overstates the
    # prediction by a few per cent.
    a1, a0 = np.polyfit(x, z, 1); z_res = z - (a1 * x + a0)
    return dict(slope_r_joint=coef[0], slope_l_joint=coef[1], slope_r_only=s1,
                scatter_lntau_r_only=resid.std(),
                scatter_pred=abs(1 - b) / 2 * z_res.std(),
                q_fitted=float(a1),
                tau_ratio=tau.max() / tau.min(), r_ratio=sol["r"].max() / sol["r"].min(),
                exact_r=b - 1, exact_l=(b - 1) / 2)

def tau0_from_tree(sol, b):
    """the set-point constant implied by an optimal tree (constant across its edges)."""
    t = sol["f"] / sol["r"] ** 3 / (sol["r"] ** (b - 1) * sol["l"] ** ((b - 1) / 2))
    return float(np.median(t)), float(t.std() / t.mean())
