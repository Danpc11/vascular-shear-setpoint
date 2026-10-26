"""
replicas.py — Adaptive vascular network model (shear set-point rule with power-law
maintenance cost), replica ensembles, and window statistics comparable to fundus photos.

Model: Delaunay graph in the unit disk, source at the centre, unit-demand sinks.
Conductance w = r^4/l; maintenance cost a_e w^(b/2) with a_e = l^(3b/2) (per-segment cost).
Local rule: d ln r/dt = kappa (z - 1), z = |tau| / tau_set(r, l), tau_set = tau0 r^(b-1) l^((b-1)/2).
Edges are pruned below 1e-3 of the largest radius.

Usage
  python src/replicas.py replicas --mode staged --b 0.75 --N 800 --seeds 41-52 --out results/W_staged_b0.75.pkl
  python src/replicas.py window   --trees results/W_dense_b0.75.pkl results/W_staged_b0.75.pkl --b 0.75 --N 800 --json results/model_window.json
"""
import argparse, json, os, pickle, time
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------------- graph and flow
def make_graph(N, seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(N)); th = 2 * np.pi * rng.random(N)
    P = np.vstack([[0, 0], np.c_[r * np.cos(th), r * np.sin(th)]])
    E = set()
    for s in Delaunay(P).simplices:
        for i in range(3):
            a, b = sorted((s[i], s[(i + 1) % 3])); E.add((a, b))
    E = np.array(sorted(E)); L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    return P, E, L

class Net:
    def __init__(self, P, E, L, b, tau0=1.0):
        self.P, self.E, self.L, self.b, self.tau0 = P, E, L, b, tau0
        self.n, self.m = len(P), len(E)
        self.I = -np.ones(self.n); self.I[0] = self.n - 1
        rows = np.r_[np.arange(self.m), np.arange(self.m)]; cols = np.r_[E[:, 0], E[:, 1]]
        self.B = csr_matrix((np.r_[np.ones(self.m), -np.ones(self.m)], (rows, cols)), shape=(self.m, self.n))

    def solve(self, r, alive):
        w = np.where(alive, r ** 4 / self.L, 1e-14)
        Lap = (self.B.T @ csr_matrix((w, (np.arange(self.m), np.arange(self.m)))) @ self.B).tocsr()
        phi = np.zeros(self.n); phi[1:] = spsolve(Lap[1:, 1:].tocsc(), self.I[1:])
        dp = self.B @ phi
        return w * dp, dp, w, phi

    def tauset(self, r):
        return self.tau0 * r ** (self.b - 1) * self.L ** ((self.b - 1) / 2)

def adapt(net, lr, alive, kappa=0.2, delta=0.02, tol=1e-5, maxit=20000, prune=1e-3):
    for _ in range(maxit):
        r = np.exp(lr); f, _, _, _ = net.solve(r, alive)
        dz = np.where(alive, np.abs(f) / r ** 3 / net.tauset(r) - 1, 0)
        if np.max(np.abs(dz)) < tol: break
        lr = lr + np.clip(kappa * dz, -delta, delta)
        alive &= lr > np.max(lr[alive]) + np.log(prune)
    return lr, alive

def dense(net, eta, rng):
    lr0 = np.log(0.05) + eta * rng.standard_normal(net.m)
    return adapt(net, lr0, np.ones(net.m, bool))[::-1]

def staged(net, eta, rng, K=12, regrow=True):
    d = np.linalg.norm(net.P[1:], axis=1); order = np.argsort(d) + 1
    active = np.zeros(net.n, bool); active[0] = True
    alive = np.zeros(net.m, bool); lr = np.full(net.m, -50.0); seeded = np.zeros(net.m, bool)
    for k, ch in enumerate(np.array_split(order, K)):
        active[ch] = True
        adm = active[net.E[:, 0]] & active[net.E[:, 1]]
        new = adm & ~alive & (regrow | ~seeded)
        base = np.log(0.025) if k == 0 else np.log(0.5 * np.median(np.exp(lr[alive])))
        lr[new] = base + eta * rng.standard_normal(new.sum()); alive |= new; seeded |= new
        I = np.zeros(net.n); I[active] = -1; I[0] = 0; I[0] = -I.sum(); net.I = I
        lr, alive = adapt(net, lr, alive)
    net.I = -np.ones(net.n); net.I[0] = net.n - 1
    return alive, lr

def dtilde(net, alive, lr):
    r = np.exp(lr); f, _, w, _ = net.solve(r, alive)
    D = np.sum(f[alive] ** 2 / w[alive]); al = net.b / 2
    C = np.sum(net.L[alive] ** (3 * net.b / 2) * w[alive] ** al)
    return D * C ** (1 / al)

# ----------------------------------------------------------------------------- window statistics
def window_segments(P, E, L, b, alive, lr, rin=0.11, rout=0.30):
    """Segments of the supply tree seen through an annular window (fundus-photo analogue)."""
    net = Net(P, E, L, b); r = np.exp(lr); _, _, _, phi = net.solve(r, alive)
    rad = np.hypot(*P.T); idx = np.nonzero(alive)[0]
    a, c = E[idx, 0], E[idx, 1]
    up = np.where(phi[a] > phi[c], a, c); dn = np.where(phi[a] > phi[c], c, a)
    keep = (rad[dn] > rin) & (rad[dn] < rout) & (rad[up] < rout)
    up, dn, el, er = up[keep], dn[keep], L[idx][keep], r[idx][keep]
    rootlink = rad[up] <= rin
    par = {d: (u, l, rr, rl) for u, d, l, rr, rl in zip(up, dn, el, er, rootlink)}
    ch = {}
    for d, (u, _, _, rl) in par.items():
        if not rl: ch.setdefault(u, []).append(d)
    nodes = sorted(par, key=lambda q: phi[q])
    load = {q: par[q][1] for q in nodes}; height = {q: 0.0 for q in nodes}
    for q in nodes:
        for cc in ch.get(q, []):
            load[q] += load[cc]; height[q] = max(height[q], height[cc] + par[cc][1])
    segs = []
    for q in nodes[::-1]:
        u, l, rr, rl = par[q]
        if rl or u not in par or len(ch.get(u, [])) != 1:
            chain = [q]; v = q
            while len(ch.get(v, [])) == 1:
                v = ch[v][0]; chain.append(v)
            segs.append((load[q], sum(par[x][1] for x in chain), np.median([par[x][2] for x in chain]),
                         len(ch.get(v, [])) == 0, height[q] + par[q][1]))
    s = np.array(segs, float)
    return dict(load=s[:, 0], len=s[:, 1], w=s[:, 2], leaf=s[:, 3].astype(bool), main=s[:, 4])

def exponents(st, lo=2, hi=40):
    """gamma: slope of the complementary CDF of downstream load (in units of median terminal load);
    hack: slope of mainstream length vs load; wslope: d ln w / d ln load at fixed segment length."""
    L0 = np.median(st['load'][st['leaf']]); x = np.geomspace(lo, hi, 12) * L0
    cc = np.array([np.mean(st['load'] >= v) for v in x]); m = cc > 0
    tau = -np.polyfit(np.log(x[m]), np.log(cc[m]), 1)[0] if m.sum() > 4 else np.nan
    mm = st['load'] >= lo * L0
    hack = np.polyfit(np.log(st['load'][mm]), np.log(st['main'][mm]), 1)[0] if mm.sum() > 10 else np.nan
    q = (st['w'] > 0) & (st['len'] > 0)
    X = np.c_[np.ones(q.sum()), np.log(st['load'][q]), np.log(st['len'][q])]
    c, *_ = np.linalg.lstsq(X, np.log(st['w'][q]), rcond=None)
    return dict(gamma=float(tau), hack=float(hack), wslope=float(c[1]), nseg=int(len(st['load'])))

def pooled_tau(sts, lo=2, hi=40):
    v = np.concatenate([s['load'] / np.median(s['load'][s['leaf']]) for s in sts])
    x = np.geomspace(lo, hi, 12); cc = np.array([np.mean(v >= q) for q in x]); m = cc > 0
    return float(-np.polyfit(np.log(x[m]), np.log(cc[m]), 1)[0])

# ----------------------------------------------------------------------------- CLI
def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replicas'); r.add_argument('--mode', choices=['dense', 'staged', 'staged_noregrow'], required=True)
    r.add_argument('--b', type=float, required=True); r.add_argument('--N', type=int, default=800)
    r.add_argument('--seeds', required=True); r.add_argument('--eta', type=float, default=0.1)
    r.add_argument('--same-substrate', type=int, default=None, help='use this graph seed for all replicas')
    r.add_argument('--out', required=True)
    w = sub.add_parser('window'); w.add_argument('--trees', nargs='+', required=True)
    w.add_argument('--b', type=float, required=True); w.add_argument('--N', type=int, default=800)
    w.add_argument('--rin', type=float, default=0.11); w.add_argument('--rout', type=float, default=0.30)
    w.add_argument('--json', required=True)
    a = ap.parse_args()
    if a.cmd == 'replicas':
        res = pickle.load(open(a.out, 'rb')) if os.path.exists(a.out) else {}
        for sd in parse_seeds(a.seeds):
            if sd in res: continue
            gseed = a.same_substrate if a.same_substrate is not None else sd
            P, E, L = make_graph(a.N, gseed); net = Net(P, E, L, a.b); rng = np.random.default_rng(sd)
            t = time.time()
            alive, lr = (dense(net, a.eta, rng) if a.mode == 'dense'
                         else staged(net, a.eta, rng, regrow=(a.mode == 'staged')))
            res[sd] = dict(graph_seed=gseed, alive=alive, lr=lr, Dt=float(dtilde(net, alive, lr)))
            pickle.dump(res, open(a.out, 'wb'))
            print(f'replica {sd} (graph {gseed}) done in {time.time() - t:.1f}s', flush=True)
    else:
        rng = np.random.default_rng(0); out = {}
        for fn in a.trees:
            R = pickle.load(open(fn, 'rb')); sts = []
            for sd, v in R.items():
                P, E, L = make_graph(a.N, v['graph_seed'])
                sts.append(window_segments(P, E, L, a.b, v['alive'], v['lr'], a.rin, a.rout))
            ex = [exponents(s) for s in sts]
            boot = [pooled_tau([sts[i] for i in rng.integers(0, len(sts), len(sts))]) for _ in range(300)]
            out[os.path.basename(fn)] = dict(
                n=len(sts), nseg=float(np.mean([e['nseg'] for e in ex])),
                gamma_tree=[float(np.nanmean([e['gamma'] for e in ex])), float(np.nanstd([e['gamma'] for e in ex]))],
                gamma_pooled=[pooled_tau(sts), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                hack=[float(np.nanmean([e['hack'] for e in ex])), float(np.nanstd([e['hack'] for e in ex]))],
                wslope=float(np.mean([e['wslope'] for e in ex])), wslope_expected=1 / (a.b + 2))
            print(os.path.basename(fn), json.dumps(out[os.path.basename(fn)]))
        json.dump(out, open(a.json, 'w'), indent=2)

if __name__ == '__main__':
    main()
