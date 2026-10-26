"""
ensembles.py — Replica ensembles behind Tables 1–2 and Figures 2–3 of the manuscript.

All replicas use the model in replicas.py. Every run is seeded deterministically from its
parameters, and output files are rewritten after each replica so that interrupted runs resume.

Subcommands
  shared       replicas on one graph that differ only in initial-radius noise (Table 1, Fig. 2a–c)
  fate         dense-plexus replicas with snapshots of local signals; AUC for final survival (Fig. 2d)
  independent  replicas on independent substrates (graph seed = replica seed) with tree statistics
               and sampled point clouds for spatial overlap (Table 2, Fig. 3)
  summary      print every number used in the manuscript from the files above and write a JSON

Usage
  python src/ensembles.py shared --mode dense --b 0.75 --eta 0.1 --domain 3 --reps 30
  python src/ensembles.py fate --b 0.75 --domain 3 --reps 30
  python src/ensembles.py independent --mode staged --b 0.75 --seeds 21-32
  python src/ensembles.py independent --mode dense --b 0.75 --seeds 21-26 --graph 21 --tag same
  python src/ensembles.py summary --json results/ensembles_summary.json
"""
import argparse, glob, itertools, json, os, pickle, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__))
from replicas import make_graph, Net, adapt, dense, staged, dtilde, parse_seeds

RES = 'results'
MODES = {'dense': 0, 'staged': 1, 'staged_noregrow': 2}

def rng_for(*keys):
    return np.random.default_rng([int(k) for k in keys])

def save(obj, fn):
    tmp = fn + '.tmp'; pickle.dump(obj, open(tmp, 'wb')); os.replace(tmp, fn)

def load(fn, default):
    return pickle.load(open(fn, 'rb')) if os.path.exists(fn) else default

def run_protocol(net, mode, eta, rng):
    if mode == 'dense':
        return dense(net, eta, rng)
    return staged(net, eta, rng, regrow=(mode == 'staged'))

# ----------------------------------------------------------------------------- shared substrate
def cmd_shared(a):
    fn = f'{RES}/shared_{a.mode}_b{a.b}_eta{a.eta}_N{a.N}_d{a.domain}.pkl'
    res = load(fn, dict(S=[], D=[])); P, E, L = make_graph(a.N, a.domain); t = time.time()
    while len(res['S']) < a.reps and time.time() - t < a.budget:
        k = len(res['S']); net = Net(P, E, L, a.b)
        alive, lr = run_protocol(net, a.mode, a.eta, rng_for(a.domain, 1000 * a.eta, 100 * a.b, MODES[a.mode], k))
        res['S'].append(alive); res['D'].append(float(dtilde(net, alive, lr))); save(res, fn)
    print(f'{fn}: {len(res["S"])}/{a.reps} replicas')

# ----------------------------------------------------------------------------- fate predictability
SNAPS = (0, 10, 25, 50, 100, 200)
SIGNALS = ('flow', 'tau', 'z', 'Dc', 'r')

def auc(score, label):
    label = label.astype(bool); n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0: return np.nan
    rk = rankdata(score); return (rk[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def cmd_fate(a):
    fn = f'{RES}/fate_b{a.b}_eta{a.eta}_N{a.N}_d{a.domain}.pkl'
    res = load(fn, dict(S=[], D=[], A=[])); P, E, L = make_graph(a.N, a.domain); t = time.time()
    while len(res['S']) < a.reps and time.time() - t < a.budget:
        k = len(res['S']); net = Net(P, E, L, a.b); rng = rng_for(a.domain, 1000 * a.eta, 100 * a.b, 9, k)
        lr = np.log(0.05) + a.eta * rng.standard_normal(net.m); alive = np.ones(net.m, bool); snaps = {}
        for it in range(20000):
            r = np.exp(lr); f, _, w, _ = net.solve(r, alive)
            z = np.abs(f) / r ** 3 / net.tauset(r)
            if it in SNAPS:
                snaps[it] = dict(alive=alive.copy(), flow=np.abs(f), tau=np.abs(f) / r ** 3, z=z, r=r,
                                 Dc=(f ** 2 / w) / (L ** (3 * a.b / 2) * w ** (a.b / 2)))
            dz = np.where(alive, z - 1, 0)
            if np.max(np.abs(dz)) < 1e-5: break
            lr = lr + np.clip(0.2 * dz, -0.02, 0.02); alive &= lr > np.max(lr[alive]) + np.log(1e-3)
        A = {it: {q: auc(s[q][s['alive']], alive[s['alive']]) for q in SIGNALS} for it, s in snaps.items()}
        res['S'].append(alive); res['D'].append(float(dtilde(net, alive, lr))); res['A'].append(A); save(res, fn)
    print(f'{fn}: {len(res["S"])}/{a.reps} replicas')

# ----------------------------------------------------------------------------- independent substrates
def tree_stats(P, E, L, b, alive, lr, N):
    net = Net(P, E, L, b); r = np.exp(lr); f, _, _, phi = net.solve(r, alive)
    idx = np.nonzero(alive)[0]; k = np.abs(f[idx]); kk = np.round(k).astype(int)
    x = np.arange(2, N // 20); cc = np.array([np.mean(kk >= v) for v in x])
    gamma = -np.polyfit(np.log(x), np.log(cc), 1)[0]
    up = np.where(phi[E[idx, 0]] > phi[E[idx, 1]], E[idx, 0], E[idx, 1]); dn = np.where(phi[E[idx, 0]] > phi[E[idx, 1]], E[idx, 1], E[idx, 0])
    out = {}
    for u, d, l in zip(up, dn, L[idx]): out.setdefault(u, []).append((d, l))
    dl = np.zeros(len(P))
    for n in np.argsort(phi):
        if n in out: dl[n] = max(dl[d] + l for d, l in out[n])
    lmain = dl[dn] + L[idx]; m = kk >= 2
    hack = np.polyfit(np.log(k[m]), np.log(lmain[m]), 1)[0]
    pts, wt = [], []
    for e, kv in zip(idx, k):
        p0, p1 = P[E[e, 0]], P[E[e, 1]]; n = max(2, int(L[e] / 0.01))
        s = np.linspace(0, 1, n)[:, None]; pts.append(p0 + s * (p1 - p0)); wt.append(np.full(n, kv))
    return dict(gamma=float(gamma), hack=float(hack), load=k, pts=np.vstack(pts), k=np.concatenate(wt))

def cmd_independent(a):
    fn = f'{RES}/indep_{a.mode}_b{a.b}_N{a.N}_{a.tag}.pkl'; res = load(fn, {}); t = time.time()
    for i, sd in enumerate(parse_seeds(a.seeds)):
        key = (sd, i)
        if key in res: continue
        if time.time() - t > a.budget: break
        g = a.graph if a.graph is not None else sd
        P, E, L = make_graph(a.N, g); net = Net(P, E, L, a.b)
        alive, lr = run_protocol(net, a.mode, a.eta, rng_for(g, sd, 100 * a.b, MODES[a.mode], i, 7))
        st = tree_stats(P, E, L, a.b, alive, lr, a.N); st['Dt'] = float(dtilde(net, alive, lr)); res[key] = st; save(res, fn)
    print(f'{fn}: {len(res)} replicas')

# ----------------------------------------------------------------------------- summary
def contingent(S):
    p = np.mean(S, 0); return float(((p > 0) & (p < 1)).sum() / (p > 0).sum()), int((p == 1).sum())

def jacc(A, B=None):
    if B is None: pairs = [(A[i], A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
    else: pairs = [(x, y) for x in A for y in B]
    return float(np.mean([(x & y).sum() / (x | y).sum() for x, y in pairs]))

def overlap(A, B, sel, tol):
    best = 0
    for ang in np.deg2rad(np.arange(-20, 21, 4)):
        c, s = np.cos(ang), np.sin(ang); Br = B['pts'] @ np.array([[c, -s], [s, c]]).T
        ma = sel(A['k']) & (np.hypot(*A['pts'].T) > 0.15); mb = sel(B['k']) & (np.hypot(*Br.T) > 0.15)
        f1 = np.mean(cKDTree(Br[mb]).query(A['pts'][ma])[0] < tol); f2 = np.mean(cKDTree(A['pts'][ma]).query(Br[mb])[0] < tol)
        best = max(best, (f1 + f2) / 2)
    return best

def cmd_summary(a):
    out = {}
    def L_(pat):
        fs = sorted(glob.glob(f'{RES}/{pat}')); return {os.path.basename(f): pickle.load(open(f, 'rb')) for f in fs}
    print('== Table 1 (shared substrate, N = 400)')
    ref = {}
    for name, r in L_('shared_*.pkl').items():
        c, always = contingent(np.array(r['S'])); D = np.array(r['D'])
        out[name] = dict(n=len(D), contingent=c, always=always, D_mean=float(D.mean()), D_cv=float(D.std() / D.mean()))
        print(f'  {name}: n={len(D)} always={always} contingent={c:.2f} D={D.mean():.4g} CV={D.std()/D.mean():.3f}')
    for b in ('0.75', '0.5'):
        d0 = out.get(f'shared_dense_b{b}_eta0.1_N400_d3.pkl')
        if d0:
            for name in out:
                if f'_b{b}_' in name and name.startswith('shared_'): out[name]['D_rel_dense10'] = out[name]['D_mean'] / d0['D_mean']
        S = {m: np.array(load(f'{RES}/shared_{m}_b{b}_eta0.1_N400_d3.pkl', dict(S=[]))['S']) for m in ('dense', 'staged')}
        if len(S['dense']) and len(S['staged']):
            out[f'jaccard_b{b}'] = dict(dense=jacc(S['dense']), staged=jacc(S['staged']), between=jacc(S['dense'], S['staged']),
                                        both_deterministic=int(((S['dense'].mean(0) == 1) & (S['staged'].mean(0) == 1)).sum()))
            print(f'  Jaccard b={b}:', out[f'jaccard_b{b}'])
    print('== Fate ensembles (dense, 10% noise, N = 400)')
    for name, r in L_('fate_*.pkl').items():
        c, always = contingent(np.array(r['S'])); D = np.array(r['D'])
        aucs = {it: {q: float(np.nanmean([x[it][q] for x in r['A']])) for q in SIGNALS} for it in SNAPS}
        out[name] = dict(n=len(D), contingent=c, always=always, D_cv=float(D.std() / D.mean()), auc=aucs)
        print(f'  {name}: always={always} contingent={c:.2f} CV={D.std()/D.mean():.3f} AUC0={aucs[0]} AUC200={aucs[200]}')
    print('== Table 2 (independent substrates, N = 800)')
    for name, r in L_('indep_*_ind.pkl').items():
        g = [v['gamma'] for v in r.values()]; h = [v['hack'] for v in r.values()]
        out[name] = dict(n=len(g), gamma=[float(np.mean(g)), float(np.std(g))], hack=[float(np.mean(h)), float(np.std(h))])
        print(f'  {name}: n={len(g)} gamma={np.mean(g):.3f}±{np.std(g):.3f} hack={np.mean(h):.3f}±{np.std(h):.3f}')
    s = np.sqrt(np.pi / 800)
    for mode in ('dense', 'staged'):
        sh = list(load(f'{RES}/indep_{mode}_b0.75_N800_same.pkl', {}).values()); ind = list(load(f'{RES}/indep_{mode}_b0.75_N800_ind.pkl', {}).values())
        if len(sh) > 1 and len(ind) > 1:
            al = lambda k: k >= 1
            o_sh = float(np.mean([overlap(x, y, al, 0.2 * s) for x, y in itertools.combinations(sh, 2)]))
            o_in = float(np.mean([overlap(x, y, al, 0.2 * s) for x, y in list(itertools.combinations(ind, 2))[:20]]))
            out[f'overlap_{mode}'] = dict(shared=o_sh, independent=o_in); print(f'  overlap {mode}: shared {o_sh:.3f} independent {o_in:.3f}')
    json.dump(out, open(a.json, 'w'), indent=1); print('written', a.json)

def cmd_scaling(a):
    """Replica-to-replica fluctuations of tree statistics against network size (independent substrates).
    gamma is refitted on a fixed load range [2, 10] so that sizes are comparable."""
    out = {}
    for mode in ('dense', 'staged'):
        for N in (200, 400, 800, 1600):
            fn = f'{RES}/indep_{mode}_b0.75_N{N}_ind.pkl'
            if not os.path.exists(fn): continue
            r = list(pickle.load(open(fn, 'rb')).values()); g = []
            for v in r:
                kk = np.round(v['load']).astype(int); x = np.arange(2, 11); cc = np.array([np.mean(kk >= q) for q in x])
                g.append(-np.polyfit(np.log(x), np.log(cc), 1)[0])
            g = np.array(g); h = np.array([v['hack'] for v in r]); D = np.array([v['Dt'] for v in r])
            out[f'{mode}_N{N}'] = dict(n=len(r), gamma=[float(g.mean()), float(g.std())], hack=[float(h.mean()), float(h.std())],
                                      cv_gamma=float(g.std() / g.mean()), cv_hack=float(h.std() / h.mean()), cv_D=float(D.std() / D.mean()))
            o = out[f'{mode}_N{N}']
            print(f"{mode:6s} N={N:5d} n={len(r)} gamma[2,10]={g.mean():.3f}±{g.std():.3f} (CV {o['cv_gamma']:.3f}) hack={h.mean():.3f}±{h.std():.3f} (CV {o['cv_hack']:.3f}) CV(D)={o['cv_D']:.3f}")
    for mode in ('dense', 'staged'):
        Ns = [N for N in (200, 400, 800, 1600) if f'{mode}_N{N}' in out]
        for q in ('cv_gamma', 'cv_hack', 'cv_D'):
            y = [out[f'{mode}_N{N}'][q] for N in Ns]; sl = np.polyfit(np.log(Ns), np.log(y), 1)[0]
            out[f'{mode}_{q}_slope'] = float(sl); print(f'{mode} {q}: log-log slope vs N = {sl:.2f} (self-averaging if < 0; -0.5 for independent fluctuations)')
    json.dump(out, open(a.json, 'w'), indent=1)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('shared', 'fate', 'independent'):
        p = sub.add_parser(name); p.add_argument('--b', type=float, required=True); p.add_argument('--eta', type=float, default=0.1)
        p.add_argument('--budget', type=float, default=1e9, help='stop after this many seconds (resume later)')
        if name == 'independent':
            p.add_argument('--mode', choices=list(MODES), required=True); p.add_argument('--N', type=int, default=800)
            p.add_argument('--seeds', required=True); p.add_argument('--graph', type=int, default=None); p.add_argument('--tag', default='ind')
        else:
            p.add_argument('--N', type=int, default=400); p.add_argument('--domain', type=int, default=3); p.add_argument('--reps', type=int, default=30)
            if name == 'shared': p.add_argument('--mode', choices=list(MODES), required=True)
    sc = sub.add_parser('scaling'); sc.add_argument('--json', default=f'{RES}/scaling.json')
    s = sub.add_parser('summary'); s.add_argument('--json', default=f'{RES}/ensembles_summary.json')
    a = ap.parse_args(); os.makedirs(RES, exist_ok=True)
    {'shared': cmd_shared, 'fate': cmd_fate, 'independent': cmd_independent, 'summary': cmd_summary, 'scaling': cmd_scaling}[a.cmd](a)

if __name__ == '__main__':
    main()
