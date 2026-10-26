"""
retina_bilateral.py — Bilateral (OD vs OS) analysis of retinal vascular networks.

Subcommands
  overlap  Identity overlap between eyes: same patient vs other patients (OS mirrored, rotation search).
  metrics  Global network statistics: OD-OS correlation, raw and adjusted for image quality,
           axial length, age and disc size.
  similarity  Common similarity scale (0 = unrelated people, 1 = identical) for vessel identity and
           adjusted network statistics, with bootstrap intervals (Fig. 4b,c)
  ceiling  Overlap of re-imaged identical eyes (rotation about the disc and added noise), the
           ceiling of the identity scale; needs the original images
  trees    Disc-rooted trees, per-eye gamma / Hack / width-load exponents, pooled gamma, OD-OS
           correlation, and comparison with the model window statistics (src/replicas.py window).

Usage
  python src/retina_bilateral.py overlap --eyes eyes.pkl --out overlap.pkl
  python src/retina_bilateral.py metrics --eyes eyes.pkl
  python src/retina_bilateral.py similarity --eyes eyes.pkl --overlap overlap.pkl --ceiling 0.85
  python src/retina_bilateral.py ceiling --eyes eyes.pkl --images ImagesWithContours
  python src/retina_bilateral.py trees   --eyes eyes.pkl --gap 0.15 --model-json model_window.json
"""
import argparse, json, os, pickle, warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
warnings.filterwarnings('ignore')
R1, R2, TOL, WT = 1.3, 3.6, 0.06, 0.09
OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def load(path):
    db = pickle.load(open(path, 'rb'))
    L = np.array([db[k]['m']['len'] for k in db]); thr = 0.5 * np.median(L)
    ids = sorted({k[:3] for k in db})
    ok = [i for i in ids if i + 'OD' in db and i + 'OS' in db and db[i + 'OD']['m']['len'] >= thr and db[i + 'OS']['m']['len'] >= thr]
    print(f'patients with both eyes: {len([i for i in ids if i+"OD" in db and i+"OS" in db])}, passing QC: {len(ok)}')
    return db, ok

# ----------------------------------------------------------------------------- identity overlap
def pts(o, mirror, cls=None):
    x = (o['xs'] - o['cx']) / o['rd']; y = (o['ys'] - o['cy']) / o['rd']
    if mirror: x = -x
    w = o['w'] / o['rd']; r = np.hypot(x, y); m = (r > R1) & (r < R2)
    if cls is not None and o.get('av') is not None: m &= o['av'] == cls
    return np.c_[x[m], y[m]], w[m]

def infov(o, P, mirror):
    x = P[:, 0] * (-1 if mirror else 1) * o['rd'] + o['cx']; y = P[:, 1] * o['rd'] + o['cy']
    xi = np.clip(np.round(x).astype(int), 0, o['shape'][1] - 1); yi = np.clip(np.round(y).astype(int), 0, o['shape'][0] - 1)
    return (x >= 0) & (x < o['shape'][1]) & (y >= 0) & (y < o['shape'][0]) & o['fov'][yi, xi]

def rot(P, a):
    c, s = np.cos(a), np.sin(a); return P @ np.array([[c, -s], [s, c]]).T

def score(A, wA, B, wB, oA, oB, a, sel):
    Br = rot(B, a); okB = infov(oA, Br, False); okA = infov(oB, rot(A, -a), True)
    mA = okA & sel(wA); mB = okB & sel(wB)
    if mA.sum() < 20 or mB.sum() < 20: return np.nan
    f1 = np.mean(cKDTree(Br[mB]).query(A[mA])[0] < TOL); f2 = np.mean(cKDTree(A[mA]).query(Br[mB])[0] < TOL)
    return (f1 + f2) / 2

SELS = {'all': lambda w: np.ones(len(w), bool), 'thick': lambda w: w >= WT, 'thin': lambda w: w < WT}

def cmd_overlap(a):
    db, ok = load(a.eyes); rng = np.random.default_rng(1)
    P = {k: pts(db[k], k.endswith('OS')) for k in db}; rows = []
    for i in ok:
        others = rng.choice([j for j in ok if j != i], min(a.others, len(ok) - 1), replace=False)
        for j in [i] + list(others):
            A, wA = P[i + 'OD']; B, wB = P[j + 'OS']; oA, oB = db[i + 'OD'], db[j + 'OS']
            best = (-1, 0)
            for ang in np.deg2rad(np.arange(-20, 21, 2)):
                s = score(A, wA, B, wB, oA, oB, ang, SELS['all'])
                if s == s and s > best[0]: best = (s, ang)
            rows.append((i, j, i == j) + tuple(score(A, wA, B, wB, oA, oB, best[1], SELS[c]) for c in ('all', 'thick', 'thin')))
    pickle.dump(rows, open(a.out, 'wb'))
    Rr = np.array([r[3:] for r in rows], float); same = np.array([r[2] for r in rows]); pid = np.array([r[0] for r in rows])
    for c, name in enumerate(('all', 'thick', 'thin')):
        delta = []
        for i in ok:
            m = pid == i; own = Rr[m & same, c]; oth = Rr[m & ~same, c]; oth = oth[oth == oth]
            if own.size and own[0] == own[0] and oth.size >= 3: delta.append(own[0] - oth.mean())
        delta = np.array(delta)
        ci = np.percentile([rng.choice(delta, len(delta)).mean() for _ in range(5000)], [2.5, 97.5])
        p = wilcoxon(delta).pvalue if len(delta) > 5 else np.nan
        print(f'{name:6s} same={np.nanmean(Rr[same, c]):.3f} other={np.nanmean(Rr[~same, c]):.3f} '
              f'delta={delta.mean():+.4f} CI95=[{ci[0]:.4f},{ci[1]:.4f}] Wilcoxon p={p:.2g} n={len(delta)}')

# ----------------------------------------------------------------------------- global metrics
def resid(v, X):
    X1 = np.column_stack([np.ones(len(v))] + X); m = np.all(np.isfinite(X1), 1) & np.isfinite(v)
    b, *_ = np.linalg.lstsq(X1[m], v[m], rcond=None); r = np.full(len(v), np.nan); r[m] = v[m] - X1[m] @ b; return r

def cmd_metrics(a):
    db, ok = load(a.eyes)
    def cov(e, full):
        X = [np.array([db[i + e][q] for i in ok], float) for q in ('contrast', 'qnoise')]
        if full: X += [np.array([db[i + e].get(q, np.nan) for i in ok], float) for q in ('axial', 'age', 'rd')]
        return X
    print(f"{'metric':8s} {'rho_raw':>8s} {'rho_quality':>12s} {'rho_full':>9s} {'p_full':>8s} {'intra/inter':>12s}")
    for k in ('dens', 'len', 'branch', 'wmean'):
        x = np.array([db[i + 'OD']['m'][k] for i in ok], float); y = np.array([db[i + 'OS']['m'][k] for i in ok], float)
        lx, ly = np.log(x + 1e-9), np.log(y + 1e-9)
        r0 = spearmanr(x, y)[0]
        rq = spearmanr(resid(lx, cov('OD', False)), resid(ly, cov('OS', False)), nan_policy='omit')[0]
        rf, pf = spearmanr(resid(lx, cov('OD', True)), resid(ly, cov('OS', True)), nan_policy='omit')
        intra = np.mean(np.abs(x - y)); inter = np.mean(np.abs(x[:, None] - y[None, :])[~np.eye(len(x), dtype=bool)])
        print(f'{k:8s} {r0:+8.2f} {rq:+12.2f} {rf:+9.2f} {pf:8.1e} {intra / inter:12.2f}')

# ----------------------------------------------------------------------------- rooted trees
def build_tree(o, gap, cls=None, rin=1.45, rout=R2):
    ys, xs, w = o['ys'], o['xs'], o['w']; rd = o['rd']
    r = np.hypot(ys - o['cy'], xs - o['cx']) / rd; keep = (r > 1.15) & (r < rout)
    if cls is not None and o.get('av') is not None: keep &= o['av'] == cls
    ys, xs, w, r = ys[keep], xs[keep], w[keep], r[keep]; n = len(ys)
    if n < 20: return None
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}; I, J, W = [], [], []
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in OFF:
            j = idx.get((y + dy, x + dx))
            if j is not None and j > i: I.append(i); J.append(j); W.append(np.hypot(dy, dx) / rd)
    nc, lab = connected_components(coo_matrix((W, (I, J)), shape=(n, n)), directed=False)
    root = n; rooted = set(lab[r < rin])
    for i in np.nonzero(r < rin)[0]: I.append(i); J.append(root); W.append(1e-6)
    P = np.c_[ys, xs]; kd = cKDTree(P)
    for c in range(nc):
        if c in rooted: continue
        m = np.nonzero(lab == c)[0]; p = m[np.argmin(r[m])]
        cand = [q for q in kd.query_ball_point(P[p], gap * rd) if lab[q] != c and r[q] < r[p]]
        if cand:
            q = min(cand, key=lambda q: np.hypot(*(P[q] - P[p]))); I.append(p); J.append(q); W.append(np.hypot(*(P[q] - P[p])) / rd)
    A = coo_matrix((W, (I, J)), shape=(n + 1, n + 1)).tocsr(); A = A + A.T
    dist, pred = dijkstra(A, indices=root, return_predecessors=True)
    return dict(n=n, w=w / rd, dist=dist, pred=pred, ok=np.isfinite(dist[:n]))

def segments(t):
    n, pred, d = t['n'], t['pred'], t['dist']; nodes = np.nonzero(t['ok'])[0]
    if len(nodes) < 20: return None
    order = nodes[np.argsort(d[nodes])]; par = np.where(t['ok'], pred[:n], -1)
    elen = np.zeros(n)
    for v in nodes: elen[v] = d[v] - d[par[v]] if par[v] < n else 0.0
    nch = np.zeros(n + 1, int)
    for v in nodes: nch[par[v]] += 1
    load = elen.copy(); height = np.zeros(n)
    for v in order[::-1]:
        u = par[v]
        if u < n: load[u] += load[v]; height[u] = max(height[u], height[v] + elen[v])
    seg = np.full(n, -1); s = 0; start = []
    for v in order:
        u = par[v]
        if u == n or nch[u] != 1: seg[v] = s; start.append(v); s += 1
        else: seg[v] = seg[u]
    S_ = np.array(start); sl = np.bincount(seg[nodes], weights=elen[nodes], minlength=s)
    sw = np.array([np.median(t['w'][nodes[seg[nodes] == k]]) for k in range(s)])
    leaf = np.array([nch[v] == 0 for v in nodes], bool); segleaf = np.zeros(s, bool); segleaf[seg[nodes][leaf]] = True
    m = sl > 0.02
    return dict(load=load[S_][m], len=sl[m], w=sw[m], leaf=segleaf[m], main=(height[S_] + elen[S_])[m])

def exponents(st, lo=2, hi=40):
    L0 = np.median(st['load'][st['leaf']]); x = np.geomspace(lo, hi, 12) * L0
    cc = np.array([np.mean(st['load'] >= v) for v in x]); m = cc > 0
    tau = -np.polyfit(np.log(x[m]), np.log(cc[m]), 1)[0] if m.sum() > 4 else np.nan
    mm = st['load'] >= lo * L0
    hack = np.polyfit(np.log(st['load'][mm]), np.log(st['main'][mm]), 1)[0] if mm.sum() > 10 else np.nan
    q = (st['w'] > 0) & (st['len'] > 0.05)
    if q.sum() < 5: return dict(gamma=tau, hack=hack, wslope=np.nan, nseg=len(st['load']))
    X = np.c_[np.ones(q.sum()), np.log(st['load'][q]), np.log(st['len'][q])]; c, *_ = np.linalg.lstsq(X, np.log(st['w'][q]), rcond=None)
    return dict(gamma=tau, hack=hack, wslope=c[1], nseg=len(st['load']))

def pooled_tau(sts, lo=2, hi=40):
    v = np.concatenate([s['load'] / np.median(s['load'][s['leaf']]) for s in sts])
    x = np.geomspace(lo, hi, 12); cc = np.array([np.mean(v >= q) for q in x]); m = cc > 0
    return -np.polyfit(np.log(x[m]), np.log(cc[m]), 1)[0]

def cmd_trees(a):
    db, ok = load(a.eyes); rng = np.random.default_rng(0)
    has_av = any(db[k].get('av') is not None for k in db)
    classes = {'artery': 1, 'vein': 2} if has_av else {'all': None}
    for cname, cls in classes.items():
        ST = {}
        for i in ok:
            for e in ('OD', 'OS'):
                t = build_tree(db[i + e], a.gap, cls); st = segments(t) if t else None
                if st is not None and len(st['load']) >= 15 and st['leaf'].sum() >= 3: ST[i + e] = st
        pats = [i for i in ok if i + 'OD' in ST and i + 'OS' in ST]
        ex = {k: exponents(s) for k, s in ST.items()}
        boot = [pooled_tau([ST[p + e] for p in rng.choice(pats, len(pats)) for e in ('OD', 'OS')]) for _ in range(300)]
        print(f'\n[{cname}] eyes {len(ST)}, pairs {len(pats)}, segments/eye {np.median([v["nseg"] for v in ex.values()]):.0f}')
        print(f'  gamma pooled {pooled_tau(list(ST.values())):.3f} [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]'
              f' | gamma/eye {np.nanmean([v["gamma"] for v in ex.values()]):.2f} | Hack {np.nanmean([v["hack"] for v in ex.values()]):.2f}'
              f' | width-load slope {np.nanmean([v["wslope"] for v in ex.values()]):.3f}')
        for q in ('gamma', 'hack', 'wslope'):
            x = np.array([ex[p + 'OD'][q] for p in pats]); y = np.array([ex[p + 'OS'][q] for p in pats]); m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 5:
                rho, p = spearmanr(x[m], y[m]); print(f'  {q:6s} Spearman OD-OS {rho:+.2f} (p={p:.2g}, n={m.sum()})')
    if a.model_json:
        print('\nmodel (same window and estimators):')
        for k, v in json.load(open(a.model_json)).items():
            print(f"  {k}: gamma pooled {v['gamma_pooled'][0]:.3f} [{v['gamma_pooled'][1]:.3f}, {v['gamma_pooled'][2]:.3f}] | "
                  f"Hack {v['hack'][0]:.2f} | width-load slope {v['wslope']:.3f} (untruncated expectation {v['wslope_expected']:.3f})")


# ----------------------------------------------------------------------------- common similarity scale
def crossings(o, r0=2.5, half=0.03):
    """Vessels crossing the circle of radius r0 disc radii: count and Murray-type capacity sum(w^3)."""
    from scipy import ndimage as ndi
    r = np.hypot(o['ys'] - o['cy'], o['xs'] - o['cx']) / o['rd']; m = np.abs(r - r0) < half
    img = np.zeros(o['shape'], bool); img[o['ys'][m], o['xs'][m]] = True
    wimg = np.zeros(o['shape']); wimg[o['ys'][m], o['xs'][m]] = o['w'][m] / o['rd']
    lab, n = ndi.label(img, structure=np.ones((3, 3)))
    if n == 0: return 0.0, 0.0
    wm = ndi.maximum(wimg, lab, index=np.arange(1, n + 1))
    yy, xx = np.mgrid[:o['shape'][0], :o['shape'][1]]; ring = np.abs(np.hypot(yy - o['cy'], xx - o['cx']) / o['rd'] - r0) < half
    cov = (ring & o['fov']).sum() / max(ring.sum(), 1)
    return n / max(cov, 1e-9), float(np.sum(wm ** 3)) / max(cov, 1e-9)

def cmd_similarity(a):
    db, ok = load(a.eyes); rows = pickle.load(open(a.overlap, 'rb')); rng = np.random.default_rng(0)
    def ci(v): return np.percentile([rng.choice(v, len(v)).mean() for _ in range(4000)], [2.5, 97.5])
    Rr = np.array([r[3:] for r in rows], float); same = np.array([r[2] for r in rows]); pid = np.array([r[0] for r in rows])
    out = {}
    for c, name in enumerate(('identity, all', 'identity, large', 'identity, thin')):
        v = []
        for i in ok:
            m = pid == i; own = Rr[m & same, c]; oth = Rr[m & ~same, c]; oth = oth[oth == oth]
            if own.size and own[0] == own[0] and oth.size >= 3: v.append((own[0] - oth.mean()) / (a.ceiling - oth.mean()))
        out[name] = np.array(v)
    for e in ('OD', 'OS'):
        for i in ok:
            n, w3 = crossings(db[i + e]); db[i + e]['m']['n2.5'] = n; db[i + e]['m']['w3_2.5'] = w3
    def cov(e):
        return [np.array([db[i + e][q] for i in ok], float) for q in ('contrast', 'qnoise')] + \
               [np.array([db[i + e].get(q, np.nan) for i in ok], float) for q in ('axial', 'age', 'rd')]
    labels = {'branch': 'branch points per area', 'len': 'vessel length per area', 'wmean': 'mean calibre',
              'dens': 'vessel area fraction', 'w3_2.5': 'Murray-type capacity', 'n2.5': 'vessels crossing 2.5 disc radii'}
    for k, name in labels.items():
        x = resid(np.log(np.array([db[i + 'OD']['m'][k] for i in ok], float) + 1e-6), cov('OD'))
        y = resid(np.log(np.array([db[i + 'OS']['m'][k] for i in ok], float) + 1e-6), cov('OS'))
        g = np.isfinite(x) & np.isfinite(y); x, y = x[g], y[g]
        out[name] = np.array([1 - abs(x[j] - y[j]) / np.mean(np.abs(x[j] - np.delete(y, j))) for j in range(len(x))])
    res = {}
    for k, v in out.items():
        lo, hi = ci(v); res[k] = dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), n=int(len(v)), values=v.tolist())
        print(f'{k:35s} {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(v)}')
    json.dump(res, open(a.out, 'w'))

def cmd_ceiling(a):
    import retina_segment as RS
    from skimage import io, transform
    db, ok = load(a.eyes); rng = np.random.default_rng(0); vals = []
    idx = np.linspace(0, len(ok) - 1, a.n).round().astype(int)
    for n_, j in enumerate(idx):
        k = ok[j] + ('OD' if n_ % 2 == 0 else 'OS')
        o = db[k]; im = io.imread(f'{a.images}/Opht_cont_RET{k}.jpg').astype(float)
        im2 = transform.rotate(im, 6, center=(o['cx'] / RS.S, o['cy'] / RS.S), preserve_range=True) * 0.9 + rng.normal(0, 6, im.shape)
        im2 = np.clip(im2, 0, 255) / 255; cy, cx, rd, cont = RS.disc_from_contours(im2)
        ves, fov, con, _ = RS.classical(im2, cont, o['cy'], o['cx'], o['rd']); o2 = RS.finish(ves, fov, o['cy'], o['cx'], o['rd'], con)
        mir = k.endswith('OS'); A, _ = pts(o, mir); B, _ = pts(o2, mir); best = 0
        for ang in np.deg2rad(np.arange(-20, 21, 2)):
            Br = rot(B, ang); f1 = np.mean(cKDTree(Br).query(A)[0] < TOL); f2 = np.mean(cKDTree(A).query(Br)[0] < TOL); best = max(best, (f1 + f2) / 2)
        vals.append(best); print(f'{k}: {best:.3f}')
    print(f'ceiling (mean of {len(vals)} eyes): {np.mean(vals):.3f}')
    if a.out: json.dump(dict(values=[float(v) for v in vals]), open(a.out, 'w'))

# ----------------------------------------------------------------------------- sensitivity of identity overlap
TOLS = (0.03, 0.06, 0.12)

def cmd_sensitivity(a):
    """Identity overlap for every person with both eyes (no quality exclusion), at three tolerances,
    against an independent random draw of unrelated people (seeded by --seed). Quality thresholds are
    applied afterwards by subsetting, and several seeds give independent control draws."""
    import time
    db = pickle.load(open(a.eyes, 'rb')); ids = sorted({k[:3] for k in db if k[:3] + 'OD' in db and k[:3] + 'OS' in db})
    P = {k: pts(db[k], k.endswith('OS')) for k in db}; rng = np.random.default_rng(a.seed)
    out = pickle.load(open(a.out, 'rb')) if os.path.exists(a.out) else dict(seed=a.seed, rows=[], done=set())
    t = time.time()
    for i in ids:
        if i in out['done']: continue
        if time.time() - t > a.budget: break
        others = np.random.default_rng([a.seed, int(i)]).choice([j for j in ids if j != i], a.others, replace=False)
        for j in [i] + list(others):
            A, wA = P[i + 'OD']; B, wB = P[j + 'OS']; oA, oB = db[i + 'OD'], db[j + 'OS']
            best = (-1, 0)
            for ang in np.deg2rad(np.arange(-20, 21, 2)):
                sc = score(A, wA, B, wB, oA, oB, ang, SELS['all'])
                if sc == sc and sc > best[0]: best = (sc, ang)
            res = {}
            for tol in TOLS:
                global TOL
                keep = TOL; TOL = tol
                res[tol] = tuple(score(A, wA, B, wB, oA, oB, best[1], SELS[c]) for c in ('all', 'thick', 'thin'))
                TOL = keep
            out['rows'].append((i, str(j), i == j, res))
        out['done'].add(i); pickle.dump(out, open(a.out, 'wb'))
    print(f'{a.out}: {len(out["done"])}/{len(ids)} people')

# ----------------------------------------------------------------------------- robustness summary
def cmd_robustness(a):
    """Combine sensitivity runs, the re-imaging ceiling and network statistics into the robustness
    analyses of the electronic supplementary material, and write a JSON."""
    rng = np.random.default_rng(0); db = pickle.load(open(a.eyes, 'rb'))
    runs = [pickle.load(open(f, 'rb')) for f in a.sens]; ceil = np.array(json.load(open(a.ceiling))['values'])
    ids = sorted({k[:3] for k in db if k[:3] + 'OD' in db and k[:3] + 'OS' in db})
    L = np.array([db[k]['m']['len'] for k in db]); med = np.median(L)
    def passes(i, f): return f is None or (db[i + 'OD']['m']['len'] >= f * med and db[i + 'OS']['m']['len'] >= f * med)
    def boot(v, B=4000): return np.percentile([rng.choice(v, len(v)).mean() for _ in range(B)], [2.5, 97.5])
    out = {'ceiling': dict(n=len(ceil), mean=float(ceil.mean()), lo=float(boot(ceil)[0]), hi=float(boot(ceil)[1]),
                           min=float(ceil.min()), max=float(ceil.max()))}
    classes = ('all', 'large', 'thin')
    def collect(runs_, tol, c, f):
        own, oth = {}, {}
        for r in runs_:
            for i, j, same, res in r['rows']:
                if not passes(i, f): continue
                v = res[tol][c]
                if v != v: continue
                if same: own[i] = v
                elif passes(j, f): oth.setdefault(i, []).append(v)
        ii = [i for i in own if len(oth.get(i, [])) >= 3]
        return ii, np.array([own[i] for i in ii]), [np.array(oth[i]) for i in ii]
    tab = []
    for f in (None, 0.3, 0.5, 0.7):
        for tol in TOLS:
            for c, cname in enumerate(classes):
                for label, rr in [('pooled', runs)] + [(f'draw{k + 1}', [r]) for k, r in enumerate(runs)]:
                    if label != 'pooled' and not (tol == 0.06 and f == 0.5): continue
                    ii, own, oth = collect(rr, tol, c, f)
                    d = own - np.array([o.mean() for o in oth]); lo, hi = boot(d)
                    pct = np.array([(np.sum(o < v) + 0.5 * np.sum(o == v)) / len(o) for v, o in zip(own, oth)])
                    plo, phi = boot(pct)
                    tab.append(dict(qc=f, tol=tol, cls=cname, controls=label, n=len(ii), own=float(own.mean()),
                                    others=float(np.mean([o.mean() for o in oth])), excess=float(d.mean()), lo=float(lo), hi=float(hi),
                                    p=float(wilcoxon(d).pvalue), percentile=float(pct.mean()), plo=float(plo), phi=float(phi)))
    out['identity'] = tab
    # normalized identity with joint bootstrap over people and ceiling eyes (qc 0.5, tol 0.06, pooled controls)
    norm = {}
    for c, cname in enumerate(classes):
        ii, own, oth = collect(runs, 0.06, c, 0.5); m = np.array([o.mean() for o in oth])
        val = np.mean((own - m) / (ceil.mean() - m)); bs = []
        for _ in range(4000):
            k = rng.integers(0, len(own), len(own)); cm = rng.choice(ceil, len(ceil)).mean()
            bs.append(np.mean((own[k] - m[k]) / (cm - m[k])))
        norm[cname] = dict(mean=float(val), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    out['normalized_identity'] = norm
    # percentile scale for network statistics (qc 0.5, full adjustment)
    ok = [i for i in ids if passes(i, 0.5)]
    for e in ('OD', 'OS'):
        for i in ok:
            n, w3 = crossings(db[i + e]); db[i + e]['m']['n2.5'] = n; db[i + e]['m']['w3_2.5'] = w3
    def cov(e): return [np.array([db[i + e][q] for i in ok], float) for q in ('contrast', 'qnoise')] + \
                       [np.array([db[i + e].get(q, np.nan) for i in ok], float) for q in ('axial', 'age', 'rd')]
    stats = {}
    for k, name in (('branch', 'branch points per area'), ('len', 'vessel length per area'), ('wmean', 'mean calibre'),
                    ('dens', 'vessel area fraction'), ('n2.5', 'vessels crossing 2.5 disc radii'), ('w3_2.5', 'Murray-type capacity')):
        x = resid(np.log(np.array([db[i + 'OD']['m'][k] for i in ok], float) + 1e-6), cov('OD'))
        y = resid(np.log(np.array([db[i + 'OS']['m'][k] for i in ok], float) + 1e-6), cov('OS'))
        g = np.isfinite(x) & np.isfinite(y); x, y = x[g], y[g]
        pct = np.array([np.mean(np.abs(x[j] - np.delete(y, j)) > abs(x[j] - y[j])) for j in range(len(x))])
        lo, hi = boot(pct); stats[name] = dict(n=int(len(x)), percentile=float(pct.mean()), lo=float(lo), hi=float(hi), values=pct.tolist())
    out['statistics_percentile'] = stats
    for c, cname in enumerate(classes):
        ii, own, oth = collect(runs, 0.06, c, 0.5)
        pct = np.array([(np.sum(o < v) + 0.5 * np.sum(o == v)) / len(o) for v, o in zip(own, oth)])
        lo, hi = boot(pct); out.setdefault('identity_percentile', {})[cname] = dict(n=len(ii), percentile=float(pct.mean()), lo=float(lo), hi=float(hi), values=pct.tolist())
    json.dump(out, open(a.out, 'w'))
    print('ceiling', {k: round(v, 3) for k, v in out['ceiling'].items()})
    for r in tab:
        if r['controls'] == 'pooled' or (r['cls'] == 'all'):
            print(f"qc={r['qc']} tol={r['tol']} {r['cls']:5s} {r['controls']:6s} n={r['n']} own={r['own']:.3f} oth={r['others']:.3f} excess={r['excess']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.2g} pct={r['percentile']:.3f} [{r['plo']:.3f},{r['phi']:.3f}]")
    print('normalized', norm)
    print('identity percentile', {k: (round(v['percentile'], 3), round(v['lo'], 3), round(v['hi'], 3)) for k, v in out['identity_percentile'].items()})
    print('statistics percentile', {k: (round(v['percentile'], 3), round(v['lo'], 3), round(v['hi'], 3)) for k, v in stats.items()})

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('overlap'); o.add_argument('--eyes', required=True); o.add_argument('--out', default='overlap.pkl'); o.add_argument('--others', type=int, default=10)
    m = sub.add_parser('metrics'); m.add_argument('--eyes', required=True)
    s_ = sub.add_parser('similarity'); s_.add_argument('--eyes', required=True); s_.add_argument('--overlap', required=True); s_.add_argument('--ceiling', type=float, default=0.85); s_.add_argument('--out', default='similarity.json')
    c_ = sub.add_parser('ceiling'); c_.add_argument('--eyes', required=True); c_.add_argument('--images', required=True); c_.add_argument('--n', type=int, default=5); c_.add_argument('--out', default=None)
    v_ = sub.add_parser('sensitivity'); v_.add_argument('--eyes', required=True); v_.add_argument('--seed', type=int, default=1); v_.add_argument('--others', type=int, default=10); v_.add_argument('--budget', type=float, default=1e9); v_.add_argument('--out', required=True)
    b_ = sub.add_parser('robustness'); b_.add_argument('--eyes', required=True); b_.add_argument('--sens', nargs='+', required=True); b_.add_argument('--ceiling', required=True); b_.add_argument('--out', required=True)
    t = sub.add_parser('trees'); t.add_argument('--eyes', required=True); t.add_argument('--gap', type=float, default=0.15); t.add_argument('--model-json')
    a = ap.parse_args()
    {'overlap': cmd_overlap, 'metrics': cmd_metrics, 'similarity': cmd_similarity, 'ceiling': cmd_ceiling, 'sensitivity': cmd_sensitivity, 'robustness': cmd_robustness, 'trees': cmd_trees}[a.cmd](a)

if __name__ == '__main__':
    main()
