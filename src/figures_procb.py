"""
figures_procb.py — Figures 1–4 of the manuscript from the outputs of ensembles.py, replicas.py,
retina_segment.py and retina_bilateral.py, and from the campaign table (summary.tsv).

Usage: python src/figures_procb.py --summary-tsv summary.tsv --out results/figures
"""
import argparse, json, os, pickle, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__))
from replicas import make_graph, Net, dense, staged, dtilde
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
CD, CS, CO = '#1b4f72', '#c0392b', '#e67e22'
R = 'results'
L_ = lambda f: pickle.load(open(f'{R}/{f}', 'rb'))

def tag(ax, s): ax.text(-0.14, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

def save(fig, out, name):
    fig.savefig(f'{out}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{out}/{name}.png', dpi=300, bbox_inches='tight'); plt.close(fig)

def draw_tree(ax, P, E, alive, r, hl, scale=3.5):
    idx = np.nonzero(alive)[0]; w = r[idx] / r[idx].max()
    ax.add_collection(LineCollection([P[E[e]] for e in idx], linewidths=0.3 + scale * w, colors=[CS if hl[e] else 'k' for e in idx]))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=0.4, color='0.7'))

def contingent(S):
    p = np.mean(S, 0); return ((p > 0) & (p < 1)).sum() / (p > 0).sum()

def figure1(out, tsv):
    fig = plt.figure(figsize=(7.0, 5.4)); gs = fig.add_gridspec(2, 6, hspace=0.45, wspace=0.9)
    ax = fig.add_subplot(gs[0, 0:2]); r = np.logspace(-2, 0, 50)
    for b, ls in ((1, '-'), (0.75, '--'), (0.5, '-.'), (0.25, ':')): ax.loglog(r, r ** (b - 1), ls, color='k', lw=1, label=f'b = {b:g}')
    ax.set_xlabel('radius r / r$_{max}$'); ax.set_ylabel(r'shear target $\tau_{set}/\tau_0$'); ax.legend(frameon=False); tag(ax, 'a')
    P, E, L = make_graph(250, 7); net = Net(P, E, L, 0.75); rng = np.random.default_rng(2)
    lr = np.log(0.05) + 0.1 * rng.standard_normal(net.m); alive = np.ones(net.m, bool); al = 0.375; lam = 1 / al; F = []
    for _ in range(5000):
        rr = np.exp(lr); f, _, w, _ = net.solve(rr, alive)
        F.append(np.sum(f[alive] ** 2 / w[alive]) + lam * np.sum(L[alive] ** (3 * 0.75 / 2) * w[alive] ** al))
        dz = np.where(alive, np.abs(f) / rr ** 3 / net.tauset(rr) - 1, 0)
        if np.max(np.abs(dz)) < 1e-5: break
        lr = lr + np.clip(0.2 * dz, -0.02, 0.02); alive &= lr > np.max(lr[alive]) + np.log(1e-3)
    ax = fig.add_subplot(gs[0, 2:4]); F = np.array(F); ax.semilogy(np.arange(len(F) - 1), F[:-1] - F[-1], color='k', lw=1)
    ax.set_xlabel('iteration'); ax.set_ylabel(r'$F - F_\infty$ (a.u.)'); ax.set_title('the local rule descends F', fontsize=7)
    P, E, L = make_graph(400, 3)
    n1 = Net(P, E, L, 0.75); a1, l1 = dense(n1, 0.1, np.random.default_rng(5))
    n2 = Net(P, E, L, 0.75); a2, l2 = staged(n2, 0.1, np.random.default_rng(5))
    d1, d2 = dtilde(n1, a1, l1), dtilde(n2, a2, l2)
    ax = fig.add_subplot(gs[1, 0:2]); draw_tree(ax, P, E, a1, np.exp(l1), a1 & ~a2); tag(ax, 'b'); ax.set_title(f'dense plexus (D̃ = {d1 / d2:.2f})', fontsize=7)
    ax = fig.add_subplot(gs[1, 2:4]); draw_tree(ax, P, E, a2, np.exp(l2), a2 & ~a1); ax.set_title('staged growth (D̃ = 1)', fontsize=7)
    T = pd.read_csv(tsv, sep='\t'); T['E'] = 100 * (T['D_ratio'] - 1)
    G = T[(T.exp == 'growth') & (T.growth_mode == 'peripheral') & (T.stages == 24) & (T.r_seed == 0.05)]
    ax = fig.add_subplot(gs[0, 4:])
    for lab, q, c in (('random radii', T[T.exp == 'starts'], '0.6'), ('dense plexus', T[T.exp == 'dense'], CD),
                      ('staged, no regrowth', G[G.reactivate == False], CO), ('staged + regrowth', G[G.reactivate == True], CS)):
        m = q.groupby(['n_sinks', 'b'])['E'].mean()
        for N, ls, mk in ((400, '-', 'o'), (700, '--', 's')):
            y = m.loc[N]; ax.plot(y.index, y.values, ls, marker=mk, ms=2.5, lw=1, color=c, label=lab if N == 400 else None)
    ax.set_yscale('log'); ax.set_xlabel('maintenance exponent b'); ax.set_ylabel('excess dissipation (%)'); tag(ax, 'c')
    ax.set_title('rest points vs best tree found', fontsize=7)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color='k', ls='-', marker='o', ms=2.5, lw=1), Line2D([], [], color='k', ls='--', marker='s', ms=2.5, lw=1)]; l += ['N = 400', 'N = 700']
    lx = fig.add_subplot(gs[1, 4:]); lx.axis('off'); lx.legend(h, l, frameon=False, fontsize=7, loc='upper left', title='panel c', title_fontsize=7)
    save(fig, out, 'Figure1'); return d1 / d2, int((a1 & ~a2).sum())

def figure2(out):
    P, E, L = make_graph(400, 3); fig = plt.figure(figsize=(7.0, 5.4)); gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.6); cmap = plt.get_cmap('viridis')
    for j, (fn, tt) in enumerate((('shared_dense_b0.75_eta0.1_N400_d3.pkl', 'dense plexus'), ('shared_staged_b0.75_eta0.1_N400_d3.pkl', 'staged growth'))):
        p = np.mean(L_(fn)['S'], 0); ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2]); idx = np.nonzero(p > 0)[0]
        ax.add_collection(LineCollection([P[E[e]] for e in idx], linewidths=0.4 + 1.6 * p[idx], colors=cmap(p[idx])))
        ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect('equal'); ax.axis('off'); ax.add_patch(plt.Circle((0, 0), 1, fill=False, lw=0.4, color='0.7'))
        ax.set_title(f'{tt}: {(p == 1).sum()} edges always present,\n{((p > 0) & (p < 1)).sum()} contingent (30 replicas)', fontsize=7)
        if j == 0: tag(ax, 'a')
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), ax=fig.axes, fraction=0.02, pad=0.01, location='right'); cb.set_label('survival probability', fontsize=7)
    d = [L_(f'shared_dense_b0.75_eta{e}_N400_d3.pkl') for e in ('0.01', '0.1', '0.3')]; D0 = np.mean(d[1]['D'])
    ax = fig.add_subplot(gs[1, 0]); eta = [1, 10, 30]
    ax.plot(eta, [contingent(x['S']) for x in d], 'o-', color=CD, ms=3, lw=1, label='contingent fraction')
    ax.plot(eta, [np.mean(x['D']) / D0 for x in d], 's--', color=CS, ms=3, lw=1, label='relative dissipation')
    ax.set_xscale('log'); ax.set_xticks(eta); ax.set_xticklabels(['1%', '10%', '30%']); ax.set_ylim(0, 1.15); ax.set_xlabel('initial noise')
    ax.legend(frameon=False, loc='lower right', fontsize=6, bbox_to_anchor=(1.08, 0.0)); tag(ax, 'b')
    ax = fig.add_subplot(gs[1, 1]); pr = [('dense', 'dense'), ('staged\nno regr.', 'staged_noregrow'), ('staged\n+ regr.', 'staged')]
    dd = [L_(f'shared_{m}_b0.75_eta0.1_N400_d3.pkl') for _, m in pr]; x = np.arange(3)
    ax.bar(x - 0.18, [np.mean(q['D']) / D0 for q in dd], 0.36, color=CS, label='relative dissipation')
    ax.bar(x + 0.18, [contingent(q['S']) for q in dd], 0.36, color=CD, label='contingent fraction')
    ax.set_xticks(x); ax.set_xticklabels([a for a, _ in pr], fontsize=6); ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.18)); tag(ax, 'c')
    ax = fig.add_subplot(gs[1, 2:]); its = [0, 10, 25, 50, 100, 200]; fs = [L_(f'fate_b0.75_eta0.1_N400_d{dm}.pkl') for dm in (3, 4, 5)]
    for q, lab, st in (('flow', 'flow', '-'), ('tau', 'shear', ':'), ('z', 'normalized shear', '--'), ('Dc', 'dissipation / cost', '-.'), ('r', 'radius', '-')):
        y = [np.nanmean([A[t][q] for f in fs for A in f['A']]) for t in its]
        ax.plot(its, y, st, marker='o', ms=2.5, lw=1, color='0.6' if q == 'r' else 'k', label=lab)
    ax.set_xlabel('iteration at which the signal is read'); ax.set_ylabel('AUC for final survival'); ax.set_ylim(0.5, 1.01); ax.legend(frameon=False, fontsize=6); tag(ax, 'd')
    save(fig, out, 'Figure2')

def figure3(out):
    js = json.load(open(f'{R}/ensembles_summary.json')); fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]
    for mode, c, lab in (('dense', CD, 'dense plexus'), ('staged', CS, 'staged growth')):
        for v in L_(f'indep_{mode}_b0.75_N800_ind.pkl').values():
            xs = np.sort(np.round(v['load'])); ax.loglog(xs, 1 - np.arange(len(xs)) / len(xs), color=c, lw=0.4, alpha=0.35)
        ax.plot([], [], color=c, lw=1, label=lab)
    ax.set_xlabel('downstream load (sinks supplied)'); ax.set_ylabel('P(load ≥ x)'); ax.legend(frameon=False); tag(ax, 'a'); ax.set_title('independent substrates, b = 3/4', fontsize=7)
    ax = axs[1]; bs = [1.0, 0.75, 0.5]
    for mode, c, mk in (('dense', CD, 'o'), ('staged', CS, 's')):
        g = [js[f'indep_{mode}_b{b}_N800_ind.pkl']['gamma'] for b in bs]; h = [js[f'indep_{mode}_b{b}_N800_ind.pkl']['hack'] for b in bs]
        ax.errorbar(bs, [x[0] for x in g], [x[1] for x in g], fmt=mk + '-', color=c, ms=3, lw=1, capsize=2, label=f'γ, {mode}')
        ax.errorbar(bs, [x[0] for x in h], [x[1] for x in h], fmt=mk + '--', color=c, ms=3, lw=1, capsize=2, mfc='white', label=f'Hack, {mode}')
    ax.set_xlabel('maintenance exponent b'); ax.set_ylabel('exponent'); ax.set_ylim(0.5, 1.0); ax.legend(frameon=False, fontsize=5.8, ncol=2, loc='upper center'); tag(ax, 'b')
    ax = axs[2]; x = np.arange(2)
    ax.bar(x - 0.18, [js['overlap_dense']['shared'], js['overlap_staged']['shared']], 0.36, color='0.35', label='shared substrate')
    ax.bar(x + 0.18, [js['overlap_dense']['independent'], js['overlap_staged']['independent']], 0.36, color='0.75', label='independent substrates')
    ax.set_xticks(x); ax.set_xticklabels(['dense plexus', 'staged growth']); ax.set_ylabel('spatial overlap between replicas'); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6); tag(ax, 'c')
    save(fig, out, 'Figure3')

def figure4(out):
    import retina_bilateral as RB
    db, ok = RB.load(f'{R}/eyes.pkl'); rows = L_('overlap.pkl'); sim = json.load(open(f'{R}/similarity.json'))
    same = np.array([r[2] for r in rows]); pid = np.array([r[0] for r in rows]); own = {r[0]: r[3] for r in rows if r[2]}
    vals = np.array([own[i] for i in ok]); lo_, hi_ = np.percentile(vals, [40, 60])
    mid = [i for i, v in zip(ok, vals) if lo_ <= v <= hi_]
    pick = max(mid, key=lambda i: len(RB.pts(db[i + 'OD'], False)[0]) + len(RB.pts(db[i + 'OS'], True)[0]))
    A, _ = RB.pts(db[pick + 'OD'], False); B, _ = RB.pts(db[pick + 'OS'], True)
    best = max(np.deg2rad(np.arange(-20, 21, 2)), key=lambda a: np.mean(cKDTree(RB.rot(B, a)).query(A)[0] < RB.TOL)); Br = RB.rot(B, best)
    fig = plt.figure(figsize=(7.0, 5.6)); gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.45, width_ratios=[1, 1.15], height_ratios=[1, 0.95])
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(A[:, 0], -A[:, 1], s=0.15, color='k', lw=0, label='right eye'); axA.scatter(Br[:, 0], -Br[:, 1], s=0.15, color=CS, lw=0, label='left eye, mirrored')
    for rr in (1, RB.R1, RB.R2): axA.add_patch(plt.Circle((0, 0), rr, fill=False, lw=0.5, color='0.6', ls='-' if rr == 1 else ':'))
    axA.set_aspect('equal'); axA.set_xlim(-3.7, 3.7); axA.set_ylim(-3.7, 3.7); axA.set_xlabel('disc radii'); axA.set_yticks([]); axA.spines['left'].set_visible(False)
    axA.legend(frameon=False, markerscale=20, fontsize=6, loc='upper left', bbox_to_anchor=(0.06, 1.17), ncol=2)
    axB = fig.add_subplot(gs[0, 1]); rng = np.random.default_rng(1); names = ['identity, all', 'identity, large', 'identity, thin']
    for j, n in enumerate(names):
        v = np.array(sim[n]['values']); axB.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=3, color='0.55', lw=0)
        axB.errorbar(j, sim[n]['mean'], [[sim[n]['mean'] - sim[n]['lo']], [sim[n]['hi'] - sim[n]['mean']]], fmt='o', color=CS, ms=4, capsize=3, zorder=3)
    axB.axhline(0, color='k', lw=0.6); axB.axhline(1, color='k', lw=0.6, ls='--')
    axB.text(2.55, 1.0, 'identical\nimage', ha='left', va='center', fontsize=6.5, clip_on=False); axB.text(2.55, 0.0, 'two unrelated\npeople', ha='left', va='center', fontsize=6.5, clip_on=False)
    axB.set_xticks(range(3)); axB.set_xticklabels(['all', 'large', 'thin']); axB.set_xlim(-0.5, 2.5); axB.set_ylim(-0.3, 1.12)
    axB.set_ylabel('vessel identity shared\nby the two eyes'); axB.set_title(f'each dot: one person (n = {len(sim[names[0]]["values"])})', fontsize=6.5)
    axC = fig.add_subplot(gs[1, :])
    stats = ['branch points per area', 'vessel length per area', 'mean calibre', 'vessel area fraction', 'vessels crossing 2.5 disc radii', 'Murray-type capacity']
    labels = ['vessel identity, all', 'vessel identity, large', 'vessel identity, thin'] + stats
    ys = np.arange(len(labels))[::-1].astype(float); ys[:3] += 0.6
    for y, n, lab in zip(ys, names + stats, labels):
        s = sim[n]; col = CS if n.startswith('identity') else CD
        axC.plot([s['lo'], s['hi']], [y, y], color=col, lw=1.2); axC.plot(s['mean'], y, 'o', ms=4.5, color=col, mfc=col if s['lo'] > 0 else 'white')
    axC.axvline(0, color='k', lw=0.6); axC.axvline(1, color='k', lw=0.6, ls='--')
    axC.set_yticks(ys); axC.set_yticklabels(labels, fontsize=7); axC.set_xlim(-0.15, 1.0); axC.set_ylim(ys[-1] - 0.7, ys[0] + 1.0)
    axC.set_xlabel('similarity of the two eyes\n(0 = two unrelated people, 1 = identical)')
    axC.text(0.98, ys[1], 'vessel layout:\nalmost entirely contingent', ha='right', va='center', fontsize=7, color=CS, fontweight='bold')
    axC.text(0.98, ys[5], 'network statistics:\nlargely individual-specific', ha='right', va='center', fontsize=7, color=CD, fontweight='bold')
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    left = inv.transform(axC.get_tightbbox(r))[0, 0]; right = axC.get_position().x1; pa, pb = axA.get_position(), axB.get_position()
    wa = pa.height * fig.get_figheight() / fig.get_figwidth(); axA.set_position([left, pa.y0, wa, pa.height])
    axB.set_position([left + wa + 0.12, pb.y0, right - (left + wa + 0.12) - 0.085, pb.height]); fig.canvas.draw()
    bl = inv.transform(axB.get_tightbbox(r))[0, 0]; pa, pc = axA.get_position(), axC.get_position()
    fig.text(left, pa.y1 + 0.035, 'a', fontsize=10, fontweight='bold'); fig.text(bl, pa.y1 + 0.035, 'b', fontsize=10, fontweight='bold'); fig.text(left, pc.y1 + 0.02, 'c', fontsize=10, fontweight='bold')
    save(fig, out, 'Figure4'); return pick

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--summary-tsv', required=True); ap.add_argument('--out', default=f'{R}/figures')
    ap.add_argument('--only', default='1234'); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if '1' in a.only: print('Figure 1: dense/staged example D ratio %.2f, %d differing edges' % figure1(a.out, a.summary_tsv))
    if '2' in a.only: figure2(a.out); print('Figure 2 done')
    if '3' in a.only: figure3(a.out); print('Figure 3 done')
    if '4' in a.only: print('Figure 4: example individual', figure4(a.out))
