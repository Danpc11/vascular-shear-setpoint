"""
retina_segment.py — Build the per-eye vessel database used by src/retina_bilateral.py.

Disc centre and radius come from the experts' disc polygons (--contours, PAPILA
ExpertsSegmentations/Contours); without that folder they are read from the yellow and blue
contours drawn on the "ImagesWithContours" images, which gives a median centre error of about
0.06 disc radii relative to the experts' polygons. Vessels come either from the built-in classical segmentation
(black top-hat + hysteresis) or from external masks such as AutoMorph outputs.

Usage
  # classical segmentation
  python src/retina_segment.py --images ImagesWithContours --contours Contours --clinical-od patient_data_od.xlsx \
      --clinical-os patient_data_os.xlsx --out eyes.pkl
  # external masks (binary vessel mask, optional artery/vein colour map), matched by RETxxxOD/OS in the file name
  python src/retina_segment.py --images ImagesWithContours --clinical-od ... --clinical-os ... \
      --vessel-glob "Results/M2/binary_vessel/raw_binary/*.png" \
      --av-glob "Results/M2/artery_vein/raw_binary/*.png" --out eyes_dl.pkl
  The folder names above are examples; point the globs to wherever your tool writes its masks.
  In the A/V map, red-dominant pixels are read as arteries and blue-dominant pixels as veins.
"""
import argparse, glob, os, pickle, re, time, warnings
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, io, measure, morphology, transform
warnings.filterwarnings('ignore')
S = 0.35          # working scale for the classical pipeline
R1, R2 = 1.3, 3.6 # analysis annulus, in disc radii

def eye_key(path):
    m = re.search(r'RET(\d+)(OD|OS)', os.path.basename(path)); return (m.group(1) + m.group(2)) if m else None

def healthy_pairs(od_xlsx, os_xlsx):
    od = pd.read_excel(od_xlsx, header=None, skiprows=3); os_ = pd.read_excel(os_xlsx, header=None, skiprows=3)
    od.index = od[0].astype(str).str[1:]; os_.index = os_[0].astype(str).str[1:]
    ids = [i for i in od.index if i in os_.index and int(od.loc[i, 3]) == 0 and int(os_.loc[i, 3]) == 0]
    clin = {}
    for i in ids:
        for e, t in (('OD', od), ('OS', os_)):
            clin[i + e] = dict(age=float(t.loc[i, 1]), axial=float(t.loc[i, 11]))
    return ids, clin

def disc_from_contours(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    cont = ((R > 0.75) & (G > 0.75) & (B < 0.35)) | ((B > 0.6) & (R < 0.35) & (G < 0.35))
    ys, xs = np.nonzero(cont); cy, cx = ys.mean(), xs.mean()
    d = np.hypot(ys - cy, xs - cx); outer = d > 0.7 * np.percentile(d, 99)
    cy, cx = ys[outer].mean(), xs[outer].mean()
    rd = np.median(np.hypot(ys[outer] - cy, xs[outer] - cx))
    return cy, cx, rd, cont

def disc_from_expert(k, folder):
    """Disc centre and equivalent radius (full resolution) averaged over the experts' polygons
    (PAPILA ExpertsSegmentations/Contours: RETxxxOD_disc_exp1.txt, ..._exp2.txt, columns x y)."""
    est = []
    for exp in (1, 2):
        p = os.path.join(folder, f'RET{k}_disc_exp{exp}.txt')
        if not os.path.exists(p): continue
        xy = np.loadtxt(p); x, y = xy[:, 0], xy[:, 1]; cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = 0.5 * cr.sum()
        cx = ((x + np.roll(x, -1)) * cr).sum() / (6 * a); cy = ((y + np.roll(y, -1)) * cr).sum() / (6 * a)
        est.append((cy, cx, np.sqrt(abs(a) / np.pi)))
    return tuple(np.mean(est, 0)) if est else None

def classical(im, cont, cy, cx, rd):
    im = transform.rescale(im, (S, S, 1), anti_aliasing=True); G = im[..., 1]
    contS = transform.rescale(cont.astype(float), S) > 0.05
    fov = ndi.binary_erosion(im.sum(2) > 0.15, iterations=20)
    g = G.copy(); g[ndi.binary_dilation(contS, iterations=2)] = np.median(G[fov]); g[~fov] = np.median(G[fov])
    bg = transform.resize(ndi.median_filter(g[::3, ::3], size=15), g.shape); gn = g / np.maximum(bg, 1e-3)
    th = ndi.gaussian_filter(morphology.black_tophat(gn, morphology.disk(8)), 0.7)
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]; rr = np.hypot(yy - cy, xx - cx) / rd
    roi = fov & (rr > 1.15)
    hi, lo = np.percentile(th[roi], [97, 86])
    ves = filters.apply_hysteresis_threshold(np.where(roi, th, 0), lo, hi)
    ves = ndi.binary_opening(ves, iterations=1)
    ves = morphology.remove_small_objects(ves, max_size=100)
    return ves, fov, float(np.std(gn[roi])), None

def external(vpath, avpath, shape_full, im_full):
    v = io.imread(vpath); v = v[..., 0] if v.ndim == 3 else v; ves = v > (v.max() / 2)
    sc = ves.shape[0] / shape_full[0]
    im = transform.resize(im_full, ves.shape + (3,), anti_aliasing=True)
    fov = ndi.binary_erosion(im.sum(2) > 0.15, iterations=max(3, int(20 * sc / S)))
    av = None
    if avpath:
        a = io.imread(avpath)[..., :3].astype(float)
        if a.shape[:2] != ves.shape: a = transform.resize(a, ves.shape + (3,), order=0, preserve_range=True)
        av = np.zeros(ves.shape, np.uint8); av[(a[..., 0] > a[..., 2]) & (a[..., 0] > 0)] = 1; av[(a[..., 2] > a[..., 0])] = 2
    return ves, fov, float(np.std(im[..., 1][fov])), av, sc

def finish(ves, fov, cy, cx, rd, contrast, av=None, minlen=70):
    sk0 = morphology.skeletonize(ves)
    lab = measure.label(ves); L = ndi.sum(sk0, lab, index=np.arange(1, lab.max() + 1))
    ves = np.isin(lab, 1 + np.nonzero(L >= minlen)[0])
    sk = morphology.skeletonize(ves); w = 2 * ndi.distance_transform_edt(ves)
    ys, xs = np.nonzero(sk)
    yy, xx = np.mgrid[:ves.shape[0], :ves.shape[1]]; rr = np.hypot(yy - cy, xx - cx) / rd
    ann = (rr > R1) & (rr < R2) & fov; area = ann.sum() / rd ** 2; ska = sk & ann
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3)), mode='constant')
    o = dict(cy=cy, cx=cx, rd=rd, shape=ves.shape, fov=fov, ys=ys, xs=xs, w=w[ys, xs],
             qnoise=1 - sk.sum() / max(sk0.sum(), 1), contrast=contrast,
             av=None if av is None else av[ys, xs],
             m=dict(dens=(ves & ann).sum() / max(ann.sum(), 1), len=ska.sum() / rd / max(area, 1e-9),
                    branch=(ska & (nb >= 4)).sum() / max(area, 1e-9), wmean=float(np.mean(w[ska] / rd)) if ska.any() else np.nan))
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True); ap.add_argument('--clinical-od', required=True); ap.add_argument('--clinical-os', required=True)
    ap.add_argument('--contours', help='folder with expert disc polygons (recommended)'); ap.add_argument('--vessel-glob'); ap.add_argument('--av-glob'); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='process only the first N patients (testing)')
    a = ap.parse_args()
    ids, clin = healthy_pairs(a.clinical_od, a.clinical_os)
    if a.limit: ids = ids[:a.limit]
    vmap = {eye_key(p): p for p in glob.glob(a.vessel_glob)} if a.vessel_glob else {}
    amap = {eye_key(p): p for p in glob.glob(a.av_glob)} if a.av_glob else {}
    db = pickle.load(open(a.out, 'rb')) if os.path.exists(a.out) else {}
    t = time.time()
    for i in ids:
        for e in ('OD', 'OS'):
            k = i + e
            if k in db: continue
            path = os.path.join(a.images, f'Opht_cont_RET{k}.jpg')
            if not os.path.exists(path): continue
            im = io.imread(path).astype(float) / 255
            cy, cx, rd, cont = disc_from_contours(im)
            if a.contours:
                ex = disc_from_expert(k, a.contours)
                if ex is not None: cy, cx, rd = ex
            if a.vessel_glob:
                if k not in vmap: continue
                ves, fov, con, av, sc = external(vmap[k], amap.get(k), im.shape[:2], im)
                db[k] = finish(ves, fov, cy * sc, cx * sc, rd * sc, con, av, minlen=int(70 * sc / S))
            else:
                ves, fov, con, av = classical(im, cont, cy * S, cx * S, rd * S)
                db[k] = finish(ves, fov, cy * S, cx * S, rd * S, con)
            db[k].update(clin[k])
        pickle.dump(db, open(a.out, 'wb'))
        print(f'{i} ok  ({len(db)} eyes, {time.time() - t:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
