"""
mesentery_shear.py — Shear, radius, segment length and pressure in the rat mesenteric network
of Pries and Secomb (network of 28-10-90, 546 segments), distributed with
github.com/secomb/NetFlowV2.

Flows are solved from the measured boundary conditions with the in vivo viscosity law
(Pries et al. 1994); subsegments are rejoined into vessel segments, and segments are classified
as arteriolar or venular from the topology of flow divergence. The script reports

  * the dissipation-to-cost law  D_e (4 - m_e) ~ r_e^(2b) l_e^(p'),  m_e = dln(mu)/dln(D),
    which holds for any length scaling of the maintenance cost, and the radius-only shear fit;
  * how much of the variation of log shear is explained by intravascular pressure, by radius,
    and by both together.

In arterioles pressure explains far more of the shear variation than radius does, which is why
the manuscript tests the set-point rule through network architecture rather than shear profiles.

Usage: python src/mesentery_shear.py   (downloads network.dat if absent)
"""
import os, urllib.request
from collections import defaultdict, Counter
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

URL = "https://raw.githubusercontent.com/secomb/NetFlowV2/master/network.dat"
MMHG = 133.322
MU_PLASMA = 1.1e-3  # Pa s

def load(path="network.dat"):
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    L = open(path, encoding="latin1").read().splitlines()
    nseg = int(L[6].split()[0])
    segs = []
    for l in L[8:8 + nseg]:
        t = l.split()
        segs.append(dict(name=int(t[0]), a=int(t[2]), b=int(t[3]), D=float(t[4]), H=float(t[6])))
    i = 8 + nseg
    nn = int(L[i].split()[0])
    nodes = {int(l.split()[0]): np.array(list(map(float, l.split()[1:4]))) for l in L[i + 2:i + 2 + nn]}
    j = i + 2 + nn
    nb = int(L[j].split()[0])
    bnd = {}
    for l in L[j + 2:j + 2 + nb]:
        t = l.split()
        bnd[int(t[0])] = (int(t[1]), float(t[2]))
    return segs, nodes, bnd

def mu_rel(D, H):
    """In vivo apparent viscosity relative to plasma (Pries et al. 1994)."""
    m45 = 6 * np.exp(-0.085 * D) + 3.2 - 2.44 * np.exp(-0.06 * D ** 0.645)
    s = 1 / (1 + 1e-11 * D ** 12)
    C = (0.8 + np.exp(-0.075 * D)) * (-1 + s) + s
    x = (D / (D - 1.1)) ** 2
    return (1 + (m45 - 1) * ((1 - H) ** C - 1) / ((1 - 0.45) ** C - 1) * x) * x

def dlnmu_dlnD(D, H, h=1e-4):
    return (np.log(mu_rel(D * (1 + h), H)) - np.log(mu_rel(D * (1 - h), H))) / (np.log(1 + h) - np.log(1 - h))

def solve_flow(segs, nodes, bnd):
    ids = sorted(nodes); ix = {n: k for k, n in enumerate(ids)}
    A = lil_matrix((len(ids), len(ids))); rhs = np.zeros(len(ids))
    for s in segs:
        s["L"] = np.linalg.norm(nodes[s["a"]] - nodes[s["b"]]) * 1e-6
        s["mu"] = MU_PLASMA * mu_rel(s["D"], s["H"])
        s["R"] = 128 * s["mu"] * s["L"] / (np.pi * (s["D"] * 1e-6) ** 4)
        a, b, g = ix[s["a"]], ix[s["b"]], 1 / s["R"]
        A[a, a] += g; A[b, b] += g; A[a, b] -= g; A[b, a] -= g
    for n, (bt, val) in bnd.items():
        k = ix[n]
        if bt == 0:
            A[k, :] = 0; A[k, k] = 1; rhs[k] = val * MMHG
        else:
            rhs[k] += val * 1e-12 / 60
    p = spsolve(A.tocsr(), rhs)
    for s in segs:
        s["q"] = (p[ix[s["a"]]] - p[ix[s["b"]]]) / s["R"]
    return {n: p[ix[n]] for n in ids}

def merge_subsegments(segs, P):
    """Rejoin the artificial subsegments (nodes >= 5000) into the original vessel segments."""
    adj = defaultdict(list)
    for s in segs:
        adj[s["a"]].append(s); adj[s["b"]].append(s)
    used, out = set(), []
    for s in segs:
        if s["name"] in used:
            continue
        chain = [s]; used.add(s["name"])
        for end in (s["a"], s["b"]):
            n, prev = end, s
            while n >= 5000:
                nxt = [t for t in adj[n] if t["name"] != prev["name"]][0]
                if nxt["name"] in used:
                    break
                chain.append(nxt); used.add(nxt["name"]); prev = nxt
                n = nxt["b"] if nxt["a"] == n else nxt["a"]
        cnt = Counter()
        for c in chain:
            cnt[c["a"]] += 1; cnt[c["b"]] += 1
        e0, e1 = [n for n, k in cnt.items() if k == 1]
        up, dn = (e0, e1) if P[e0] > P[e1] else (e1, e0)
        D, H = chain[0]["D"], chain[0]["H"]
        Q = abs(chain[0]["q"]); mu = MU_PLASMA * mu_rel(D, H)
        out.append(dict(D=D, H=H, L=sum(c["L"] for c in chain), Q=Q,
                        diss=sum(c["q"] ** 2 * c["R"] for c in chain),
                        mu=mu, m=float(dlnmu_dlnD(D, H)),
                        tau=32 * mu * Q / (np.pi * (D * 1e-6) ** 3),
                        P=(P[up] + P[dn]) / 2 / MMHG, up=up, dn=dn))
    return out

def classify(out, bnd):
    """Arteriolar: both ends diverging (or inflow/outflow boundary); venular: both ends converging."""
    inc, outg = Counter(), Counter()
    for s in out:
        outg[s["up"]] += 1; inc[s["dn"]] += 1
    def kind(n):
        if n in bnd:
            return "bin" if (bnd[n][0] == 2 and bnd[n][1] > 0) else "bout"
        if inc[n] == 1 and outg[n] >= 2: return "div"
        if inc[n] >= 2 and outg[n] == 1: return "conv"
        return "other"
    for s in out:
        ku, kd = kind(s["up"]), kind(s["dn"])
        if ku in ("div", "bin") and kd in ("div", "bout") and not (ku == "bin" and kd == "bout"):
            s["cls"] = "A"
        elif ku in ("conv", "bin") and kd in ("conv", "bout"):
            s["cls"] = "V"
        else:
            s["cls"] = "C"

def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta, 1 - (y - X1 @ beta).var() / y.var()

def boot_ci(y, X, f, B=2000, seed=1):
    rng = np.random.default_rng(seed); n = len(y)
    vals = [f(ols(y[i], [x[i] for x in X])[0]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(vals, [2.5, 97.5], axis=0)

def main():
    segs, nodes, bnd = load()
    P = solve_flow(segs, nodes, bnd)
    out = merge_subsegments(segs, P)
    classify(out, bnd)
    print("segments:", len(out), dict(Counter(s["cls"] for s in out)))
    for grp in ["A", "V", "ALL"]:
        S = [s for s in out if grp == "ALL" or s["cls"] == grp]
        r = np.array([s["D"] / 2 for s in S]); L = np.array([s["L"] for s in S])
        tau = np.array([s["tau"] for s in S]); m = np.array([s["m"] for s in S])
        diss = np.array([s["diss"] for s in S]); Pm = np.array([s["P"] for s in S])
        lr, lL = np.log(r), np.log(L)
        print(f"\n[{grp}] n = {len(S)}")
        b, r2 = ols(np.log(tau), [lr])
        print(f"  Newtonian shear, radius only: b = {1 + b[1]:.2f}  95% CI {boot_ci(np.log(tau), [lr], lambda b: 1 + b[1]).round(2)}  R2 = {r2:.2f}")
        y = np.log(diss * (4 - m))
        b, r2 = ols(y, [lr, lL])
        ci = boot_ci(y, [lr, lL], lambda b: np.array([b[1] / 2, b[2]]))
        print(f"  Dissipation-cost law: b = {b[1] / 2:.2f} {ci[:, 0].round(2)}, p' = {b[2]:.2f} {ci[:, 1].round(2)}  R2 = {r2:.2f}")
        b1, r21 = ols(np.log(tau), [np.log(Pm)]); b2, r22 = ols(np.log(tau), [np.log(Pm), lr])
        print(f"  Shear vs pressure: R2 = {r21:.2f}; pressure + radius: R2 = {r22:.2f} (P exp {b2[1]:.2f}, r exp {b2[2]:.2f})")

if __name__ == "__main__":
    main()
