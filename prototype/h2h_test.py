"""Test: nesú vzájomné zápasy (H2H) informáciu nad rámec sily tímov?

Pre každý zápas vezmeme out-of-sample predpoveď a pozrieme sa, či odchýlka (skutočnosť − predpoveď)
v PREDCHÁDZAJÚCICH vzájomných zápasoch tej istej dvojice (do 10 rokov) predpovedá odchýlku v tomto zápase.
Ak je korelácia ≈ 0, faktor H2H iba pridáva šum."""
import json

import numpy as np

import engine as E
import walk as W

if __name__ == "__main__":
    w = E.load_world()
    cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
    z = np.load(E.PROC / "wide_walk.npz")
    rows = z["rows"]
    lh, la = W.blend(z["lh"], z["eh"], cfg["elo_weight"]), W.blend(z["la"], z["ea"], cfg["elo_weight"])
    h, a, day = w.h.values[rows], w.a.values[rows], w.day.values[rows]
    hg, ag = w.hg.values[rows], w.ag.values[rows]
    hist = {}
    cur_tot, prev_tot, cur_gd, prev_gd, nprev = [], [], [], [], []
    for i in np.argsort(day, kind="stable"):
        key = (min(h[i], a[i]), max(h[i], a[i]))
        sign = 1 if h[i] == key[0] else -1            # orientácia na tím s nižším indexom
        r_tot = (hg[i] + ag[i]) - (lh[i] + la[i])
        r_gd = sign * ((hg[i] - ag[i]) - (lh[i] - la[i]))
        past = [p for p in hist.get(key, []) if day[i] - p[0] <= 3653]
        if past:
            cur_tot.append(r_tot)
            prev_tot.append(np.mean([p[1] for p in past]))
            cur_gd.append(r_gd)
            prev_gd.append(np.mean([p[2] for p in past]))
            nprev.append(len(past))
        hist.setdefault(key, []).append((day[i], r_tot, r_gd))
    cur_tot, prev_tot, cur_gd, prev_gd, nprev = map(np.array, (cur_tot, prev_tot, cur_gd, prev_gd, nprev))
    out = {}
    for name, x, y in (("súčet gólov", prev_tot, cur_tot), ("gólový rozdiel", prev_gd, cur_gd)):
        for label, mk in (("všetky", nprev >= 1), ("≥3 vzájomné", nprev >= 3)):
            r = float(np.corrcoef(x[mk], y[mk])[0, 1])
            n = int(mk.sum())
            se = 1 / np.sqrt(n - 3)
            slope = float(np.polyfit(x[mk], y[mk], 1)[0])
            out[f"{name} / {label}"] = dict(n=n, r=r, ci95=[r - 1.96 * se, r + 1.96 * se], slope=slope)
            print(f"{name:15s} {label:12s} n={n:5d}  korelácia r={r:+.3f}  (95 % CI {r - 1.96 * se:+.3f} … {r + 1.96 * se:+.3f}), sklon {slope:+.3f}")
    (E.ROOT / "results" / "h2h_test.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
