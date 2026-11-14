"""Priame porovnanie: model stránky (jej vlastné λ „len model“) vs prototyp vs trh, na tých istých
nadchádzajúcich reprezentačných zápasoch (snapshot z webu). Trh = najlepší dostupný odhad pravdy,
takže menšia vzdialenosť od trhu = lepší model. Výstup: results/site_comparison.json"""
from __future__ import annotations

import json

import numpy as np

import engine as E
import predict as PR
from build_dataset import canon


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(p * np.log(p / q)))


if __name__ == "__main__":
    snap = json.loads((E.PROC / "site_snapshot.json").read_text())
    pr = PR.Predictor(with_counts=False, with_scorers=False)
    rows = []
    for eid, a in snap.items():
        fx, L = a["fixture"], a["lambdas"]
        if fx["state"] != "pre" or not L.get("market") or fx["date"] < "2026-11-14T16:00Z":
            continue
        home, away = canon(a["home"]["name"]), canon(a["away"]["name"])
        if home not in pr.tid or away not in pr.tid:
            continue
        x12 = {}
        for book in ("Fortuna", "DraftKings"):
            if book in a["odds"].get("1x2", {}):
                x12 = a["odds"]["1x2"][book]
                break
        if not x12:
            continue
        pm = E.devig(x12["1"], x12["X"], x12["2"])
        site_f = [f for f in a["factors"] if f["key"] != "market"][-1]
        ps = np.array([site_f["p1"], site_f["px"], site_f["p2"]])
        g = pr.goals(home, away, bool(fx.get("neutral")))
        rows.append(dict(id=eid, date=fx["date"], home=home, away=away,
                         trh=[round(float(x), 4) for x in pm], web=[round(float(x), 4) for x in ps], prototyp=[round(float(x), 4) for x in g["P"]],
                         lam_trh=[round(x, 3) for x in L["market"]], lam_web=[round(x, 3) for x in L["model"]], lam_prototyp=[round(g["lh"], 3), round(g["la"], 3)],
                         kl_web=kl(pm, ps), kl_prototyp=kl(pm, g["P"])))
    n = len(rows)
    kw, kp = np.array([r["kl_web"] for r in rows]), np.array([r["kl_prototyp"] for r in rows])
    lt, lw, lp = (np.array([r[k] for r in rows]) for k in ("lam_trh", "lam_web", "lam_prototyp"))
    rng = np.random.default_rng(0)
    d = kp - kw
    bs = d[rng.integers(0, n, (4000, n))].mean(1)
    out = dict(n=n,
               kl_web=float(kw.mean()), kl_prototyp=float(kp.mean()), rozdiel=float(d.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
               prototyp_blizsie_v=int((kp < kw).sum()),
               mae_p_web=float(np.mean(np.abs(np.array([r["web"] for r in rows]) - np.array([r["trh"] for r in rows])))),
               mae_p_prototyp=float(np.mean(np.abs(np.array([r["prototyp"] for r in rows]) - np.array([r["trh"] for r in rows])))),
               total_trh=float(lt.sum(1).mean()), total_web=float(lw.sum(1).mean()), total_prototyp=float(lp.sum(1).mean()),
               mae_lam_web=float(np.mean(np.abs(lw - lt))), mae_lam_prototyp=float(np.mean(np.abs(lp - lt))), zapasy=rows)
    (E.ROOT / "results" / "site_comparison.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"zápasov: {n}")
    print(f"KL(trh ‖ model): web {out['kl_web']:.4f}  prototyp {out['kl_prototyp']:.4f}  rozdiel {out['rozdiel']:+.4f} CI95 {out['ci95']}; prototyp bližšie v {out['prototyp_blizsie_v']}/{n}")
    print(f"priem. |Δp| voči trhu: web {out['mae_p_web']:.4f}  prototyp {out['mae_p_prototyp']:.4f}")
    print(f"priem. súčet λ: trh {out['total_trh']:.2f}  web {out['total_web']:.2f}  prototyp {out['total_prototyp']:.2f};  |Δλ|: web {out['mae_lam_web']:.3f} prototyp {out['mae_lam_prototyp']:.3f}")
    for r in rows:
        print(f"  {r['home'][:16]:16s}-{r['away'][:16]:16s} trh {r['trh']} web {r['web']} proto {r['prototyp']} | λ trh {r['lam_trh']} web {r['lam_web']} proto {r['lam_prototyp']}")
