"""Záverečný backtest gólového modelu.

  prototyp   = stacking(sila, Elo) + kalibrácia 1/X/2, všetko učené len z minulosti (po rokoch)
  web-like   = replika prístupu webu: len ESPN dáta 6 rokov, nezávislé priory útok/obrana,
               dodatočné roztiahnutie k=1,46 (g=0,25), ρ=−0,06
  trh        = záverečné kurzy DraftKings (bez marže) tam, kde ich ESPN má

Výstup: results/goals_backtest.json + data/raw/cache/final_goals.npz + results/calibration.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calib as C
import engine as E
import walk as W

CAL_T = ["UEFA Nations League", "UEFA Euro qualification", "UEFA Euro", "FIFA World Cup"]
SITE_COMPS = ["uefa.nations", "fifa.worldq.uefa", "uefa.euroq", "uefa.euro", "fifa.world", "fifa.friendly"]


def calpop(w: pd.DataFrame) -> np.ndarray:
    """Zápasy „ako tie, čo predpovedáme“: súťažné zápasy zóny UEFA + MS a ME."""
    uefa_t = w.tournament.isin(["UEFA Euro qualification", "UEFA Nations League"])
    uefa = set(w.home[uefa_t]) | set(w.away[uefa_t])
    both = w.home.isin(uefa) & w.away.isin(uefa)
    return (w.tournament.isin(CAL_T) | w.comp.isin(E.EVAL_COMPS) | ((w.tournament == "FIFA World Cup qualification") & both)).values


def short(m):
    return {k: round(v, 4) for k, v in m.items()}


def boot_diff(a, b, n=4000, seed=1):
    """95 % bootstrap interval priemerného párového rozdielu a − b."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


if __name__ == "__main__":
    w = E.load_world()
    cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
    rho = cfg["rho"]
    z = np.load(E.PROC / "wide_walk.npz")
    rows = z["rows"]
    ls_h, ls_a, le_h, le_a = z["lh"], z["la"], z["eh"], z["ea"]
    FEATS = [(z["lh"], z["la"]), (z["sh"], z["sa"]), (z["eh"], z["ea"])]      # sila z gólov, sila zo striel, Elo
    sub = lambda mask: [(a[mask], b[mask]) for a, b in FEATS]
    hg, ag = w.hg.values[rows], w.ag.values[rows]
    hf = (~w.neutral.values[rows]).astype(float)
    year = w.dt.dt.year.values[rows]
    date = w.date.values[rows]
    pop = calpop(w)[rows]
    ev = np.isin(w.comp.values[rows], E.EVAL_COMPS) & (date >= "2018-08-01")
    o = C.outcome(hg, ag)

    n = len(rows)
    ch, ca = np.full(n, np.nan), np.full(n, np.nan)
    M = np.full((n, E.MAXG, E.MAXG), np.nan)
    P = np.full((n, 3), np.nan)
    cals = {}
    for y in range(2018, 2027):
        tr = pop & (year < y)
        cal = C.fit_calibration(sub(tr), hf[tr], hg[tr], ag[tr], rho)
        cals[y] = cal
        sel = ev & (year == y)
        if sel.any():
            ch[sel], ca[sel], M[sel], P[sel] = C.predict_calibrated(cal, sub(sel), hf[sel], rho)
        print(y, "train", int(tr.sum()), "beta", cal[0].round(3), "theta", cal[1].round(3), flush=True)
    # kalibráciu pre ostré predpovede (uefa + world) počíta calibrate_live.py

    # ---------------------------------------------------------------- replika prístupu webu
    ws = w[w.comp.isin(SITE_COMPS) & (w.date >= "2016-07-01")].reset_index(drop=True)
    ws.attrs["teams"] = w.attrs["teams"]
    srows = np.flatnonzero((ws.comp.isin(E.EVAL_COMPS) & (ws.date >= "2018-08-01")).values)
    best = None
    for sig in (0.2, 0.3, 0.45):
        for hl in (500, 800, 1200):
            c = dict(halflife=hl, window=6.0, imp=(0.5, 1.0, 1.0, 1.0), sigma_s=sig / np.sqrt(2), sigma_t=sig / np.sqrt(2), cap=10, sot_mix=0.7)
            bh, ba = W.walk_strength(ws, srows, c)
            sh, sa = E.stretch(bh, ba, 1.46, 0.25)
            tune_mask = ws.date.values[srows] < "2024-01-01"
            # k*, ktoré by na ladiacom období vyšlo ako optimum – web má 1,46
            ks = [(E.metrics(*E.stretch(bh[tune_mask], ba[tune_mask], k, 0.25), ws.hg.values[srows][tune_mask], ws.ag.values[srows][tune_mask], rho)["logloss"], k)
                  for k in (1.0, 1.15, 1.3, 1.46, 1.6, 1.8)]
            kstar = min(ks)[1]
            ll = E.metrics(sh[tune_mask], sa[tune_mask], ws.hg.values[srows][tune_mask], ws.ag.values[srows][tune_mask], rho)["logloss"]
            print(f"web-like σ={sig} hl={hl}: optimum k*={kstar}, logloss ladenie pri k=1.46: {ll:.4f}", flush=True)
            cand = (abs(kstar - 1.46), ll, sig, hl, sh, sa)
            if best is None or cand[:2] < best[:2]:
                best = cand
    _, _, sig, hl, sh, sa = best
    print(f"web-like vybrané: σ={sig}, hl={hl} (optimum roztiahnutia najbližšie k 1,46)")
    site = {eid: (a, b) for eid, a, b in zip(ws.espn_id.values[srows], sh, sa)}

    # ---------------------------------------------------------------- porovnanie na rovnakých zápasoch
    eid = w.espn_id.values[rows]
    both = ev & np.array([e in site for e in eid]) & np.isfinite(ch)
    s_h = np.array([site[e][0] if e in site else np.nan for e in eid])
    s_a = np.array([site[e][1] if e in site else np.nan for e in eid])
    out = dict(config=cfg, site_like=dict(sigma=sig, halflife=hl, k=1.46, g=0.25))
    ll_proto = -np.log(np.maximum(P[np.arange(n), o], 1e-12))
    for name, mk in (("ladenie", both & (date < "2024-01-01")), ("test", both & (date >= "2024-01-01"))):
        res = dict(
            sila=E.metrics(ls_h[mk], ls_a[mk], hg[mk], ag[mk], rho),
            elo=E.metrics(le_h[mk], le_a[mk], hg[mk], ag[mk], rho),
            web_like=E.metrics(s_h[mk], s_a[mk], hg[mk], ag[mk], rho),
            prototyp=C.matrix_metrics(M[mk], P[mk], hg[mk], ag[mk]),
        )
        ll_site = E.per_match_logloss(s_h[mk], s_a[mk], hg[mk], ag[mk], rho)[0]
        res["rozdiel_prototyp_minus_web_like"] = boot_diff(ll_proto[mk], ll_site)
        out[name] = res
        print(f"== {name} (n={int(mk.sum())})")
        for k_, v in res.items():
            print(f"   {k_:10s}", short(v) if isinstance(v, dict) else [round(x, 4) for x in v])

    # ---------------------------------------------------------------- trh
    oh, od, oa = (w[c].values[rows] for c in ("oh", "od", "oa"))
    has = both & np.isfinite(oh) & np.isfinite(od) & np.isfinite(oa)
    Pm = np.full((n, 3), np.nan)
    Pm[has] = E.devig(oh[has], od[has], oa[has])
    ll_mkt = -np.log(np.maximum(Pm[np.arange(n), o], 1e-12))
    ll_site_all = np.full(n, np.nan)
    ll_site_all[both] = E.per_match_logloss(s_h[both], s_a[both], hg[both], ag[both], rho)[0]
    mk_res = {}
    for name, mk in (("ladenie", has & (date < "2024-01-01")), ("test", has & (date >= "2024-01-01"))):
        r = dict(n=int(mk.sum()), trh=float(ll_mkt[mk].mean()), prototyp=float(ll_proto[mk].mean()), web_like=float(ll_site_all[mk].mean()))
        for a in (0.5, 0.7, 0.85):
            Q = Pm[mk] ** a * P[mk] ** (1 - a)
            Q /= Q.sum(1, keepdims=True)
            r[f"mix_{a}"] = float(-np.mean(np.log(Q[np.arange(mk.sum()), o[mk]])))
            Qs = Pm[mk] ** a * E.outcome_probs(E.score_matrix(s_h[mk], s_a[mk], rho)) ** (1 - a)
            Qs /= Qs.sum(1, keepdims=True)
            r[f"web_mix_{a}"] = float(-np.mean(np.log(Qs[np.arange(mk.sum()), o[mk]])))
        r["rozdiel_prototyp_minus_trh"] = boot_diff(ll_proto[mk], ll_mkt[mk])
        mk_res[name] = r
        print(f"== trh {name}:", {k_: (round(v, 4) if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k_, v in r.items()})
    out["trh"] = mk_res
    (E.ROOT / "results" / "goals_backtest.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez(E.ROOT / "data" / "raw" / "cache" / "final_goals.npz", rows=rows, ch=ch, ca=ca, P=P, ev=ev, Pm=Pm, s_h=s_h, s_a=s_a)
