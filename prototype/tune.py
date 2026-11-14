"""Ladiaca slučka gólového modelu.

Pravidlá proti preučeniu:
* všetky predpovede sú walk-forward (model vidí len minulosť),
* nastavenia sa VYBERAJÚ len na období LADENIE (2018-08 … 2023-12),
* obdobie TEST (2024-01 … 2026-11) sa používa len na záverečné porovnanie.

Spustenie:  python3 tune.py            (všetky fázy, výsledok do results/goals_tuning.json)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import engine as E
import walk as W

CACHE = E.ROOT / "data" / "raw" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RES = E.ROOT / "results"
RES.mkdir(exist_ok=True)

w = E.load_world()
ALL = W.eval_rows(w, "2018-08-01", "2026-12-31")
DATE = w.date.values[ALL]
TUNE = DATE < "2024-01-01"
TEST = ~TUNE
HG, AG = w.hg.values[ALL], w.ag.values[ALL]


def _key(tag: str, cfg: dict) -> str:
    return tag + "_" + hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def lam_strength(cfg: dict):
    f = CACHE / (_key("str", cfg) + ".npz")
    if f.exists():
        z = np.load(f)
        return z["lh"], z["la"]
    lh, la = W.walk_strength(w, ALL, cfg)
    np.savez(f, lh=lh, la=la)
    return lh, la


def J(lh, la, mask, rho=0.0):
    m = E.metrics(lh[mask], la[mask], HG[mask], AG[mask], rho)
    return m["logloss"] + 0.1 * m["score_nll"], m


def _run(cfg):
    lh, la = lam_strength(cfg)
    return cfg, J(lh, la, TUNE)


def short(m):
    return {k: round(v, 4) for k, v in m.items() if k != "n"}


if __name__ == "__main__":
    out = {}
    t0 = time.time()
    # ---------------------------------------------------------------- fáza A: model sily
    grid = [dict(halflife=hl, sigma_s=1.5, sigma_t=st, sot_mix=g, imp=(f, 0.8, 1.0, 1.0))
            for hl in (1000, 1300, 1700) for st in (0.2, 0.3) for g in (0.6, 0.75, 1.0) for f in (0.4, 0.7)]
    with Pool(4) as p:
        res = p.map(_run, grid)
    res.sort(key=lambda r: r[1][0])
    print(f"[A] {len(grid)} nastavení modelu sily, {time.time() - t0:.0f}s – najlepších 6 (ladenie):")
    for cfg, (j, m) in res[:6]:
        print("   ", {k: v for k, v in cfg.items()}, "J", round(j, 4), short(m))
    print("    najhoršie:", round(res[-1][1][0], 4))
    best_cfg = res[0][0]
    ls = lam_strength(best_cfg)
    out["strength_cfg"] = best_cfg

    # ---------------------------------------------------------------- fáza B: Elo
    elo_res = []
    for ks, hfa in itertools.product((0.7, 0.85, 1.0, 1.2, 1.4), (50, 70, 90, 110)):
        pre = E.run_elo(w, hfa=hfa, k_scale=ks)
        le = W.walk_elo(w, ALL, pre=pre)
        j_e, m_e = J(le[0], le[1], TUNE)
        best_b = min(((J(W.blend(ls[0], le[0], b), W.blend(ls[1], le[1], b), TUNE)[0], b) for b in (0.2, 0.3, 0.4, 0.5, 0.6)))
        elo_res.append((best_b[0], best_b[1], ks, hfa, j_e, m_e["logloss"]))
    elo_res.sort()
    print("[B] Elo – najlepšie (J mixu, váha Elo, k_scale, hfa, J Elo, logloss Elo):")
    for r in elo_res[:5]:
        print("   ", [round(x, 4) for x in r])
    _, b_w, ks, hfa, _, _ = elo_res[0]
    out["elo_cfg"] = dict(k_scale=ks, hfa=hfa)
    out["elo_weight"] = b_w
    pre = E.run_elo(w, hfa=hfa, k_scale=ks)
    le = W.walk_elo(w, ALL, pre=pre)
    lb = (W.blend(ls[0], le[0], b_w), W.blend(ls[1], le[1], b_w))

    # ---------------------------------------------------------------- fáza C: ρ a diagnostika kalibrácie
    rho_res = [(J(lb[0], lb[1], TUNE, rho)[0], rho) for rho in (-0.12, -0.09, -0.06, -0.03, 0.0, 0.03)]
    rho = min(rho_res)[1]
    print("[C] ρ:", [(r, round(j, 4)) for j, r in rho_res], "->", rho)
    out["rho"] = rho
    ks_res = []
    for k in (0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.46):
        sh, sa = E.stretch(lb[0], lb[1], k, 1.0)
        ks_res.append((k, round(J(sh, sa, TUNE, rho)[1]["logloss"], 4), round(J(sh, sa, TEST, rho)[1]["logloss"], 4)))
    print("[C] roztiahnutie k (logloss ladenie, test) – ideálne má byť optimum pri k=1:", ks_res)
    out["stretch_diag"] = ks_res

    # ---------------------------------------------------------------- záverečné porovnanie
    rows = {}
    for name, (lh, la) in dict(elo=le, sila=ls, prototyp=lb).items():
        rows[name] = dict(ladenie=short(J(lh, la, TUNE, rho)[1]), test=short(J(lh, la, TEST, rho)[1]))
    out["compare"] = rows
    print("[D] porovnanie (ρ=%.2f):" % rho)
    for name, r in rows.items():
        print(f"    {name:9s} ladenie {r['ladenie']}")
        print(f"    {'':9s} test    {r['test']}")
    np.savez(CACHE / "final_lambdas.npz", rows=ALL, lh=lb[0], la=lb[1], lh_s=ls[0], la_s=ls[1], lh_e=le[0], la_e=le[1])
    (RES / "goals_tuning.json").write_text(json.dumps(out, indent=1, default=float))
    print("hotovo", round(time.time() - t0), "s")
