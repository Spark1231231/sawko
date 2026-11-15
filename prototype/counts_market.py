"""Audit ostrej predpovede počtov: čo sa stane, keď sa do modelu (naučeného na rozdiele síl z MODELU)
pri predpovedi vloží rozdiel síl z TRHU?  Varianty (vyhodnotené len na zápasoch so záverečnými kurzmi):

  model      – s z modelu (to, čo bolo doteraz backtestované)
  trh surovo – s z finálnej zmesi 85 % trh (to, čo doteraz robila ostrá predpoveď)
  trh v mierke modelu – s z trhu prepočítané na mierku modelu: s' = a + b·s_trh, kde (a, b) je regresia
               s_model ~ s_trh z minulých zápasov (rolling po rokoch)
"""
from __future__ import annotations

import json
import sys

import numpy as np

import counts as K
import engine as E
import experiment as X
import walk as W

LINES = {"shots": (24.5,), "sot": (8.5,), "corners": (9.5,), "yellow": (3.5,), "fouls": (24.5,)}

if __name__ == "__main__":
    w = X.w
    z = np.load(E.PROC / "espn_walk.npz")
    bw = X.CFG["elo_weight"]
    D = K.frame(w, z["rows"], W.blend(z["lh"], z["eh"], bw), W.blend(z["la"], z["ea"], bw))
    has = np.isfinite(D.mlh.values) & np.isfinite(D.mla.values)
    s_mod = D.s.values
    s_mkt = np.where(has, np.log(np.where(has, D.mlh.values, 1.0) / np.where(has, D.mla.values, 1.0)), np.nan)
    # finálna zmes: 85 % trh + 15 % model (v log λ => aj v s)
    s_blend = np.where(has, 0.85 * s_mkt + 0.15 * s_mod, s_mod)
    # mierka: s_model ~ a + b * s_blend, rolling po rokoch
    s_scaled = s_mod.copy()
    year = D.year.values
    coefs = {}
    for y in range(2022, 2027):
        tr = has & (year < y)
        if tr.sum() < 150:
            continue
        b, a = np.polyfit(s_blend[tr], s_mod[tr], 1)
        coefs[y] = (float(a), float(b))
        sel = has & (year == y)
        s_scaled[sel] = a + b * s_blend[sel]
    print("mierka s_model = a + b·s_zmes po rokoch:", {y: (round(a, 3), round(b, 3)) for y, (a, b) in coefs.items()})
    ev = D.comp.isin(E.EVAL_COMPS).values & has
    tune, test = ev & (D.date.values >= "2022-01-01") & (D.date.values < "2024-01-01"), ev & (D.date.values >= "2024-01-01")
    fit_mask = D.comp.isin(E.EVAL_COMPS).values & (D.date.values < "2024-01-01")
    cfgs = json.loads((E.ROOT / "results" / "counts_cfg.json").read_text())
    out = {}
    for kind in ("shots", "sot", "corners", "yellow", "fouls"):
        out[kind] = {}
        for name, sp in (("model", None), ("trh surovo", s_blend), ("trh v mierke modelu", s_scaled)):
            res = K.walk_counts(D, kind, cfgs.get(kind), s_pred=sp)
            et, es = K.evaluate(res, fit_mask, tune, LINES[kind]), K.evaluate(res, fit_mask, test, LINES[kind])
            out[kind][name] = dict(ladenie=et, test=es)
            l = str(LINES[kind][0])
            print(f"{kind:8s} {name:20s} ladenie n={et['n']:4d} tím NLL {et['team_nll_model']:.4f} súčet {et['total_nll_model']:.4f} O/U {et['lines'][l]['model']:.4f} | "
                  f"test n={es['n']:4d} tím NLL {es['team_nll_model']:.4f} súčet {es['total_nll_model']:.4f} O/U {es['lines'][l]['model']:.4f}", flush=True)
    (E.ROOT / "results" / "counts_market.json").write_text(json.dumps(dict(scale=coefs, results=out), indent=1, default=float))
