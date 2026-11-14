"""Backtest modelu strelcov: log-loss „hráč základnej jedenástky skóruje“ oproti základu (len post)."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import engine as E
import scorers as S
import walk as W

w = E.load_world()
_cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
_z = np.load(E.ROOT / "data" / "raw" / "cache" / "espn_walk.npz")
_rows = _z["rows"]
_lh, _la = W.blend(_z["lh"], _z["eh"], _cfg["elo_weight"]), W.blend(_z["la"], _z["ea"], _cfg["elo_weight"])
LAM = {eid: (lh, la, d, c) for eid, lh, la, d, c in zip(w.espn_id.values[_rows], _lh, _la, w.date.values[_rows], w.comp.values[_rows])}
LU = S.load_lineups()


def run(params):
    return params, S.backtest(w, LAM, LU, params, eval_from="2023-01-01"), S.backtest(w, LAM, LU, params, eval_from="2025-01-01")


if __name__ == "__main__":
    grid = [dict(kappa=k, alpha=a, kappa_sot=ks) for k in (6, 12, 25, 50) for a in (1.0, 0.6, 0.3) for ks in ((4,) if a == 1.0 else (8, 20))]
    with Pool(4) as p:
        res = p.map(run, grid)
    res.sort(key=lambda r: r[1]["ll_model"])
    for params, r, r2 in res:
        print(params, f"2023+: n={r['n']} LL {r['ll_model']:.4f} (základ {r['ll_base']:.4f}) top1 {r['top1_hit_model']:.3f} (p {r['top1_p_model']:.3f}, základ {r['top1_hit_base']:.3f}) | 2025+: LL {r2['ll_model']:.4f} (základ {r2['ll_base']:.4f})")
    best = res[0][0]
    (E.ROOT / "results" / "scorers_backtest.json").write_text(json.dumps(dict(best=best, eval_2023=res[0][1], eval_2025=res[0][2],
                                                                           grid=[(a, b, c) for a, b, c in res]), indent=1, default=float))
