"""Zaostrenie podielov strelcov: výber gamma na ladení (2022-07 … 2023-12), overenie na teste (2024 – 2026)."""
import json
import sys
from multiprocessing import Pool

import numpy as np

import engine as E
import scorers as S
import scorers_eval as SE

BEST = json.loads((E.ROOT / "results" / "scorers_backtest.json").read_text())["best"]
BEST.pop("gamma", None)


def ll(p, y):
    return -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))


def run(g):
    _, df, _ = S.backtest(SE.w, SE.LAM, SE.LU, {**BEST, "gamma": g}, eval_from="2022-07-01", collect=True)
    return g, df


if __name__ == "__main__":
    grid = [1.0, 1.1, 1.2, 1.3, 1.45, 1.6]
    with Pool(3) as p:
        res = dict(p.map(run, grid))
    base = res[1.0]
    out = {}
    for g in grid:
        df = res[g]
        row = {}
        for per, lo, hi in (("ladenie", "2022-07-01", "2024-01-01"), ("test", "2024-01-01", "2027-01-01")):
            m = ((df.date >= lo) & (df.date < hi)).values
            d = df[m]
            top = d.loc[d.groupby(["id", "side"]).p.idxmax()]
            dd = (ll(df.p.values, df.y.values) - ll(base.p.values, base.y.values))[m]
            ids = df.id.values[m]
            uid, inv = np.unique(ids, return_inverse=True)
            sums, cnt = np.bincount(inv, dd), np.bincount(inv)
            rng = np.random.default_rng(0)
            bs = [sums[i].sum() / cnt[i].sum() for i in rng.integers(0, len(uid), (2000, len(uid)))]
            hi_bin = d[d.p >= 0.3]
            row[per] = dict(n=int(m.sum()), ll=float(ll(d.p.values, d.y.values).mean()), diff=float(dd.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                            top1_hit=float(top.y.mean()), top1_p=float(top.p.mean()), p30_pred=float(hi_bin.p.mean()) if len(hi_bin) else None,
                            p30_real=float(hi_bin.y.mean()) if len(hi_bin) else None)
        out[g] = row
        t, s_ = row["ladenie"], row["test"]
        print(f"gamma {g:.2f} | ladenie LL {t['ll']:.5f} ({t['diff']:+.5f} [{t['ci'][0]:+.5f},{t['ci'][1]:+.5f}]) top1 {t['top1_hit']:.3f}/{t['top1_p']:.3f} | "
              f"test LL {s_['ll']:.5f} ({s_['diff']:+.5f} [{s_['ci'][0]:+.5f},{s_['ci'][1]:+.5f}]) top1 {s_['top1_hit']:.3f}/{s_['top1_p']:.3f} p≥30%: {s_['p30_pred']:.3f}→{s_['p30_real']:.3f}", flush=True)
    (E.ROOT / "results" / "scorers_gamma.json").write_text(json.dumps(out, indent=1, default=float))
