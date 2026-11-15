"""Audit ostrej predpovede strelcov: λ tímu z modelu vs zo zmesi 85 % trh + 15 % model (len zápasy s kurzami)."""
import json

import numpy as np

import engine as E
import experiment as X
import scorers as S
import walk as W

if __name__ == "__main__":
    w = X.w
    z = np.load(E.PROC / "espn_walk.npz")
    rows = z["rows"]
    lh, la = W.blend(z["lh"], z["eh"], X.CFG["elo_weight"]), W.blend(z["la"], z["ea"], X.CFG["elo_weight"])
    eid = w.espn_id.values[rows]
    lam = {e: (a, b, d, c) for e, a, b, d, c in zip(eid, lh, la, w.date.values[rows], w.comp.values[rows])}
    mh, ma = w.mlh.values[rows], w.mla.values[rows]
    has = np.isfinite(mh) & np.isfinite(ma)
    pred = {e: (float(np.exp(0.85 * np.log(mh[i]) + 0.15 * np.log(lh[i]))), float(np.exp(0.85 * np.log(ma[i]) + 0.15 * np.log(la[i]))))
            for i, e in enumerate(eid) if has[i]}
    lu = S.load_lineups()
    sp = json.loads((E.ROOT / "results" / "scorers_backtest.json").read_text())["best"]
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
    out = {}
    dfs = {}
    for name, lp in (("model", None), ("zmes s trhom", pred)):
        _, df, _ = S.backtest(w, lam, lu, sp, eval_from="2022-01-01", collect=True, lam_pred=lp)
        dfs[name] = df[df.id.isin(pred)].reset_index(drop=True)
    for per, lo, hi in (("ladenie 2022-23", "2022-01-01", "2024-01-01"), ("test 2024-26", "2024-01-01", "2027-01-01")):
        row = {}
        for name, df in dfs.items():
            d = df[(df.date >= lo) & (df.date < hi)]
            top = d.loc[d.groupby(["id", "side"]).p.idxmax()]
            row[name] = dict(n=len(d), ll=ll(d.p.values, d.y.values), ll_post=ll(d.pb.values, d.y.values), top1_hit=float(top.y.mean()), top1_p=float(top.p.mean()))
        a, b = dfs["model"], dfs["zmes s trhom"]
        m = (a.date >= lo) & (a.date < hi)
        la_ = -(a.y * np.log(a.p.clip(1e-9, 1)) + (1 - a.y) * np.log((1 - a.p).clip(1e-9, 1)))
        lb_ = -(b.y * np.log(b.p.clip(1e-9, 1)) + (1 - b.y) * np.log((1 - b.p).clip(1e-9, 1)))
        dd = (lb_ - la_)[m].values
        # bootstrap po zápasoch (hráči jedného zápasu sú závislí)
        ids = a.id[m].values
        uid, inv = np.unique(ids, return_inverse=True)
        sums, cnt = np.bincount(inv, dd), np.bincount(inv)
        rng = np.random.default_rng(0)
        bs = [sums[i].sum() / cnt[i].sum() for i in rng.integers(0, len(uid), (3000, len(uid)))]
        row["rozdiel"] = [float(dd.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        out[per] = row
        print(per, {k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else [round(x, 5) for x in v]) for k, v in row.items()})
    (E.ROOT / "results" / "scorers_market.json").write_text(json.dumps(out, indent=1, default=float))
