"""Ladenie zmrštenia (sigma) a polčasu útlmu pre KAŽDÝ druh počtu zvlášť (doteraz mali strely na bránu, rohy
a fauly prebraté nastavenie striel a mriežka končila na okraji). Výber podľa obdobia ladenie, overenie na teste."""
from __future__ import annotations

import json
from multiprocessing import Pool

import counts as K
import counts_eval as CE
import engine as E

GRID = [dict(sigma=s, halflife=h) for s in (0.05, 0.1, 0.18, 0.3) for h in (350, 600, 1100, 2200)]


def run(args):
    kind, cfg = args
    r = K.walk_counts(CE.D, kind, cfg)
    return kind, cfg, K.evaluate(r, CE.TUNE, CE.TUNE, CE.LINES[kind]), K.evaluate(r, CE.TUNE, CE.TEST, CE.LINES[kind])


if __name__ == "__main__":
    cur = json.loads((E.ROOT / "results" / "counts_cfg.json").read_text())
    with Pool(4) as p:
        res = p.map(run, [(k, c) for k in K.KINDS for c in GRID])
    score = lambda e: e["team_nll_model"] + e["total_nll_model"]
    new, report = {}, {}
    for kind in K.KINDS:
        rows = [(c, et, es) for k, c, et, es in res if k == kind]
        best = min(rows, key=lambda x: score(x[1]))
        old = next(x for x in rows if x[0] == {**K.DEFAULT, **cur[kind]} or (x[0]["sigma"] == cur[kind]["sigma"] and x[0]["halflife"] == cur[kind]["halflife"]))
        new[kind] = best[0]
        report[kind] = dict(old=cur[kind], new=best[0],
                            ladenie=dict(old=score(old[1]), new=score(best[1])), test=dict(old=score(old[2]), new=score(best[2])),
                            test_team=dict(old=old[2]["team_nll_model"], new=best[2]["team_nll_model"]),
                            test_total=dict(old=old[2]["total_nll_model"], new=best[2]["total_nll_model"]),
                            test_mae=dict(old=old[2]["mae_model"], new=best[2]["mae_model"]))
        print(f"{kind:8s} {cur[kind]} -> {best[0]} | ladenie (tím+súčet NLL) {score(old[1]):.4f} -> {score(best[1]):.4f} | "
              f"test {score(old[2]):.4f} -> {score(best[2]):.4f} (tím {old[2]['team_nll_model']:.4f} -> {best[2]['team_nll_model']:.4f}, "
              f"súčet {old[2]['total_nll_model']:.4f} -> {best[2]['total_nll_model']:.4f}, MAE {old[2]['mae_model']:.3f} -> {best[2]['mae_model']:.3f})", flush=True)
    (E.ROOT / "results" / "counts_tune.json").write_text(json.dumps(dict(report=report, proposed=new), indent=1, default=float))
