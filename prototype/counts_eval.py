"""Walk-forward backtest modelov počtov (strely, strely na bránu, rohy, žlté karty, fauly).

Porovnáva tri úrovne:
  základ   – konštantný priemer z minulosti (nič nevie o tímoch),
  útok/obr – iba tímové efekty (takto počty počíta web),
  prototyp – tímové efekty + regresia na papierovú silu a typ súťaže.

Spustenie: python3 counts_eval.py [sweep]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import counts as K
import engine as E
import walk as W

LINES = {"shots": (22.5, 24.5, 26.5), "sot": (7.5, 8.5, 9.5), "corners": (8.5, 9.5, 10.5),
         "yellow": (2.5, 3.5, 4.5), "fouls": (22.5, 24.5, 26.5)}

w = E.load_world()
_cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
_z = np.load(E.ROOT / "data" / "raw" / "cache" / "espn_walk.npz")
D = K.frame(w, _z["rows"], W.blend(_z["lh"], _z["eh"], _cfg["elo_weight"]), W.blend(_z["la"], _z["ea"], _cfg["elo_weight"]))
EV = D.comp.isin(E.EVAL_COMPS).values
TUNE = EV & (D.date.values < "2024-01-01")
TEST = EV & (D.date.values >= "2024-01-01")


def referee_factor(res, kappa: float = 40.0):
    """Efekt rozhodcu na karty, walk-forward: pomer (skutočné + κ)/(predpovedané + κ) z jeho MINULÝCH zápasov."""
    ref = D.referee.values
    order = res["target"][np.argsort(D.day.values[res["target"]], kind="stable")]
    act, exp = {}, {}
    f = np.ones(len(D))
    for i in order:
        r = ref[i]
        if isinstance(r, str):
            f[i] = (act.get(r, 0.0) + kappa) / (exp.get(r, 0.0) + kappa)
            act[r] = act.get(r, 0.0) + res["yh"][i] + res["ya"][i]
            exp[r] = exp.get(r, 0.0) + res["mh"][i] + res["ma"][i]
    return f


def run(args):
    kind, cfg, feats = args
    res = K.walk_counts(D, kind, cfg, use_features=feats)
    out = dict(ladenie=K.evaluate(res, TUNE, TUNE, LINES[kind]), test=K.evaluate(res, TUNE, TEST, LINES[kind]))
    if kind == "yellow" and feats:
        known = D.referee.notna().values
        for kappa in (20.0, 40.0, 80.0):
            f = referee_factor(res, kappa)
            out[f"ref_k{int(kappa)}"] = dict(
                bez=K.evaluate(res, TUNE, EV & known & (D.date.values >= "2023-01-01"), LINES[kind]),
                s_rozhodcom=K.evaluate(res, TUNE, EV & known & (D.date.values >= "2023-01-01"), LINES[kind], factor=f))
    return kind, feats, out


def fmt(e):
    ln = " ".join(f"{l}:{v['model']:.4f}/{v['const']:.4f}" for l, v in e["lines"].items())
    return (f"n={e['n']} priemer={e['mean']:.2f} | tím NLL {e['team_nll_model']:.4f}/{e['team_nll_const']:.4f} | súčet NLL "
            f"{e['total_nll_model']:.4f}/{e['total_nll_const']:.4f} MAE {e['mae_model']:.3f}/{e['mae_const']:.3f} "
            f"kor {e['corr']:.3f} r_súčet={e['r_total']:.1f} | O/U {ln}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        jobs = [(k, dict(sigma=s_, halflife=hl), True) for k in ("shots", "yellow") for s_ in (0.06, 0.1, 0.18) for hl in (350, 600, 1100)]
        with Pool(4) as p:
            for kind, feats, o in p.map(run, jobs):
                print(kind, f"ladenie tím NLL {o['ladenie']['team_nll_model']:.4f} súčet {o['ladenie']['total_nll_model']:.4f}", flush=True)
        sys.exit()
    best = json.loads((E.ROOT / "results" / "counts_cfg.json").read_text())
    jobs = [(k, best.get(k, {}), feats) for k in K.KINDS for feats in (True, False)]
    out = {}
    print("formát: model/konštanta (nižšie = lepšie); konštanta = najlepší možný konštantný tip na danom období")
    with Pool(4) as p:
        for kind, feats, o in p.map(run, jobs):
            name = "prototyp" if feats else "útok/obrana"
            out.setdefault(kind, {})[name] = o
            print(f"{kind:8s} {name:12s} ladenie: {fmt(o['ladenie'])}")
            print(f"{'':8s} {'':12s} test:    {fmt(o['test'])}", flush=True)
            for k_, v in o.items():
                if k_.startswith("ref_"):
                    print(f"{'':8s} rozhodca {k_}: bez {v['bez']['total_nll_model']:.4f} / s {v['s_rozhodcom']['total_nll_model']:.4f} "
                          f"(n={v['bez']['n']}); O/U 3.5: {v['bez']['lines']['3.5']['model']:.4f} / {v['s_rozhodcom']['lines']['3.5']['model']:.4f}")
    (E.ROOT / "results" / "counts_backtest.json").write_text(json.dumps(out, indent=1, default=float))
