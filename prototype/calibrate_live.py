"""Kalibrácia pre ostré predpovede zo všetkých dostupných out-of-sample predpovedí (data/processed/wide_walk.npz).

  uefa  – súťažné zápasy zóny UEFA + MS/ME (pre zápasy dvoch európskych tímov)
  world – všetky súťažné zápasy sveta (pre ostatné zápasy; mimo Európy model favorita nepodhodnocuje)
"""
import json

import numpy as np

import calib as C
import engine as E
from final_goals import calpop

if __name__ == "__main__":
    w = E.load_world()
    cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
    rho = cfg["rho"]
    z = np.load(E.PROC / "wide_walk.npz")
    rows = z["rows"]
    hg, ag = w.hg.values[rows], w.ag.values[rows]
    hf = (~w.neutral.values[rows]).astype(float)
    out = dict(rho=rho, features=["1", "log λ_sila", "log λ_sila súper", "log λ_elo", "log λ_elo súper", "doma"])
    for name, pop in (("uefa", calpop(w)[rows]), ("world", (w.cls.values[rows] >= 2))):
        beta, theta = C.fit_calibration(z["lh"][pop], z["la"][pop], z["eh"][pop], z["ea"][pop], hf[pop], hg[pop], ag[pop], rho)
        out[name] = dict(beta=list(map(float, beta)), theta=list(map(float, theta)), n=int(pop.sum()))
        print(name, int(pop.sum()), "beta", beta.round(3), "theta", theta.round(3))
    out["beta"], out["theta"], out["n"] = out["uefa"]["beta"], out["uefa"]["theta"], out["uefa"]["n"]
    t = w.tournament.isin(["UEFA Euro qualification", "UEFA Nations League"])
    out["uefa_teams"] = sorted(set(w.home[t]) | set(w.away[t]))
    (E.ROOT / "results" / "calibration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
