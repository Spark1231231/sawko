"""Walk-forward λ pre všetky zápasy s ESPN údajmi (podklad pre modely striel, kariet a strelcov)."""
import json
import os
import time

import numpy as np

import engine as E
import walk as W

if __name__ == "__main__":
    w = E.load_world()
    cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
    scfg = cfg["strength_cfg"]
    scfg["imp"] = tuple(scfg["imp"])
    rows = np.flatnonzero((w.comp.notna() & (w.date >= "2016-07-01") & w.hg.notna()).values)
    t = time.time()
    lh, la = W.walk_strength(w, rows, scfg)
    pre = E.run_elo(w, **cfg["elo_cfg"])
    eh, ea = W.walk_elo(w, rows, pre=pre)
    tmp = str(E.ROOT / "data" / "raw" / "cache" / ("tmp_%s.npz" % os.getpid()))
    np.savez(tmp, rows=rows, lh=lh, la=la, eh=eh, ea=ea)
    os.replace(tmp, E.ROOT / "data" / "raw" / "cache" / "espn_walk.npz")
    print("espn walk", len(rows), "zápasov", round(time.time() - t), "s")
