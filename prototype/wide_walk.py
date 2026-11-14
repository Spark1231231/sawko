"""Walk-forward predpovede pre VŠETKY súťažné zápasy sveta od 2008 (podklad pre kalibráciu)."""
import json
import os
import sys
import time

import numpy as np

import engine as E
import walk as W

if __name__ == "__main__":
    w = E.load_world()
    cfg = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
    scfg = cfg["strength_cfg"]
    scfg["imp"] = tuple(scfg["imp"])
    if len(sys.argv) > 1:
        scfg["halflife"] = float(sys.argv[1])
    rows = np.flatnonzero(((w.cls >= 1) & (w.date >= "2008-01-01") & w.hg.notna()).values)
    t = time.time()
    lh, la = W.walk_strength(w, rows, scfg)
    pre = E.run_elo(w, **cfg["elo_cfg"])
    eh, ea = W.walk_elo(w, rows, pre=pre)
    tag = "" if len(sys.argv) <= 1 else "_hl" + sys.argv[1]
    tmp = str(E.ROOT / "data" / "raw" / "cache" / ("tmp_%s.npz" % os.getpid()))
    np.savez(tmp, rows=rows, lh=lh, la=la, eh=eh, ea=ea)
    os.replace(tmp, E.PROC / "wide_walk.npz" if not tag else E.ROOT / "data" / "raw" / "cache" / f"wide{tag}.npz")
    print("wide walk", len(rows), "zápasov", round(time.time() - t), "s")
