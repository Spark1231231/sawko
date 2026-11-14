"""Predpovede pre všetky zápasy daného dňa zo snapshotu webu + (po zápasoch) porovnanie so skutočnosťou.

    python3 today.py 2026-11-14            # predpovede
    python3 today.py 2026-11-14 check      # doplní skutočné výsledky z ESPN
"""
from __future__ import annotations

import json
import sys

import numpy as np
import requests

import engine as E
import predict as PR
from report import SK, num, pct

RES = E.ROOT / "results"


def actual(eid: str, comp: str = "uefa.nations"):
    d = requests.get(f"{PR.ESPN}/{comp}/summary?event={eid}", timeout=60).json()
    comp_ = d["header"]["competitions"][0]
    st = comp_["status"]["type"]
    out = dict(state=st["state"], detail=st.get("shortDetail"))
    sides = {c["homeAway"]: c for c in comp_["competitors"]}
    out["hg"], out["ag"] = int(sides["home"].get("score") or 0), int(sides["away"].get("score") or 0)
    for t in (d.get("boxscore") or {}).get("teams", []):
        side = "h" if t["team"]["id"] == sides["home"]["team"]["id"] else "a"
        stt = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
        for k, name in (("sh", "totalShots"), ("sot", "shotsOnTarget"), ("y", "yellowCards"), ("r", "redCards"), ("cor", "wonCorners")):
            try:
                out[side + k] = float(stt.get(name))
            except (TypeError, ValueError):
                out[side + k] = None
    out["scorers"] = [(e.get("team", {}).get("displayName"), (e.get("participants") or [{}])[0].get("athlete", {}).get("displayName"), e.get("clock", {}).get("displayValue"))
                      for e in d.get("keyEvents", []) if e.get("scoringPlay")]
    return out


if __name__ == "__main__":
    day = sys.argv[1]
    f = RES / f"today_{day}.json"
    if len(sys.argv) > 2 and sys.argv[2] == "check":
        rows = json.loads(f.read_text())
        for r in rows:
            try:
                r["actual"] = actual(r["id"])
            except Exception as ex:  # noqa: BLE001
                r["actual"] = dict(state="?", error=str(ex))
        f.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float))
    else:
        snap = json.loads((E.PROC / "site_snapshot.json").read_text())
        pr = PR.Predictor()
        rows = []
        for eid, a in sorted(snap.items(), key=lambda kv: kv[1]["fixture"]["date"]):
            if not a["fixture"]["date"].startswith(day):
                continue
            r = PR.predict_match(pr, a)
            (RES / f"prediction_{eid}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float))
            rows.append(dict(id=eid, date=r["date"], home=r["home"], away=r["away"], model=r["model"], market=r.get("market"), final=r["final"],
                             yellow=r["counts"]["yellow"], shots=r["counts"]["shots"], sot=r["counts"]["sot"], corners=r["counts"]["corners"],
                             top={s: [(p["name"], p["p_score"]) for p in r["scorers"][s]["players"][:2]] for s in r["scorers"]}))
        f.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=float))
    L = [f"# Zápasy {day} – finálne predpovede prototypu (85 % trh + 15 % model)", "",
         "| výkop (UTC) | zápas | 1 / X / 2 | tip | skóre | oč. góly | žlté | strely | najpravdep. strelci | skutočnosť |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fz = r["final"]
        h, a = SK.get(r["home"], r["home"]), SK.get(r["away"], r["away"])
        tip = max((fz["p1"], "1"), (fz["px"], "X"), (fz["p2"], "2"))[1]
        top = "; ".join(f"{n} {pct(p)}" for s in ("h", "a") for n, p in r["top"].get(s, [])[:1])
        act = r.get("actual") or {}
        real = "–"
        if act.get("state") in ("post", "in"):
            real = f"{act['hg']}:{act['ag']}" + (" (hrá sa)" if act["state"] == "in" else "")
            if act.get("hy") is not None:
                real += f", žlté {int(act['hy'] + act['ay'])}, strely {int(act['hsh'] + act['ash'])}"
        L.append(f"| {r['date'][11:16]} | {h} – {a} | {pct(fz['p1'])} / {pct(fz['px'])} / {pct(fz['p2'])} | {tip} | {fz['scores'][0][0]}:{fz['scores'][0][1]} | "
                 f"{num(fz['lam'][0], 2)} : {num(fz['lam'][1], 2)} | {num(r['yellow']['total'])} | {num(r['shots']['total'])} | {top} | {real} |")
    (RES / f"today_{day}.md").write_text("\n".join(L))
    print("\n".join(L))
