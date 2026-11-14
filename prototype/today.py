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
    done = [r for r in rows if (r.get("actual") or {}).get("state") == "post"]
    if done:
        tips = sum((r["actual"]["hg"] > r["actual"]["ag"], r["actual"]["hg"] == r["actual"]["ag"], r["actual"]["hg"] < r["actual"]["ag"])
                   [int(np.argmax([r["final"]["p1"], r["final"]["px"], r["final"]["p2"]]))] for r in done)
        def pr_out(r, key):
            a_ = r["actual"]
            o = 0 if a_["hg"] > a_["ag"] else 1 if a_["hg"] == a_["ag"] else 2
            return [r[key]["p1"], r[key]["px"], r[key]["p2"]][o]
        ll_f = -np.mean([np.log(pr_out(r, "final")) for r in done])
        ll_m = -np.mean([np.log(pr_out(r, "model")) for r in done])
        exact = sum((r["final"]["scores"][0][0], r["final"]["scores"][0][1]) == (r["actual"]["hg"], r["actual"]["ag"]) for r in done)
        top3 = sum(any((i, j) == (r["actual"]["hg"], r["actual"]["ag"]) for i, j, _ in r["final"]["scores"][:3]) for r in done)
        cnt = [r for r in done if r["actual"].get("hy") is not None and r["actual"].get("hsh") is not None]
        mae_y = np.mean([abs(r["yellow"]["total"] - (r["actual"]["hy"] + r["actual"]["ay"])) for r in cnt]) if cnt else float("nan")
        mae_s = np.mean([abs(r["shots"]["total"] - (r["actual"]["hsh"] + r["actual"]["ash"])) for r in cnt]) if cnt else float("nan")
        in_y = sum(r["yellow"]["q10"] <= r["actual"]["hy"] + r["actual"]["ay"] <= r["yellow"]["q90"] for r in cnt)
        in_s = sum(r["shots"]["q10"] <= r["actual"]["hsh"] + r["actual"]["ash"] <= r["shots"]["q90"] for r in cnt)
        goals = sum(r["actual"]["hg"] + r["actual"]["ag"] for r in done)
        exp_goals = sum(r["final"]["lam"][0] + r["final"]["lam"][1] for r in done)
        L += ["", f"**Vyhodnotenie ({len(done)} dohraných zápasov – malá vzorka, len orientačne):** tip 1/X/2 trafený {tips}/{len(done)}; "
              f"log-loss finálnej predpovede {num(ll_f, 3)} (len model {num(ll_m, 3)}); presné skóre {exact}/{len(done)}, v troch najpravdepodobnejších {top3}/{len(done)}; "
              f"góly spolu {goals} (očakávané {num(exp_goals, 1)}); žlté karty: priem. chyba {num(mae_y, 2)}, v 80 % intervale {in_y}/{len(cnt)}; "
              f"strely: priem. chyba {num(mae_s, 1)}, v 80 % intervale {in_s}/{len(cnt)}."]
        sc_lines = []
        for r in done:
            names = {n for _, n, _ in r["actual"].get("scorers", []) if n}
            picks = [(n, p) for s_ in ("h", "a") for n, p in r["top"].get(s_, [])[:1]]
            sc_lines.append(f"{SK.get(r['home'], r['home'])} – {SK.get(r['away'], r['away'])}: strelci {', '.join(sorted(names)) or '–'}; "
                            + "; ".join(f"{n} ({pct(p)}) {'✓' if any(PR.norm_name(n) == PR.norm_name(x) for x in names) else '✗'}" for n, p in picks))
        L += ["", "Najpravdepodobnejší strelci vs skutočnosť:", ""] + [f"- {x}" for x in sc_lines]
    (RES / f"today_{day}.md").write_text("\n".join(L))
    print("\n".join(L))
