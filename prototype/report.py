"""Z results/prediction_<id>.json urobí čitateľnú správu (Markdown, po slovensky).

    python3 report.py 401861167
"""
from __future__ import annotations

import json
import sys

import engine as E

SK = {"Portugal": "Portugalsko", "Denmark": "Dánsko", "Norway": "Nórsko", "Wales": "Wales", "Austria": "Rakúsko",
      "Republic of Ireland": "Írsko", "Sweden": "Švédsko", "Bosnia and Herzegovina": "Bosna a Hercegovina", "Romania": "Rumunsko",
      "Poland": "Poľsko", "Northern Ireland": "Severné Írsko", "Ukraine": "Ukrajina", "Georgia": "Gruzínsko", "Hungary": "Maďarsko",
      "Kosovo": "Kosovo", "Israel": "Izrael"}


def pct(x, d=0):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def report(r: dict) -> str:
    h, a = SK.get(r["home"], r["home"]), SK.get(r["away"], r["away"])
    f, m = r["final"], r["model"]
    mk = r.get("market")
    L = []
    L.append(f"# {h} – {a}")
    L.append(f"{r.get('comp', '')} · výkop {r['date'].replace('T', ' ').replace('Z', ' UTC')} · {r.get('venue') or ''}")
    L.append("")
    fav = max((f["p1"], h), (f["px"], "remíza"), (f["p2"], a))
    L.append(f"**Tip na víťaza: {fav[1]} ({pct(fav[0])})** · najpravdepodobnejšie skóre **{f['scores'][0][0]}:{f['scores'][0][1]}**")
    L.append("")
    L.append("## Výsledok (1 / X / 2)")
    L.append("")
    L.append("| zdroj | 1 | X | 2 | oč. góly |")
    L.append("|---|---|---|---|---|")
    L.append(f"| prototyp – len model | {pct(m['p1'])} | {pct(m['px'])} | {pct(m['p2'])} | {num(m['lam'][0], 2)} : {num(m['lam'][1], 2)} |")
    if mk:
        L.append(f"| trh (kurzy bez marže) | {pct(mk['p1'])} | {pct(mk['px'])} | {pct(mk['p2'])} | {num(mk['lam'][0], 2)} : {num(mk['lam'][1], 2)} |")
    if r.get("site"):
        s = [x for x in r["site"]["factors"] if x["key"] != "market"]
        if s:
            L.append(f"| web – len model | {pct(s[-1]['p1'])} | {pct(s[-1]['px'])} | {pct(s[-1]['p2'])} | {num(r['site']['lam_model'][0], 2)} : {num(r['site']['lam_model'][1], 2)} |")
    L.append(f"| **finálna predpoveď** ({pct(f['market_weight'])} trh) | **{pct(f['p1'])}** | **{pct(f['px'])}** | **{pct(f['p2'])}** | **{num(f['lam'][0], 2)} : {num(f['lam'][1], 2)}** |")
    L.append("")
    L.append("## Skóre a góly")
    L.append("")
    L.append("| skóre | " + " | ".join(f"{i}:{j}" for i, j, _ in f["scores"]) + " |")
    L.append("|---|" + "---|" * len(f["scores"]))
    L.append("| šanca | " + " | ".join(pct(p, 1) for _, _, p in f["scores"]) + " |")
    L.append("")
    L.append(f"Nad 1,5 gólu {pct(f['over15'])} · nad 2,5 {pct(f['over25'])} · nad 3,5 {pct(f['over35'])} · obaja skórujú {pct(f['btts'])}")
    L.append("")
    c = r["counts"]
    y = c["yellow"]
    L.append("## Karty")
    L.append("")
    L.append(f"Žlté karty: **{h} {num(y['home'])} · {a} {num(y['away'])} · spolu {num(y['total'])}** "
             f"(najčastejšie {y['mode']}, 80 % interval {y['q10']}–{y['q90']}).")
    L.append("Nad 2,5: {} · nad 3,5: {} · nad 4,5: {} · nad 5,5: {}.".format(*(pct(y["over"][k]) for k in ("2.5", "3.5", "4.5", "5.5"))))
    L.append(f"Červená karta v zápase: {pct(c['red']['p_any'])}.")
    if r.get("referee", {}).get("name"):
        ref = y.get("ref") or {}
        L.append(f"Rozhodca: {r['referee']['name']}" + (f" – faktor kariet ×{num(ref['factor'], 2)} (z {ref['n']} zápasov, silno zmrštené k 1)." if ref else "."))
    L.append("")
    L.append("## Strely")
    L.append("")
    L.append("| | " + h + " | " + a + " | spolu | 80 % interval |")
    L.append("|---|---|---|---|---|")
    for key, name in (("shots", "strely"), ("sot", "strely na bránu"), ("corners", "rohy"), ("fouls", "fauly")):
        x = c[key]
        L.append(f"| {name} | {num(x['home'])} | {num(x['away'])} | **{num(x['total'])}** | {x['q10']}–{x['q90']} |")
    L.append("")
    sh = c["shots"]
    L.append("Strely spolu: " + " · ".join(f"nad {k.replace('.', ',')}: {pct(v)}" for k, v in sh["over"].items()))
    L.append("")
    L.append("## Strelci")
    L.append("")
    conf = "potvrdené zostavy" if r.get("lineup_confirmed") else "PREDPOKLADANÉ zostavy (potvrdené bývajú hodinu pred výkopom)"
    L.append(f"Šanca, že hráč dá aspoň jeden gól ({conf}).")
    L.append("")
    for side in ("h", "a"):
        s = r["scorers"].get(side)
        if not s:
            continue
        L.append(f"**{SK.get(s['team'], s['team'])}** (očakávané góly tímu {num(s['lam'], 2)})")
        L.append("")
        L.append("| hráč | gól | 2+ góly | kurz Fortuny | repre | klub | trh |")
        L.append("|---|---|---|---|---|---|---|")
        for p in s["players"][:7]:
            parts = p["parts"]
            L.append(f"| {p['name']} | **{pct(p['p_score'])}** | {pct(p['p_2plus'], 1)} | {p['odds'] or '–'} | "
                     f"{pct(parts['nat']) if 'nat' in parts else '–'} | {pct(parts['club']) if 'club' in parts else '–'} | {pct(parts['market']) if 'market' in parts else '–'} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    eid = sys.argv[1]
    r = json.loads((E.ROOT / "results" / f"prediction_{eid}.json").read_text())
    md = report(r)
    (E.ROOT / "results" / f"prediction_{eid}.md").write_text(md)
    print(md)
