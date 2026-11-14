"""Stiahne z webu (stavkova-pravdepodobnost.vercel.app) to, čo model stránky práve predpovedá pre
reprezentačné zápasy: λ modelu, λ trhu, faktory, zostavy, hráčov, kurzy. Uloží data/processed/site_snapshot.json.

Slúži na (a) porovnanie prototypu s modelom stránky na tých istých zápasoch, (b) vstupy pre dnešnú predpoveď.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import engine as E

SITE = "https://stavkova-pravdepodobnost.vercel.app"
s = requests.Session()
s.headers["User-Agent"] = "match-prediction-prototype/1.0"


def match_view(eid: str):
    r = s.get(f"{SITE}/zapas/{eid}", timeout=90)
    if r.status_code != 200:
        return None
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', r.text)
    flight = "".join(json.loads('"' + c + '"') for c in chunks)
    found = []

    def walk(x):
        if isinstance(x, list):
            if len(x) >= 4 and x[0] == "$" and isinstance(x[3], dict) and "a" in x[3] and isinstance(x[3]["a"], dict) and "lambdas" in x[3]["a"]:
                found.append(x[3]["a"])
                return
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    for line in flight.split("\n"):
        i = line.find(":")
        if i < 0 or not line[i + 1:].startswith(("[", "{")):
            continue
        try:
            walk(json.loads(line[i + 1:]))
        except json.JSONDecodeError:
            continue
        if found:
            break
    return found[0] if found else None


def slim(a: dict) -> dict:
    keep = ("fixture", "home", "away", "lambdas", "base", "factors", "strength", "absences", "lineup", "referee",
            "weather", "odds", "oddsSources", "generated")
    out = {k: a.get(k) for k in keep}
    out["players"] = {side: [{k: p.get(k) for k in ("id", "name", "pos", "apps", "clubApps", "starts", "min", "g", "a", "yc", "rc",
                                                     "sot", "sh", "g90", "a90", "yc90", "sot90", "expMin", "playMin", "avail",
                                                     "status", "attShare", "formF")}
                             for p in a.get("players", {}).get(side, [])] for side in ("h", "a")}
    if out.get("referee"):
        out["referee"] = {k: out["referee"].get(k) for k in ("known", "source", "compCardsPM", "fm")}
    return out


if __name__ == "__main__":
    home = s.get(SITE + "/", timeout=90).text
    ids = list(dict.fromkeys(re.findall(r'href="/zapas/(\d+)"', home)))
    m = pd.read_csv(E.PROC / "matches.csv.gz", dtype={"id": str})
    nat = set(m[m.comp.isin(["uefa.nations", "fifa.friendly"])].id)
    recent = m[(m.comp == "uefa.nations") & (m.date >= "2026-11-12")].id.tolist()
    todo = [i for i in dict.fromkeys(ids + recent) if i in nat or i in recent]
    print("zápasov na stiahnutie:", len(todo), flush=True)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(match_view, todo))
    snap = {i: slim(a) for i, a in zip(todo, res) if a}
    (E.PROC / "site_snapshot.json").write_text(json.dumps(snap, ensure_ascii=False))
    print("uložené:", len(snap), "zápasov;", "bez dát:", [i for i, a in zip(todo, res) if not a][:10])
