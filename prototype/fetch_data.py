"""Stiahne surové dáta pre prototyp.

Zdroje (rovnaké, aké používa web – ESPN verejný feed – plus svetová databáza
medzištátnych výsledkov):

* ESPN scoreboard po rokoch: výsledok, strely, strely na bránu, rohy, fauly,
  góly (strelec, minúta) a karty (hráč, minúta) každého zápasu.
* ESPN summary zápasu: zostavy (základ/striedanie, post, štatistiky hráča) a
  záverečné kurzy DraftKings (1X2, over/under) – benchmark trhu.
* martj42/international_results (GitHub): všetky medzištátne zápasy od 1872.

Použitie:  python3 fetch_data.py scoreboards | summaries | intl | all
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Reprezentačné súťaže na ESPN. Prvých šesť používa aj web (NAT_COMPS).
NAT_COMPS = [
    "uefa.nations", "fifa.worldq.uefa", "uefa.euroq", "uefa.euro", "fifa.world", "fifa.friendly",
    "fifa.worldq.conmebol", "fifa.worldq.concacaf", "fifa.worldq.caf", "fifa.worldq.afc",
    "conmebol.america", "concacaf.gold", "concacaf.nations.league", "caf.nations",
]
YEARS = list(range(2016, 2027))
INTL_FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]
INTL_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"

session = requests.Session()
session.headers["User-Agent"] = "match-prediction-prototype/1.0"


def get_json(url: str, tries: int = 4):
    for i in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def fetch_scoreboards(force_recent: bool = True) -> None:
    out = RAW / "scoreboard"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, y) for c in NAT_COMPS for y in YEARS]

    def one(job):
        comp, year = job
        path = out / f"{comp}_{year}.json.gz"
        if path.exists() and not (force_recent and year >= 2026):
            return comp, year, None
        data = get_json(f"{ESPN}/{comp}/scoreboard?dates={year}&limit=1000")
        n = len(data.get("events", [])) if data else 0
        if data is not None:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        return comp, year, n

    with ThreadPoolExecutor(6) as ex:
        for comp, year, n in ex.map(one, jobs):
            if n is not None:
                print(f"{comp} {year}: {n} zápasov", flush=True)


def iter_events():
    for path in sorted((RAW / "scoreboard").glob("*.json.gz")):
        comp = path.name.rsplit("_", 1)[0]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("events", []):
            yield comp, ev


def fetch_summaries(since: str = "2021-07-01", comps: list[str] | None = None, refresh_after: str | None = None) -> None:
    """Zostavy + kurzy pre odohrané zápasy od `since` (predvolene súťaže, ktoré používa web)."""
    out = RAW / "summary"
    out.mkdir(parents=True, exist_ok=True)
    comps = comps or NAT_COMPS[:6]
    jobs = []
    for comp, ev in iter_events():
        if comp not in comps or ev["date"] < since:
            continue
        state = ev["competitions"][0]["status"]["type"]["state"]
        path = out / f"{ev['id']}.json.gz"
        if state != "post":
            continue
        if path.exists() and not (refresh_after and ev["date"] >= refresh_after):
            continue
        jobs.append((comp, ev["id"], path))
    print(f"summaries na stiahnutie: {len(jobs)}", flush=True)

    keep = ("boxscore", "gameInfo", "pickcenter", "odds", "rosters", "keyEvents", "header")

    def one(job):
        comp, eid, path = job
        data = get_json(f"{ESPN}/{comp}/summary?event={eid}")
        if data is None:
            return eid, False
        slim = {k: data.get(k) for k in keep}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(slim, f)
        return eid, True

    done = fail = 0
    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            _, ok = fut.result()
            done += ok
            fail += not ok
            if (done + fail) % 200 == 0:
                print(f"  {done + fail}/{len(jobs)} (chýb {fail})", flush=True)
    print(f"hotovo: {done}, zlyhalo: {fail}", flush=True)


def fetch_intl() -> None:
    out = RAW / "intl"
    out.mkdir(parents=True, exist_ok=True)
    for name in INTL_FILES:
        r = session.get(f"{INTL_BASE}/{name}", timeout=120)
        r.raise_for_status()
        (out / name).write_bytes(r.content)
        print(f"{name}: {len(r.content)} B", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("intl", "all"):
        fetch_intl()
    if what in ("scoreboards", "all"):
        fetch_scoreboards()
    if what in ("summaries", "all"):
        fetch_summaries()
