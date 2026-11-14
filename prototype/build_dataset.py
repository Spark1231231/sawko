"""Zo surových ESPN/GitHub súborov postaví čisté tabuľky v data/processed/.

matches.csv   – jeden riadok na zápas (skóre po 90 min, strely, karty, rohy, fauly, kurzy, neutrál)
events.csv    – góly a karty (hráč, minúta, penalta, vlastný gól)
lineups.csv   – hráč × zápas (základ, minúty, góly, strely, karty) zo summary
intl.csv      – svetová databáza medzištátnych výsledkov (martj42), mená zjednotené s ESPN
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DONE = {"STATUS_FULL_TIME", "STATUS_FINAL_PEN", "STATUS_FINAL_AET", "STATUS_FINAL"}

# ESPN meno -> meno v martj42/international_results
ALIAS = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Congo": "Congo",
    "IR Iran": "Iran",
    "China PR": "China PR",
    "Chinese Taipei": "Taiwan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Martin": "Saint Martin",
    "Sint Maarten": "Sint Maarten",
    "Curacao": "Curaçao",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "Timor-Leste": "East Timor",
    "Hong Kong, China": "Hong Kong",
    "Macau": "Macau",
    "US Virgin Islands": "United States Virgin Islands",
    "British Virgin Islands": "British Virgin Islands",
    "São Tomé e Príncipe": "São Tomé and Príncipe",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Gambia": "Gambia",
    "The Gambia": "Gambia",
    "Guinea Bissau": "Guinea-Bissau",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Turks & Caicos Islands": "Turks and Caicos Islands",
    "Bonaire": "Bonaire",
    "Northern Mariana Islands": "Northern Mariana Islands",
    "Ireland": "Republic of Ireland",
    "Serbia and Montenegro": "Serbia and Montenegro",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Myanmar": "Myanmar",
    "Palestine": "Palestine",
    "Tahiti": "Tahiti",
    "Eswatini": "Eswatini",
}


def canon(name: str) -> str:
    return ALIAS.get(name, name)


def minute(disp: str | None) -> tuple[float, int, bool]:
    """'45'+2'' -> (47.0, 45, False); '105'' -> (105.0, 105, True = predĺženie)."""
    if not disp:
        return np.nan, -1, False
    m = re.match(r"(\d+)'(?:\+(\d+)')?", disp)
    if not m:
        return np.nan, -1, False
    base, add = int(m.group(1)), int(m.group(2) or 0)
    return float(base + add), base, base > 90


def stat(comp: dict, name: str) -> float:
    for s in comp.get("statistics", []) or []:
        if s.get("name") == name:
            try:
                return float(s.get("displayValue"))
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def american(ml) -> float:
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    if ml == 0 or not np.isfinite(ml):
        return np.nan
    return 1 + (ml / 100 if ml > 0 else 100 / -ml)


def build_matches():
    rows, events = [], []
    seen = set()
    for path in sorted((RAW / "scoreboard").glob("*.json.gz")):
        comp = path.name.rsplit("_", 1)[0]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for ev in data.get("events", []):
            if ev["id"] in seen:
                continue
            seen.add(ev["id"])
            c = ev["competitions"][0]
            st = c["status"]["type"]["name"]
            cs = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in cs or "away" not in cs:
                continue
            h, a = cs["home"], cs["away"]
            venue = c.get("venue") or {}
            country = (venue.get("address") or {}).get("country")
            hn, an = canon(h["team"]["displayName"]), canon(a["team"]["displayName"])
            row = dict(
                id=ev["id"], date=ev["date"], comp=comp, season=ev.get("season", {}).get("year"),
                stage=ev.get("season", {}).get("slug"), status=st, done=st in DONE,
                hid=h["team"]["id"], aid=a["team"]["id"], home=hn, away=an,
                venue=venue.get("fullName"), city=(venue.get("address") or {}).get("city"), country=country,
                attendance=c.get("attendance") or np.nan,
            )
            # neutrál: štadión nie je v krajine domáceho tímu (ak krajinu poznáme)
            if country:
                row["neutral"] = canon(country) != hn
            else:
                row["neutral"] = comp in ("fifa.world", "uefa.euro", "conmebol.america", "concacaf.gold", "caf.nations")
            row["hg_final"] = float(h.get("score") or 0) if row["done"] else np.nan
            row["ag_final"] = float(a.get("score") or 0) if row["done"] else np.nan
            det = c.get("details") or []
            cnt = {k: 0 for k in ("hg90", "ag90", "hy", "ay", "hr", "ar", "hpen", "apen", "hg_et", "ag_et", "hg1", "ag1")}
            for x in det:
                tid = (x.get("team") or {}).get("id")
                side = "h" if tid == h["team"]["id"] else "a" if tid == a["team"]["id"] else None
                if side is None:
                    continue
                mn, base, et = minute((x.get("clock") or {}).get("displayValue"))
                ath = (x.get("athletesInvolved") or [{}])[0]
                kind = None
                if x.get("shootout"):
                    kind = "shootout"
                elif x.get("scoringPlay"):
                    kind = "goal"
                    cnt[f"{side}g_et" if et else f"{side}g90"] += 1
                    if 0 <= base <= 45:
                        cnt[f"{side}g1"] += 1
                    if x.get("penaltyKick"):
                        cnt[f"{side}pen"] += 1
                elif x.get("yellowCard"):
                    kind = "yellow"
                    if not et:
                        cnt[f"{side}y"] += 1
                elif x.get("redCard"):
                    kind = "red"
                    if not et:
                        cnt[f"{side}r"] += 1
                if kind:
                    events.append(dict(
                        id=ev["id"], date=ev["date"], comp=comp, side=side, team=hn if side == "h" else an,
                        kind=kind, minute=mn, et=et, pen=bool(x.get("penaltyKick")), og=bool(x.get("ownGoal")),
                        pid=ath.get("id"), player=ath.get("displayName"), pos=ath.get("position"),
                        text=(x.get("type") or {}).get("text"),
                    ))
            row["has_details"] = len(det) > 0
            # skóre po 90 minútach: pri predĺžení/penaltách z udalostí, inak finálne
            if row["done"]:
                if st in ("STATUS_FINAL_PEN", "STATUS_FINAL_AET") and det:
                    row["hg"], row["ag"] = cnt["hg90"], cnt["ag90"]
                elif st in ("STATUS_FINAL_PEN", "STATUS_FINAL_AET"):
                    row["hg"], row["ag"] = np.nan, np.nan
                else:
                    row["hg"], row["ag"] = row["hg_final"], row["ag_final"]
            else:
                row["hg"], row["ag"] = np.nan, np.nan
            det_ok = row["done"] and (len(det) > 0 or (row["hg_final"] == 0 and row["ag_final"] == 0))
            # kontrola: počet gólov v udalostiach sedí so skóre -> udalosti sú úplné
            goals_match = row["done"] and (cnt["hg90"] + cnt["hg_et"] == row["hg_final"]) and (cnt["ag90"] + cnt["ag_et"] == row["ag_final"])
            row["events_ok"] = bool(det_ok and goals_match)
            for k in ("hy", "ay", "hr", "ar", "hpen", "apen", "hg1", "ag1"):
                row[k] = cnt[k] if row["events_ok"] else np.nan
            for side, comp_ in (("h", h), ("a", a)):
                row[f"{side}sh"] = stat(comp_, "totalShots")
                row[f"{side}sot"] = stat(comp_, "shotsOnTarget")
                row[f"{side}cor"] = stat(comp_, "wonCorners")
                row[f"{side}fouls"] = stat(comp_, "foulsCommitted")
                row[f"{side}poss"] = stat(comp_, "possessionPct")
            # štatistiky sú platné len ak aspoň jeden tím má strelu alebo držanie
            ok = row["done"] and ((row["hsh"] or 0) + (row["ash"] or 0) > 0) and (row["hposs"] or 0) > 0
            row["stats_ok"] = bool(ok)
            if not ok:
                for k in ("hsh", "ash", "hsot", "asot", "hcor", "acor", "hfouls", "afouls", "hposs", "aposs"):
                    row[k] = np.nan
            rows.append(row)
    m = pd.DataFrame(rows).sort_values(["date", "id"]).reset_index(drop=True)
    e = pd.DataFrame(events)
    return m, e


def build_summaries(matches: pd.DataFrame):
    odds_rows, lu_rows, box_rows = [], [], []
    idx = matches.set_index("id")
    for path in sorted((RAW / "summary").glob("*.json.gz")):
        eid = path.name.split(".")[0]
        if eid not in idx.index:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        hid, aid = idx.at[eid, "hid"], idx.at[eid, "aid"]
        for o in (d.get("pickcenter") or d.get("odds") or [])[:1]:
            if not o:
                continue
            ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
            odds_rows.append(dict(
                id=eid, book=(o.get("provider") or {}).get("name"),
                oh=american(ho.get("moneyLine")), od=american((o.get("drawOdds") or {}).get("moneyLine")),
                oa=american(ao.get("moneyLine")), ou_line=o.get("overUnder"),
                o_over=american(o.get("overOdds")), o_under=american(o.get("underOdds")),
            ))
        off = ((d.get("gameInfo") or {}).get("officials") or [])
        ref = off[0].get("displayName") if off else None
        bx = {"id": eid, "referee": ref}
        for t in ((d.get("boxscore") or {}).get("teams") or []):
            side = "h" if t["team"]["id"] == hid else "a"
            st = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
            for k, name in (("y", "yellowCards"), ("r", "redCards"), ("off", "offsides"), ("blk", "blockedShots"),
                            ("pkg", "penaltyKickGoals"), ("pks", "penaltyKickShots")):
                try:
                    bx[f"{side}{k}_box"] = float(st.get(name))
                except (TypeError, ValueError):
                    bx[f"{side}{k}_box"] = np.nan
        box_rows.append(bx)
        for t in d.get("rosters") or []:
            side = "h" if t["team"]["id"] == hid else "a"
            for p in t.get("roster") or []:
                s = {x["name"]: x.get("value") for x in p.get("stats", [])}
                ath = p.get("athlete") or {}
                lu_rows.append(dict(
                    id=eid, side=side, pid=ath.get("id"), player=ath.get("displayName"),
                    pos=(p.get("position") or {}).get("abbreviation"), starter=bool(p.get("starter")),
                    sub_in=bool(p.get("subbedIn")), sub_out=bool(p.get("subbedOut")),
                    played=bool(p.get("starter")) or bool(p.get("subbedIn")),
                    g=s.get("totalGoals", 0), sh=s.get("totalShots", 0), sot=s.get("shotsOnTarget", 0),
                    yc=s.get("yellowCards", 0), rc=s.get("redCards", 0), fc=s.get("foulsCommitted", 0),
                    ast=s.get("goalAssists", 0), og=s.get("ownGoals", 0), formation=t.get("formation"),
                ))
    return pd.DataFrame(odds_rows), pd.DataFrame(lu_rows), pd.DataFrame(box_rows)


def build_intl():
    df = pd.read_csv(RAW / "intl" / "results.csv")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.rename(columns={"home_team": "home", "away_team": "away", "home_score": "hg", "away_score": "ag"})
    gs = pd.read_csv(RAW / "intl" / "goalscorers.csv")
    return df[["date", "home", "away", "hg", "ag", "tournament", "city", "country", "neutral"]], gs


if __name__ == "__main__":
    m, e = build_matches()
    odds, lu, box = build_summaries(m)
    if len(odds):
        m = m.merge(odds, on="id", how="left")
    if len(box):
        m = m.merge(box, on="id", how="left")
    m.to_csv(OUT / "matches.csv.gz", index=False)
    e.to_csv(OUT / "events.csv.gz", index=False)
    lu.to_csv(OUT / "lineups.csv.gz", index=False)
    intl, gs = build_intl()
    intl.to_csv(OUT / "intl.csv.gz", index=False)
    gs.to_csv(OUT / "intl_goalscorers.csv.gz", index=False)
    done = m[m.done]
    print(f"zápasy ESPN: {len(m)} (odohrané {len(done)}), so štatistikami {int(done.stats_ok.sum())}, "
          f"s úplnými udalosťami {int(done.events_ok.sum())}, s kurzami {int(m.get('oh', pd.Series(dtype=float)).notna().sum())}")
    print(f"udalosti: {len(e)}, zostavy: {len(lu)} riadkov / {lu['id'].nunique() if len(lu) else 0} zápasov")
    print(f"intl: {len(intl)} zápasov do {intl.date.max()}")
