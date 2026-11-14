"""Ostrá predpoveď zápasu: víťaz, skóre, karty, strely, strelci.

    python3 predict.py 401861167            # ESPN id zápasu (musí byť v data/processed/site_snapshot.json)
    python3 predict.py 401861167 --no-market

Reťazec:
  1. sila tímov (časovo vážený Poisson, MAP) + Elo  ->  surové λ
  2. stacking + kalibrácia 1/X/2 (naučené z minulých out-of-sample predpovedí)  ->  model
  3. trh (kurzy bez marže) -> λ trhu; mix 85 % trh + 15 % model (váha overená backtestom)
  4. počty (strely, strely na bránu, rohy, karty) z rozdielu síl + tímových efektov, NB rozdelenie
  5. strelci: riedenie Poissonovho procesu gólov tímu podľa podielu hráča
"""
from __future__ import annotations

import json
import sys
import unicodedata

import numpy as np
import pandas as pd

import calib as C
import counts as K
import engine as E
import scorers as S
import walk as W

MARKET_WEIGHT = 0.85
RES = E.ROOT / "results"


_TR = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "ð": "d"})


def norm_name(x: str) -> str:
    x = unicodedata.normalize("NFKD", x.translate(_TR)).encode("ascii", "ignore").decode().lower()
    return " ".join(x.replace("-", " ").replace("'", "").split())


def match_name(name: str, cands: dict):
    """Nájde hráča v slovníku {normalizované meno: hodnota} – presne, podmnožinou slov, alebo podľa priezviska."""
    n = norm_name(name)
    if n in cands:
        return cands[n]
    toks = set(n.split())
    best, score, tie = None, 0, False
    for k, v in cands.items():
        kt = set(k.split())
        one = len(toks) == 1 or len(kt) == 1          # priezvisko samotné stačí len ak jedno z mien je jednoslovné
        sc = len(toks & kt) if (toks <= kt or kt <= toks) else (0.5 if one and n.split()[-1] == k.split()[-1] else 0)
        if sc > score:
            best, score, tie = v, sc, False
        elif sc == score and sc > 0 and v is not best:
            tie = True
    return None if tie else best


class Predictor:
    def __init__(self, as_of: str | None = None, with_counts: bool = True, with_scorers: bool = True):
        self.w = w = E.load_world()
        self.cfg = json.loads((RES / "goals_tuning.json").read_text())
        scfg = dict(self.cfg["strength_cfg"])
        scfg["imp"] = tuple(scfg["imp"])
        cal = json.loads((RES / "calibration.json").read_text())
        self.cal = (np.array(cal["beta"]), np.array(cal["theta"]))
        self.cal_world = (np.array(cal["world"]["beta"]), np.array(cal["world"]["theta"])) if "world" in cal else self.cal
        self.uefa = set(cal.get("uefa_teams", []))
        self.rho = self.cfg["rho"]
        self.D = int(w.day.max()) + 1 if as_of is None else (pd.Timestamp(as_of) - pd.Timestamp("1970-01-01")).days
        self.fit = W.fit_strength_at(w, self.D, scfg)
        sub = w[w.day < self.D]
        sub.attrs["teams"] = w.attrs["teams"]
        pre_h, pre_a, self.R = E.run_elo(sub, **self.cfg["elo_cfg"])
        tr = (sub.day >= self.D - int(20 * 365.25)).values
        self.beta_elo = E.fit_elo_goals(pre_h[tr], pre_a[tr], np.minimum(sub.hg.values[tr], 8), np.minimum(sub.ag.values[tr], 8), sub.neutral.values[tr])
        self.tid = {t: i for i, t in enumerate(w.attrs["teams"])}
        self.counts = {}
        self.sc = None
        if with_counts or with_scorers:
            z = np.load(E.PROC / "espn_walk.npz")
            bw = self.cfg["elo_weight"]
            self._lam_rows = z["rows"]
            self._lh, self._la = W.blend(z["lh"], z["eh"], bw), W.blend(z["la"], z["ea"], bw)
        if with_counts:
            self.frame = K.frame(w, self._lam_rows, self._lh, self._la)
            ccfg = json.loads((RES / "counts_cfg.json").read_text())
            ev = self.frame.comp.isin(E.EVAL_COMPS).values
            for kind in ("shots", "sot", "corners", "yellow", "fouls"):
                self.counts[kind] = K.fit_live(self.frame, kind, ccfg.get(kind), self.D, eval_mask=ev)
            # červené karty: jednoduchá miera na tím a zápas (súťažné zápasy UEFA od 2021)
            f = self.frame[ev & (self.frame.date.values >= "2021-01-01") & self.frame.events_ok.fillna(False).astype(bool).values]
            self.red_rate = float((f.hr.sum() + f.ar.sum()) / (2 * len(f)))
            # efekt rozhodcu na karty
            res = self.counts["yellow"]["walk"]
            self.ref = {}
            fr = self.frame[self.frame.day < self.D].reset_index(drop=True)
            for i in res["target"]:
                r = fr.referee.values[i]
                if isinstance(r, str):
                    a, e, n = self.ref.get(norm_name(r), (0.0, 0.0, 0))
                    self.ref[norm_name(r)] = (a + res["yh"][i] + res["ya"][i], e + res["mh"][i] + res["ma"][i], n + 1)
        if with_scorers:
            lam = {eid: (lh, la, d, c) for eid, lh, la, d, c in
                   zip(w.espn_id.values[self._lam_rows], self._lh, self._la, w.date.values[self._lam_rows], w.comp.values[self._lam_rows])}
            sp = json.loads((RES / "scorers_backtest.json").read_text())["best"]
            self.sc = S.ScorerModel(**sp)
            lu = S.load_lineups()
            m = pd.read_csv(E.PROC / "matches.csv.gz", dtype={"id": str}).set_index("id")
            self._lu, self._m = lu, m
            for eid, g in sorted(lu.groupby("id"), key=lambda kv: lam[kv[0]][2] if kv[0] in lam else "9"):
                if eid not in lam:
                    continue
                lh, la, _, _ = lam[eid]
                for side, lt in (("h", lh), ("a", la)):
                    t = g[g.side == side]
                    if len(t) and bool(t.g_ok.iloc[0]):
                        self.sc.update(lt, t, team=m.at[eid, "home"] if side == "h" else m.at[eid, "away"], sot_ok=bool(t.sot_ok.iloc[0]))
            self.pid_by_name = {}
            for pid, name in self.sc.name.items():
                self.pid_by_name.setdefault(self.sc.last_team.get(pid), {})[norm_name(name)] = pid

    # ------------------------------------------------------------------ góly
    def raw(self, home: str, away: str, neutral: bool = False):
        h, a = self.tid[home], self.tid[away]
        f = self.fit
        hf = 0.0 if neutral else 1.0
        ls_h = float(np.exp(f["mu"] + f["gam"] * hf + f["att"][h] - f["dfn"][a]))
        ls_a = float(np.exp(f["mu"] + f["att"][a] - f["dfn"][h]))
        le_h, le_a = E.elo_lambdas(self.beta_elo, np.array([self.R[h]]), np.array([self.R[a]]), np.array([neutral]))
        return ls_h, ls_a, float(le_h[0]), float(le_a[0])

    def goals(self, home: str, away: str, neutral: bool = False) -> dict:
        ls_h, ls_a, le_h, le_a = self.raw(home, away, neutral)
        hf = 0.0 if neutral else 1.0
        # kalibrácia podľa populácie: dva európske tímy -> „uefa“, inak „world“
        european = (home in self.uefa and away in self.uefa) if self.uefa else True
        ch, ca, M, P = C.predict_calibrated(self.cal if european else self.cal_world, ls_h, ls_a, le_h, le_a, hf, self.rho)
        h, a = self.tid[home], self.tid[away]
        return dict(lh=float(ch[0]), la=float(ca[0]), M=M[0], P=P[0], raw=dict(sila=(ls_h, ls_a), elo=(le_h, le_a)),
                    calibration="uefa" if european else "world",
                    elo=(float(self.R[h]), float(self.R[a])), n_eff=(float(self.fit["n_eff"][h]), float(self.fit["n_eff"][a])),
                    att=(float(self.fit["att"][h]), float(self.fit["att"][a])), dfn=(float(self.fit["dfn"][h]), float(self.fit["dfn"][a])))

    def market(self, x12: dict, ou: dict | None = None, line: float = 2.5) -> dict:
        """x12 = {'1':…, 'X':…, '2':…} (kurzy), ou = {'over':…, 'under':…} na čiare `line`."""
        p = E.devig(x12["1"], x12["X"], x12["2"])
        po = None
        if ou:
            qo, qu = 1 / ou["over"], 1 / ou["under"]
            po = qo / (qo + qu)
        lh, la = E.lambdas_from_probs(float(p[0]), float(p[2]), po, self.rho, line=line)
        return dict(P=p, lh=lh, la=la, p_over=po, line=line, margin=float(1 / x12["1"] + 1 / x12["X"] + 1 / x12["2"] - 1))

    def blend(self, model: dict, mkt: dict, a: float = MARKET_WEIGHT) -> dict:
        lh = float(np.exp(a * np.log(mkt["lh"]) + (1 - a) * np.log(model["lh"])))
        la = float(np.exp(a * np.log(mkt["la"]) + (1 - a) * np.log(model["la"])))
        P = mkt["P"] ** a * model["P"] ** (1 - a)
        P = P / P.sum()
        M = C.reweight(E.score_matrix(lh, la, self.rho), P)
        return dict(lh=lh, la=la, P=P, M=M)

    # ------------------------------------------------------------------ počty
    def count(self, kind: str, home: str, away: str, lh: float, la: float, neutral: bool = False, referee: str | None = None,
              ref_prior: tuple | None = None) -> dict:
        fit = self.counts[kind]
        mh, ma = K.predict_live(fit, self.tid[home], self.tid[away], float(np.log(lh / la)), 0.0 if neutral else 1.0)
        f = 1.0
        info = None
        if kind == "yellow" and (referee or ref_prior):
            a, e, n = self.ref.get(norm_name(referee), (0.0, 0.0, 0)) if referee else (0.0, 0.0, 0)
            if ref_prior:              # doplnkový údaj mimo našich dát: (žlté na zápas, priemer súťaže, počet zápasov)
                ypm, avg, nm = ref_prior
                a, e, n = a + ypm * nm, e + avg * nm, n + nm
            f = (a + 80.0) / (e + 80.0)
            info = dict(n=n, factor=f)
        mh, ma = mh * f, ma * f
        tot = mh + ma
        pmf = K.nb_pmf(tot, fit["r_total"], 80)
        return dict(home=mh, away=ma, total=tot, r_total=fit["r_total"], r_team=fit["r_team"], pmf=pmf, ref=info,
                    mode=int(np.argmax(pmf)), q10=int(np.searchsorted(np.cumsum(pmf), 0.1)), q90=int(np.searchsorted(np.cumsum(pmf), 0.9)))

    # ------------------------------------------------------------------ strelci
    def expected_xi(self, team: str, n_matches: int = 6) -> dict | None:
        """Odhad základnej jedenástky, kým nie je potvrdená: najčastejší štartujúci z posledných zápasov tímu."""
        m = self._m
        ids = m[((m.home == team) | (m.away == team)) & m.done].sort_values("date").index[::-1]
        rows, used = [], 0
        for eid in ids:
            side = "h" if m.at[eid, "home"] == team else "a"
            t = self._lu[(self._lu.id == eid) & (self._lu.side == side) & self._lu.starter]
            if len(t) != 11:
                continue
            t = t.assign(w=0.8 ** used, date=m.at[eid, "date"])
            rows.append(t)
            used += 1
            if used >= n_matches:
                break
        if not rows:
            return None
        allr = pd.concat(rows)
        agg = allr.groupby("pid").agg(w=("w", "sum"), name=("player", "first"), pos=("pos", lambda x: x.value_counts().index[0])).sort_values("w", ascending=False)
        agg["grp"] = agg.pos.map(S.pos_group)
        # rozostavenie po postoch: 1 brankár, 4 obrancovia, 3 stredopoliari, 3 ofenzívni (z toho aspoň 1 hrotový útočník)
        take = []
        def pick(groups, k, exclude=()):
            c = agg[agg.grp.isin(groups) & ~agg.index.isin(list(exclude) + take)].head(k)
            take.extend(c.index.tolist())
        pick(["G"], 1)
        pick(["CD"], 2)
        pick(["FB"], 2)
        pick(["DM", "CM"], 2)
        pick(["F", "CF"], 1)
        pick(["AM", "AW", "WF", "WM", "F", "CF"], 3)
        if len(take) < 11:
            pick(list(S.GROUPS), 11 - len(take))
        xi = agg.loc[take]
        return dict(starters=[dict(name=r.name, espn_pos=r.pos) for r in xi.itertuples(index=False)], bench=[], formation=None,
                    based_on=used, last=str(allr.date.max())[:10])

    def scorer_shares(self, team: str, players: list[dict]) -> np.ndarray:
        """players: [{'name', 'pos' (G/D/M/F), 'm'}] -> podiel θ z reprezentačnej histórie (nenormovaný)."""
        default = {"G": "G", "D": "CD", "M": "CM", "F": "F"}
        th = []
        for p in players:
            pid = match_name(p["name"], self.pid_by_name.get(team, {}))
            grp = self.sc.group(pid) if pid and self.sc.grp_count.get(pid) else default.get(p.get("pos", "M"), "CM")
            th.append(self.sc.theta(pid if pid else "?" + p["name"], grp))
            p["_pid"], p["_grp"], p["_n"] = pid, grp, (self.sc.N[pid] if pid else 0)
        return np.array(th)


def top_scores(M: np.ndarray, k: int = 6):
    idx = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0][:k]
    return [(int(i), int(j), float(M[i, j])) for i, j in idx]


def summarize(M: np.ndarray) -> dict:
    P = E.outcome_probs(M)
    tot = E.TOT
    return dict(p1=float(P[0]), px=float(P[1]), p2=float(P[2]),
                over15=float((M * (tot > 1.5)).sum()), over25=float((M * (tot > 2.5)).sum()), over35=float((M * (tot > 3.5)).sum()),
                btts=float(M[1:, 1:].sum()), exp_h=float((M.sum(1) * np.arange(E.MAXG)).sum()), exp_a=float((M.sum(0) * np.arange(E.MAXG)).sum()),
                scores=top_scores(M))


# ====================================================================== predpoveď zápasu zo snapshotu webu
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def espn_lineups(eid: str, comp: str):
    """Potvrdené zostavy z ESPN (asi hodinu pred výkopom). Vracia {'h': [...], 'a': [...]} alebo None."""
    import requests
    try:
        d = requests.get(f"{ESPN}/{comp}/summary?event={eid}", timeout=60).json()
    except Exception:
        return None
    comps = (d.get("header") or {}).get("competitions") or [{}]
    hid = next((c["team"]["id"] for c in comps[0].get("competitors", []) if c.get("homeAway") == "home"), None)
    out = {}
    for t in d.get("rosters") or []:
        side = "h" if t["team"]["id"] == hid else "a"
        st = [dict(name=p["athlete"]["displayName"], espn_pos=(p.get("position") or {}).get("abbreviation"))
              for p in t.get("roster") or [] if p.get("starter")]
        bench = [p["athlete"]["displayName"] for p in t.get("roster") or [] if not p.get("starter")]
        if len(st) == 11:
            out[side] = dict(starters=st, bench=bench, formation=t.get("formation"))
    officials = ((d.get("gameInfo") or {}).get("officials") or [])
    return dict(lineups=out if len(out) == 2 else None, referee=officials[0].get("displayName") if officials else None)


def espn_fixture(eid: str, comp: str) -> dict:
    """Zápas mimo webu: postaví rovnaký vstup ako snapshot webu priamo z ESPN (tímy, miesto, kurzy DraftKings)."""
    import requests
    from build_dataset import american, canon
    d = requests.get(f"{ESPN}/{comp}/summary?event={eid}", timeout=60).json()
    c = d["header"]["competitions"][0]
    sides = {x["homeAway"]: x for x in c["competitors"]}
    venue = (d.get("gameInfo") or {}).get("venue") or {}
    country = (venue.get("address") or {}).get("country")
    home = sides["home"]["team"]["displayName"]
    neutral = bool(country) and canon(country) != canon(home)
    odds = {}
    o = (d.get("pickcenter") or d.get("odds") or [None])[0]
    if o:
        x12 = {"1": american((o.get("homeTeamOdds") or {}).get("moneyLine")), "X": american((o.get("drawOdds") or {}).get("moneyLine")),
               "2": american((o.get("awayTeamOdds") or {}).get("moneyLine"))}
        if all(v == v for v in x12.values()):
            odds["1x2"] = {(o.get("provider") or {}).get("name", "book"): x12}
        ov, un = american(o.get("overOdds")), american(o.get("underOdds"))
        if o.get("overUnder") and ov == ov and un == un:
            odds[f"ou:{o['overUnder']}"] = {(o.get("provider") or {}).get("name", "book"): {"over": ov, "under": un}}
    off = (d.get("gameInfo") or {}).get("officials") or []
    return dict(fixture=dict(id=eid, date=c["date"], neutral=neutral, venueName=venue.get("fullName"), comp=comp,
                             compName=(d["header"].get("season") or {}).get("name"), city=(venue.get("address") or {}).get("city")),
                home=dict(id=sides["home"]["team"]["id"], name=home), away=dict(id=sides["away"]["team"]["id"], name=sides["away"]["team"]["displayName"]),
                odds=odds, players={}, lineup={}, lambdas=dict(model=None, market=None), factors=[],
                referee=dict(known=dict(name=off[0].get("displayName")) if off else None), absences=None)


def best_x12(odds: dict):
    """Priemer férových pravdepodobností cez dostupné kancelárie + zoznam kurzov."""
    books = odds.get("1x2", {})
    ps = [E.devig(b["1"], b["X"], b["2"]) for b in books.values() if all(k in b for k in ("1", "X", "2"))]
    return (np.mean(ps, axis=0) if ps else None), books


def predict_match(pr: Predictor, a: dict, lineups: dict | None = None, referee: str | None = None, use_market: bool = True,
                  confirmed: bool | None = None) -> dict:
    from build_dataset import canon
    fx = a["fixture"]
    home, away = canon(a["home"]["name"]), canon(a["away"]["name"])
    neutral = bool(fx.get("neutral"))
    model = pr.goals(home, away, neutral)
    out = dict(id=fx["id"], date=fx["date"], home=home, away=away, venue=fx.get("venueName"), comp=fx.get("compName"))
    out["model"] = dict(lam=(model["lh"], model["la"]), **summarize(model["M"]), elo=model["elo"], raw=model["raw"])
    final = dict(lh=model["lh"], la=model["la"], M=model["M"], P=model["P"])
    pm, books = best_x12(a.get("odds") or {})
    if pm is not None:
        ou, line = None, 2.5
        for ln in ("2.5", "3.5", "1.5", "3", "2", "4.5"):
            b = (a["odds"].get(f"ou:{ln}") or {})
            b = b.get("Fortuna") or next(iter(b.values()), None)
            if b and "over" in b and "under" in b:
                ou, line = b, float(ln)
                break
        qo = (1 / ou["over"]) / (1 / ou["over"] + 1 / ou["under"]) if ou else None
        lh, la = E.lambdas_from_probs(float(pm[0]), float(pm[2]), qo, pr.rho, line=line)
        mkt = dict(P=pm, lh=lh, la=la)
        Mm = C.reweight(E.score_matrix(lh, la, pr.rho), pm)
        out["market"] = dict(lam=(lh, la), **summarize(Mm), books=books, ou=dict(line=line, **ou) if ou else None)
        if use_market:
            final = pr.blend(model, mkt)
    out["final"] = dict(lam=(final["lh"], final["la"]), **summarize(final["M"]), market_weight=MARKET_WEIGHT if (pm is not None and use_market) else 0.0)
    lh, la = final["lh"], final["la"]

    # ------------------------------------------------------------ počty
    ref_name = referee or ((a.get("referee") or {}).get("known") or {}).get("name")
    fm = (a.get("referee") or {}).get("fm") or {}
    ref_prior = (fm["yPM"], fm["yAvg"], fm["matches"]) if all(k in fm for k in ("yPM", "yAvg", "matches")) and fm.get("yAvg") else None
    out["referee"] = dict(name=ref_name, fotmob=fm or None)
    cnt = {}
    for kind in ("shots", "sot", "corners", "yellow", "fouls"):
        c = pr.count(kind, home, away, lh, la, neutral, referee=ref_name if kind == "yellow" else None,
                     ref_prior=ref_prior if kind == "yellow" else None)
        cdf = np.cumsum(c["pmf"])
        cnt[kind] = dict(home=c["home"], away=c["away"], total=c["total"], mode=c["mode"], q10=c["q10"], q90=c["q90"], r_total=c["r_total"],
                         ref=c["ref"], over={str(l): float(1 - cdf[int(np.floor(l))]) for l in
                                             {"shots": (20.5, 22.5, 24.5, 26.5, 28.5), "sot": (6.5, 7.5, 8.5, 9.5, 10.5), "corners": (7.5, 8.5, 9.5, 10.5, 11.5),
                                              "yellow": (1.5, 2.5, 3.5, 4.5, 5.5), "fouls": (20.5, 22.5, 24.5, 26.5)}[kind]},
                         pmf=[float(x) for x in c["pmf"][:15]] if kind == "yellow" else None)
    cnt["red"] = dict(per_team=pr.red_rate, p_any=float(1 - np.exp(-2 * pr.red_rate)))
    out["counts"] = cnt

    # ------------------------------------------------------------ strelci
    sc_odds = ((a.get("odds") or {}).get("scorer") or {}).get("Fortuna") or {}
    site_players = {side: {norm_name(p["name"]): p for p in (a.get("players") or {}).get(side, [])} for side in ("h", "a")}
    team_id = {"h": a["home"]["id"], "a": a["away"]["id"]}
    scor = {}
    confirmed = (lineups is not None) if confirmed is None else confirmed
    for side, team, lam_t in (("h", home, lh), ("a", away, la)):
        if lineups is not None:
            names = [(p["name"], p.get("espn_pos")) for p in lineups[side]["starters"]]
        else:
            names = [(n, None) for n in (a.get("lineup") or {}).get(side, [])]
        if len(names) != 11:
            continue
        pl = []
        team_odds = {k.split(":", 1)[1]: v for k, v in sc_odds.items() if k.startswith(team_id[side] + ":")}
        for idx, (name, epos) in enumerate(names):
            sp = match_name(name, site_players[side]) or {}
            m = float(np.clip((sp.get("playMin") or 77) / 90.0, 0.6, 1.0))
            # neznámy hráč v predpokladanej zostave: post podľa poradia v rozostavení (brankár, obrana, záloha, útok)
            fallback = "G" if idx == 0 else "D" if idx <= 4 else "M" if idx <= 8 else "F"
            pl.append(dict(name=name, pos=sp.get("pos", fallback), m=m, g90=sp.get("g90"), espn_pos=epos,
                           odds=match_name(name, team_odds)))
        th_nat = pr.scorer_shares(team, pl)
        for p, epos in zip(pl, [e for _, e in names]):
            g = S.pos_group(epos)
            if g:                                   # potvrdená zostava: presný post z rozostavenia
                p["_grp"] = g
        if lineups is not None:
            th_nat = np.array([pr.sc.theta(p["_pid"] if p["_pid"] else "?" + p["name"], p["_grp"]) for p in pl])
        m = np.array([p["m"] for p in pl])
        share = pr.sc.starters_share() * (m.sum() / (11 * 0.86))
        def normed(v):
            v = np.asarray(v, float)
            return v * share / max((v * m).sum(), 1e-12)
        comp = {"nat": normed(th_nat)}
        g90 = np.array([p["g90"] if p["g90"] is not None else np.nan for p in pl])
        if np.isfinite(g90).sum() >= 8:
            g90 = np.where(np.isfinite(g90), g90, np.nanmedian(g90))
            comp["club"] = normed(np.maximum(g90, 1e-4))
        mk = np.array([-np.log(1 - 1 / p["odds"]) / p["m"] if p["odds"] and p["odds"] > 1.01 else np.nan for p in pl])
        if np.isfinite(mk).sum() >= 6:
            # hráčom bez kurzu (brankár, stopéri) doplň podiel z ostatných zdrojov v mierke trhu
            ref = np.nanmean([comp[k] for k in comp], axis=0)
            scale = np.nansum(mk[np.isfinite(mk)]) / max(ref[np.isfinite(mk)].sum(), 1e-12)
            mk = np.where(np.isfinite(mk), mk, ref * scale)
            comp["market"] = normed(mk)
        wts = {"nat": 0.25, "club": 0.25, "market": 0.5}
        tot_w = sum(wts[k] for k in comp)
        logth = sum(wts[k] / tot_w * np.log(np.maximum(comp[k], 1e-6)) for k in comp)
        th = normed(np.exp(logth))
        mu = lam_t * th * m
        rows = []
        for i, p in enumerate(pl):
            rows.append(dict(name=p["name"], grp=p["_grp"], m=round(p["m"], 2), n_hist=int(p["_n"]), odds=p["odds"],
                             share=float(th[i] * m[i]), exp_goals=float(mu[i]), p_score=float(1 - np.exp(-mu[i])),
                             p_2plus=float(1 - np.exp(-mu[i]) * (1 + mu[i])),
                             parts={k: float(1 - np.exp(-lam_t * comp[k][i] * m[i])) for k in comp}))
        rows.sort(key=lambda r: -r["p_score"])
        scor[side] = dict(team=team, lam=lam_t, confirmed=confirmed, starters_share=float(share), players=rows)
    out["scorers"] = scor
    out["lineup_confirmed"] = confirmed
    out["absences"] = a.get("absences")
    out["calibration"] = model.get("calibration")
    out["neutral"] = neutral
    if (a.get("lambdas") or {}).get("model"):
        out["site"] = dict(lam_model=a["lambdas"]["model"], lam_market=a["lambdas"].get("market"),
                           factors=[{k: f.get(k) for k in ("key", "desc", "lh", "la", "p1", "px", "p2")} for f in a.get("factors", [])])
    return out


if __name__ == "__main__":
    eid = sys.argv[1]
    snap = json.loads((E.PROC / "site_snapshot.json").read_text())
    comp_arg = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--comp=")), None)
    a = snap[eid] if eid in snap else espn_fixture(eid, comp_arg or "uefa.nations")
    pr = Predictor()
    live = espn_lineups(eid, a["fixture"].get("comp", "uefa.nations")) if "--no-lineups" not in sys.argv else None
    lus, conf = (live or {}).get("lineups"), None
    if lus is None and not (a.get("lineup") or {}).get("h"):
        # web tento zápas nepokrýva a zostavy ešte nie sú: odhad jedenástky z posledných zápasov
        from build_dataset import canon
        exp = {sd: pr.expected_xi(canon(a[k]["name"])) for sd, k in (("h", "home"), ("a", "away"))}
        if all(exp.values()):
            lus, conf = exp, False
    res = predict_match(pr, a, lineups=lus, referee=(live or {}).get("referee"), use_market="--no-market" not in sys.argv, confirmed=conf)
    res["generated_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")
    (RES / f"prediction_{eid}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    f = res["final"]
    print(f"{res['home']} – {res['away']} | λ {f['lam'][0]:.2f}:{f['lam'][1]:.2f} | 1/X/2 {f['p1']:.3f}/{f['px']:.3f}/{f['p2']:.3f} | zostavy potvrdené: {res['lineup_confirmed']}")
    print("skóre:", [(f"{i}:{j}", round(p, 3)) for i, j, p in f["scores"]])
    for k in ("shots", "sot", "corners", "yellow", "fouls"):
        c = res["counts"][k]
        print(f"{k:8s} dom {c['home']:.2f} host {c['away']:.2f} spolu {c['total']:.2f} (modus {c['mode']}, 80 % interval {c['q10']}–{c['q90']}) over {c['over']}")
    for side in ("h", "a"):
        if side in res["scorers"]:
            print(res["scorers"][side]["team"], [(r["name"], round(r["p_score"], 3)) for r in res["scorers"][side]["players"][:6]])
