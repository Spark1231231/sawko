"""Model strelcov.

Góly tímu sú Poissonov proces s intenzitou λ_tím. Každý gól „pripadne“ hráčovi i s pravdepodobnosťou
θ_i (jeho podiel na góloch tímu, kým je na ihrisku). Riedenie Poissonovho procesu (thinning) dáva:

    góly hráča i ~ Poisson(λ_tím · θ_i · m_i),      P(skóruje) = 1 − exp(−λ_tím · θ_i · m_i)

m_i = očakávaný podiel odohraného času (základ ≈ 0,86; striedajúci ≈ 0,27).

θ_i sa odhaduje empirickým Bayesom – hráčove doterajšie góly sa zmrštia k priemeru jeho postu:

    θ_i = (G_i + κ·θ_post) / (X_i + κ),    X_i = Σ λ_tím·m  (očakávané góly tímu, kým hral)

Voliteľne sa primieša podiel na strelách na bránu (viac udalostí = menej šumu). Podiely sa nakoniec
normujú tak, aby základná jedenástka dala dokopy obvyklý podiel gólov tímu.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

import engine as E

M_FULL, M_OFF, M_SUB = 1.0, 0.70, 0.27
GROUPS = ("G", "CD", "FB", "DM", "CM", "WM", "AM", "AW", "WF", "CF", "F")
# východiskový podiel na góloch tímu / strelách na bránu podľa postu (prepíše sa dátami)
PRIOR_THETA = dict(G=0.001, CD=0.03, FB=0.03, DM=0.04, CM=0.07, WM=0.10, AM=0.13, AW=0.14, WF=0.17, CF=0.20, F=0.24)
PRIOR_PSI = dict(G=0.002, CD=0.07, FB=0.08, DM=0.12, CM=0.20, WM=0.28, AM=0.36, AW=0.38, WF=0.45, CF=0.50, F=0.60)


def pos_group(pos: str | None) -> str | None:
    """ESPN post -> skupina. Krídla (AM-L/R, LM/RM, LF/RF) sú zvlášť – strieľajú inak než stredoví hráči."""
    if not isinstance(pos, str) or pos == "SUB":
        return None
    if pos == "G":
        return "G"
    if pos.startswith(("CD", "SW")) or pos == "D":
        return "CD"
    if pos in ("LB", "RB"):
        return "FB"
    if pos == "DM":
        return "DM"
    if pos.startswith("CM") or pos == "M":
        return "CM"
    if pos in ("LM", "RM"):
        return "WM"
    if pos == "AM":
        return "AM"
    if pos in ("AM-L", "AM-R"):
        return "AW"
    if pos in ("LF", "RF"):
        return "WF"
    if pos.startswith(("CF", "RCF")):
        return "CF"
    if pos.startswith(("F", "ST")):
        return "F"
    return "CM"


class ScorerModel:
    def __init__(self, kappa: float = 8.0, alpha: float = 0.6, kappa_sot: float = 4.0, gamma: float = 1.0):
        # gamma > 1 „zaostrí“ podiely (θ^gamma pred normovaním): empirický Bayes ich inak priveľmi splošťuje k priemeru postu
        self.kappa, self.alpha, self.kappa_sot, self.gamma = kappa, alpha, kappa_sot, gamma
        self.G = defaultdict(float)       # góly hráča
        self.S = defaultdict(float)       # strely na bránu hráča
        self.X = defaultdict(float)       # expozícia: očakávané góly tímu počas jeho minút
        self.XS = defaultdict(float)      # tá istá expozícia, len zo zápasov s platnou štatistikou striel
        self.N = defaultdict(float)       # počet zápasov
        self.grp_count = defaultdict(lambda: defaultdict(float))   # hráč -> post -> počet štartov
        self.gG = {g: 0.0 for g in GROUPS}
        self.gS = {g: 0.0 for g in GROUPS}
        self.gX = {g: 0.0 for g in GROUPS}
        self.gXS = {g: 0.0 for g in GROUPS}
        self.start_goals = 0.0
        self.team_exp = 0.0
        self.name = {}
        self.last_team = {}

    # ------------------------------------------------------------------ pomocné
    def group(self, pid, pos=None) -> str:
        g = pos_group(pos)
        if g:
            return g
        c = self.grp_count.get(pid)
        return max(c, key=c.get) if c else "CM"

    def theta_pos(self, g: str) -> float:
        return (self.gG[g] + 20 * PRIOR_THETA[g]) / (self.gX[g] + 20)

    def psi_pos(self, g: str) -> float:
        return (self.gS[g] + 20 * PRIOR_PSI[g]) / (self.gXS[g] + 20)

    def theta(self, pid, g: str) -> float:
        """Podiel hráča na góloch tímu (na 90 minút na ihrisku)."""
        tp = self.theta_pos(g)
        th_goal = (self.G[pid] + self.kappa * tp) / (self.X[pid] + self.kappa)
        if self.alpha >= 1:
            return th_goal
        pp = self.psi_pos(g)
        psi = (self.S[pid] + self.kappa_sot * pp) / (self.XS[pid] + self.kappa_sot)
        conv = tp / max(pp, 1e-9)                       # góly na strelu na bránu podľa postu
        return self.alpha * th_goal + (1 - self.alpha) * psi * conv

    def starters_share(self) -> float:
        return (self.start_goals + 0.80 * 50) / (self.team_exp + 50)

    # ------------------------------------------------------------------ predpoveď
    def predict(self, lam_team: float, players: list[tuple], normalise: bool = True):
        """players: [(pid, post, m)] – m = očakávaný podiel času. Vracia (P(skóruje), očakávané góly)."""
        th = np.array([self.theta(pid, self.group(pid, pos)) for pid, pos, _ in players]) ** self.gamma
        m = np.array([p[2] for p in players], float)
        if normalise and (th * m).sum() > 0:
            starters = m >= 0.5
            target = self.starters_share() * (m[starters].sum() / (11 * 0.86)) if starters.any() else None
            if target:
                th = th * target / max((th[starters] * m[starters]).sum(), 1e-9)
        mu = lam_team * th * m
        return 1 - np.exp(-mu), mu

    # ------------------------------------------------------------------ učenie
    def update(self, lam_team: float, rows: pd.DataFrame, team: str | None = None, sot_ok: bool = True):
        """rows = zostava jedného tímu v jednom zápase (stĺpce pid, pos, starter, sub_in, sub_out, g, sot)."""
        for r in rows.itertuples(index=False):
            if not r.played:
                continue
            m = (M_OFF if r.sub_out else M_FULL) if r.starter else M_SUB
            g = self.group(r.pid, r.pos)
            x = lam_team * m
            self.G[r.pid] += r.g
            self.X[r.pid] += x
            self.N[r.pid] += 1
            self.gG[g] += r.g
            self.gX[g] += x
            if sot_ok:
                self.S[r.pid] += r.sot
                self.XS[r.pid] += x
                self.gS[g] += r.sot
                self.gXS[g] += x
            if r.starter:
                pg = pos_group(r.pos)
                if pg:
                    self.grp_count[r.pid][pg] += 1
                self.start_goals += r.g
            self.name[r.pid] = r.player
            if team:
                self.last_team[r.pid] = team
        self.team_exp += lam_team


def load_lineups() -> pd.DataFrame:
    """Zostavy + góly hráčov z UDALOSTÍ zápasu (hráčske štatistiky ESPN v časti zápasov chýbajú).

    g_ok   – súčet gólov hráčov sedí so skóre tímu (± vlastný gól súpera) -> zápas sa dá použiť,
    sot_ok – súčet striel na bránu hráčov sedí s tímovou štatistikou.
    """
    lu = pd.read_csv(E.PROC / "lineups.csv.gz", dtype={"id": str, "pid": str})
    for c in ("g", "sot", "sh", "yc", "rc"):
        lu[c] = lu[c].fillna(0).astype(float)
    ev = pd.read_csv(E.PROC / "events.csv.gz", dtype={"id": str, "pid": str})
    gl = ev[(ev.kind == "goal") & (~ev.og.astype(bool))].groupby(["id", "pid"]).size().rename("g_ev").reset_index()
    lu = lu.merge(gl, on=["id", "pid"], how="left")
    lu["g"] = lu.g_ev.fillna(0.0)
    m = pd.read_csv(E.PROC / "matches.csv.gz", dtype={"id": str}).set_index("id")
    t = lu.groupby(["id", "side"]).agg(gs=("g", "sum"), ss=("sot", "sum")).reset_index()
    t["tg"] = [m.at[i, "hg_final"] if sd == "h" else m.at[i, "ag_final"] for i, sd in zip(t.id, t.side)]
    t["ts"] = [m.at[i, "hsot"] if sd == "h" else m.at[i, "asot"] for i, sd in zip(t.id, t.side)]
    t["evok"] = [bool(m.at[i, "events_ok"]) for i in t.id]
    t["g_ok"] = t.evok & (t.gs <= t.tg) & (t.gs >= t.tg - 1)
    t["sot_ok"] = t.ts.notna() & (t.ss == t.ts)
    return lu.merge(t[["id", "side", "g_ok", "sot_ok"]], on=["id", "side"], how="left")


def backtest(w: pd.DataFrame, lam: dict, lu: pd.DataFrame, params: dict, eval_from: str = "2023-01-01",
             comps=E.EVAL_COMPS, collect: bool = False, lam_pred: dict | None = None):
    """lam_pred: voliteľne iné λ tímu použité LEN pri predpovedi (napr. zmes s trhom); učenie ostáva na lam."""
    """lam: espn_id -> (λ_dom, λ_host, dátum, súťaž). Chronologicky: predpovedz základné jedenástky, potom uč."""
    model = ScorerModel(**params)
    base = ScorerModel(kappa=1e9, alpha=1.0)        # základ: iba priemer postu
    order = sorted((v[2], k) for k, v in lam.items())
    groups = dict(tuple(lu.groupby("id")))
    out = []
    for date, eid in order:
        if eid not in groups:
            continue
        lh, la, _, comp = lam[eid]
        g = groups[eid]
        for side, lt in (("h", lh), ("a", la)):
            t = g[g.side == side]
            st = t[t.starter]
            if len(st) != 11 or not bool(t.g_ok.iloc[0]):
                continue
            if date >= eval_from and comp in comps:
                pl = [(r.pid, r.pos, 0.86) for r in st.itertuples(index=False)]
                if lam_pred is not None and eid in lam_pred:
                    lt = lam_pred[eid][0 if side == "h" else 1]
                p, _ = model.predict(lt, pl)
                pb, _ = base.predict(lt, pl)
                y = (st.g.values > 0).astype(float)
                for i, r in enumerate(st.itertuples(index=False)):
                    out.append((date, eid, side, r.pid, r.player, pos_group(r.pos) or "CM", p[i], pb[i], y[i], model.N[r.pid]))
        for side, lt in (("h", lh), ("a", la)):
            t = g[g.side == side]
            if len(t) == 0 or not bool(t.g_ok.iloc[0]):
                continue
            model.update(lt, t, sot_ok=bool(t.sot_ok.iloc[0]))
            base.update(lt, t, sot_ok=bool(t.sot_ok.iloc[0]))
    df = pd.DataFrame(out, columns=["date", "id", "side", "pid", "player", "grp", "p", "pb", "y", "n_hist"])
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
    res = dict(n=len(df), rate=float(df.y.mean()), ll_model=ll(df.p.values, df.y.values), ll_base=ll(df.pb.values, df.y.values))
    # úspešnosť „najpravdepodobnejšieho strelca tímu“
    top = df.loc[df.groupby(["id", "side"]).p.idxmax()]
    topb = df.loc[df.groupby(["id", "side"]).pb.idxmax()]
    res["top1_hit_model"], res["top1_p_model"] = float(top.y.mean()), float(top.p.mean())
    res["top1_hit_base"] = float(topb.y.mean())
    return (res, df, model) if collect else res
