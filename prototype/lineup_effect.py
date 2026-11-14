"""Test: predpovedá zloženie základnej jedenástky góly nad rámec sily tímu? (overenie úpravy „marodka“)

Pre každý tím a zápas (zostavy sú známe ~1 h pred výkopom):
  x_att = log( Σ θ_i základnej jedenástky / bežná hodnota tímu )   – koľko streleckej sily je na ihrisku
  x_reg = podiel „stálych“ hráčov v jedenástke − bežná hodnota tímu   (stálosť = štarty v posledných 8 zápasoch)
Poissonova regresia gólov s offsetom log λ (walk-forward λ z modelu):
  góly_za ~ exp(log λ_za + b1·x_att + b2·x_reg_vlastné + b3·x_reg_súper)
Koeficienty sa učia na 2022–2023, overujú na 2024–2026. Výstup: results/lineup_effect.json"""
from __future__ import annotations

import json
from collections import defaultdict, deque

import numpy as np

import engine as E
import scorers as S
import scorers_eval as SE

if __name__ == "__main__":
    lam, lu = SE.LAM, SE.LU
    sp = json.loads((E.ROOT / "results" / "scorers_backtest.json").read_text())["best"]
    model = S.ScorerModel(**sp)
    groups = dict(tuple(lu.groupby("id")))
    order = sorted((v[2], k) for k, v in lam.items() if k in groups)
    m = SE.w.set_index("espn_id")
    starts = defaultdict(lambda: deque(maxlen=8))       # tím -> posledných 8 jedenástok (množiny pid)
    ema_att, ema_reg = {}, {}
    rows = []
    for date, eid in order:
        lh, la, _, comp = lam[eid]
        g = groups[eid]
        home, away = m.at[eid, "home"], m.at[eid, "away"]
        hg, ag = m.at[eid, "hg"], m.at[eid, "ag"]
        feat = {}
        for side, team in (("h", home), ("a", away)):
            st = g[(g.side == side) & g.starter]
            if len(st) != 11:
                continue
            A = float(sum(model.theta(r.pid, model.group(r.pid, r.pos)) for r in st.itertuples(index=False)))
            hist = starts[team]
            reg = float(np.mean([sum(r.pid in h_ for h_ in hist) / len(hist) for r in st.itertuples(index=False)])) if len(hist) >= 4 else np.nan
            xa = np.log(A / ema_att[team]) if team in ema_att else np.nan
            xr = reg - ema_reg[team] if (team in ema_reg and reg == reg) else np.nan
            feat[side] = (xa, xr, A, reg, set(st.pid))
        if len(feat) == 2 and all(np.isfinite(feat[s_][0]) and np.isfinite(feat[s_][1]) for s_ in "ha"):
            rows.append((date, comp, lh, hg, feat["h"][0], feat["h"][1], feat["a"][1]))
            rows.append((date, comp, la, ag, feat["a"][0], feat["a"][1], feat["h"][1]))
        for side, team, lt in (("h", home, lh), ("a", away, la)):
            if side in feat:
                _, _, A, reg, pids = feat[side]
                ema_att[team] = A if team not in ema_att else 0.8 * ema_att[team] + 0.2 * A
                if reg == reg:
                    ema_reg[team] = reg if team not in ema_reg else 0.8 * ema_reg[team] + 0.2 * reg
                starts[team].append(pids)
            t = g[g.side == side]
            if len(t) and bool(t.g_ok.iloc[0]):
                model.update(lt, t, sot_ok=bool(t.sot_ok.iloc[0]))
    R = np.array([(r[2], r[3], r[4], r[5], r[6]) for r in rows], float)
    date = np.array([r[0] for r in rows])
    comp = np.array([r[1] for r in rows])
    out = {}
    for label, pop in (("všetky zápasy", np.ones(len(R), bool)), ("súťažné", np.isin(comp, E.EVAL_COMPS))):
        tr, te = pop & (date < "2024-01-01"), pop & (date >= "2024-01-01")
        X = np.column_stack([np.ones(len(R)), R[:, 2], R[:, 3], R[:, 4]])
        # IRLS s offsetom
        beta = np.zeros(4)
        for _ in range(50):
            mu = R[tr, 0] * np.exp(X[tr] @ beta)
            H = (X[tr] * mu[:, None]).T @ X[tr] + 1e-6 * np.eye(4)
            beta += np.linalg.solve(H, X[tr].T @ (R[tr, 1] - mu))
        se = np.sqrt(np.diag(np.linalg.inv((X[tr] * (R[tr, 0] * np.exp(X[tr] @ beta))[:, None]).T @ X[tr])))
        def nll(b, mk):
            mu = R[mk, 0] * np.exp(X[mk] @ b)
            return float(np.mean(mu - R[mk, 1] * np.log(mu)))
        b0 = np.array([beta[0], 0, 0, 0])
        out[label] = dict(n_train=int(tr.sum()), n_test=int(te.sum()), beta=dict(zip(["konst", "x_att", "x_reg_vlastne", "x_reg_super"], map(float, beta))),
                          z=dict(zip(["konst", "x_att", "x_reg_vlastne", "x_reg_super"], map(float, beta / se))),
                          test_nll_bez=nll(b0, te), test_nll_s=nll(beta, te), sd_x_att=float(R[pop, 2].std()), sd_x_reg=float(R[pop, 3].std()))
        print(label, json.dumps(out[label], ensure_ascii=False, default=float))
    (E.ROOT / "results" / "lineup_effect.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
