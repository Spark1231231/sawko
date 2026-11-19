"""Úspešnosť modelu spätne na všetkých zápasoch (walk-forward: každá predpoveď len z dát pred zápasom).

Výstup: results/success.json, results/success.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calib as C
import counts as K
import engine as E
import scorers as S
import walk as W

RES = E.ROOT / "results"
SKC = {"uefa.nations": "Liga národov", "fifa.worldq.uefa": "kvalifikácia MS (UEFA)", "uefa.euroq": "kvalifikácia ME",
       "uefa.euro": "ME", "fifa.world": "MS"}


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def tips(P):
    return np.argmax(P, 1)


if __name__ == "__main__":
    w = E.load_world()
    z = np.load(E.ROOT / "data" / "raw" / "cache" / "final_goals.npz")
    rows = z["rows"]
    ok = z["ev"] & np.isfinite(z["ch"])
    d = w.iloc[rows[ok]].reset_index(drop=True)
    P = z["P"][ok]
    ch, ca = z["ch"][ok], z["ca"][ok]
    hg, ag = d.hg.values.astype(int), d.ag.values.astype(int)
    o = C.outcome(hg, ag)
    n = len(d)
    # matica skóre prototypu (rovnaká ako v backteste: Poisson + DC, prenásobená na kalibrované 1/X/2)
    M = C.reweight(E.score_matrix(ch, ca, -0.06), P)
    t = tips(P)
    hit = t == o
    pmax = P.max(1)
    out = {"obdobie": [d.date.min(), d.date.max()], "n": int(n)}

    # ----------------------------------------------------------- 1/X/2
    alw_home = float((o == 0).mean())
    Ps = E.outcome_probs(E.score_matrix(z["s_h"][ok], z["s_a"][ok], -0.06))
    has_s = np.isfinite(z["s_h"][ok])
    Pe = None
    out["tip"] = dict(prototyp=float(hit.mean()), vzdy_domaci=alw_home,
                      web_like=float((tips(Ps[has_s]) == o[has_s]).mean()),
                      tip_remiza_pocet=int((t == 1).sum()), skutocne_remizy=float((o == 1).mean()))
    # podľa istoty
    bins = [0.33, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    cb = pd.cut(pmax, bins)
    by_conf = pd.DataFrame(dict(b=cb, hit=hit, p=pmax)).groupby("b", observed=True).agg(n=("hit", "size"), p=("p", "mean"), hit=("hit", "mean"))
    out["podla_istoty"] = [dict(kos=str(i), n=int(r.n), predpoved=float(r.p), uspesnost=float(r.hit)) for i, r in by_conf.iterrows()]
    yr = d.dt.dt.year.values
    by_year = pd.DataFrame(dict(y=yr, hit=hit)).groupby("y").agg(n=("hit", "size"), hit=("hit", "mean"))
    out["podla_roka"] = {int(i): dict(n=int(r.n), uspesnost=float(r.hit)) for i, r in by_year.iterrows()}
    by_comp = pd.DataFrame(dict(c=d.comp.values, hit=hit)).groupby("c").agg(n=("hit", "size"), hit=("hit", "mean"))
    out["podla_sutaze"] = {SKC.get(i, i): dict(n=int(r.n), uspesnost=float(r.hit)) for i, r in by_comp.iterrows()}

    # ----------------------------------------------------------- trh a finálna zmes (len zápasy so záverečnými kurzmi)
    Pm = z["Pm"][ok]
    hm = np.isfinite(Pm).all(1)
    Pf = Pm[hm] ** 0.85 * P[hm] ** 0.15
    Pf /= Pf.sum(1, keepdims=True)
    out["s_kurzami"] = dict(n=int(hm.sum()), prototyp=float(hit[hm].mean()), trh=float((tips(Pm[hm]) == o[hm]).mean()),
                            finalna_zmes=float((tips(Pf) == o[hm]).mean()),
                            web_like=float((tips(Ps[hm & has_s]) == o[hm & has_s]).mean()))

    # ----------------------------------------------------------- skóre a góly
    flat = M.reshape(n, -1)
    order = np.argsort(-flat, 1)
    hc, ac = np.minimum(hg, E.MAXG - 1), np.minimum(ag, E.MAXG - 1)
    true_idx = hc * E.MAXG + ac
    rank = (order == true_idx[:, None]).argmax(1)
    p_over = (M * (E.TOT > 2.5)).sum((1, 2))
    over = (hg + ag) > 2.5
    p_btts = M[:, 1:, 1:].sum((1, 2))
    btts = (hg > 0) & (ag > 0)
    p_o15 = (M * (E.TOT > 1.5)).sum((1, 2))
    out["skore"] = dict(presne_top1=float((rank == 0).mean()), v_top3=float((rank < 3).mean()), v_top5=float((rank < 5).mean()),
                        top1_predpoved=float(flat.max(1).mean()))
    out["goly"] = dict(
        nad_pod_25=float(((p_over > 0.5) == over).mean()), nad_25_skutocne=float(over.mean()),
        nad_pod_25_iste=dict(n=int(((p_over > 0.62) | (p_over < 0.38)).sum()),
                             uspesnost=float(((p_over > 0.5) == over)[(p_over > 0.62) | (p_over < 0.38)].mean())),
        nad_15=float(((p_o15 > 0.5) == ((hg + ag) > 1.5)).mean()),
        obaja_skoruju=float(((p_btts > 0.5) == btts).mean()), obaja_skutocne=float(btts.mean()),
        priemer_golov_skutocny=float((hg + ag).mean()), priemer_golov_predpoved=float((ch + ca).mean()))
    # silné tipy
    strong = pmax >= 0.70
    out["silne_tipy_70"] = dict(n=int(strong.sum()), uspesnost=float(hit[strong].mean()), predpoved=float(pmax[strong].mean()))
    dc = np.where(t == 0, P[:, 0] + P[:, 1], np.where(t == 2, P[:, 2] + P[:, 1], np.nan))
    dc_hit = np.where(t == 0, o != 2, np.where(t == 2, o != 0, False))
    m_dc = t != 1
    out["dvojita_sanca_favorita"] = dict(n=int(m_dc.sum()), uspesnost=float(dc_hit[m_dc].mean()), predpoved=float(np.nanmean(dc[m_dc])))

    # ----------------------------------------------------------- najnovšie zápasy (14.–17. 11. 2026)
    rec = d.date.values >= "2026-11-14"
    lst = []
    for i in np.flatnonzero(rec):
        lst.append(dict(datum=d.date.values[i], dom=d.home.values[i], host=d.away.values[i], skore=f"{hg[i]}:{ag[i]}",
                        p=[round(float(x), 3) for x in P[i]], tip="1X2"[t[i]], trafene=bool(hit[i]),
                        top_skore=f"{order[i, 0] // E.MAXG}:{order[i, 0] % E.MAXG}"))
    out["najnovsie"] = dict(n=int(rec.sum()), uspesnost=float(hit[rec].mean()) if rec.any() else None, zapasy=lst)

    # ----------------------------------------------------------- karty, strely, rohy (walk-forward modely počtov)
    ze = np.load(E.PROC / "espn_walk.npz")
    cfg = json.loads((RES / "goals_tuning.json").read_text())
    bw = cfg["elo_weight"]
    Dc = K.frame(w, ze["rows"], W.blend(ze["lh"], ze["eh"], bw), W.blend(ze["la"], ze["ea"], bw))
    ev = Dc.comp.isin(E.EVAL_COMPS).values & (Dc.date.values >= "2019-01-01")
    ccfg = json.loads((RES / "counts_cfg.json").read_text())
    out["pocty"] = {}
    for kind, line, name in (("yellow", 3.5, "žlté karty"), ("shots", 24.5, "strely"), ("sot", 8.5, "strely na bránu"), ("corners", 9.5, "rohy")):
        r = K.walk_counts(Dc, kind, ccfg.get(kind))
        T = r["target"][ev[r["target"]]]
        fitm = Dc.date.values[T] < "2024-01-01"
        mh, ma, yh, ya = r["mh"][T], r["ma"][T], r["yh"][T], r["ya"][T]
        tot, pred = yh + ya, mh + ma
        rt = K.mom_r(tot[fitm], pred[fitm])
        cdf = np.array([np.cumsum(K.nb_pmf(m_, rt, 120)) for m_ in pred])
        q10 = (cdf < 0.1).sum(1)
        q90 = (cdf < 0.9).sum(1)
        p_ov = 1 - cdf[:, int(np.floor(line))]
        ov = tot > line
        conf = (p_ov > 0.6) | (p_ov < 0.4)
        out["pocty"][name] = dict(n=int(len(T)), ciara=line, nad_pod=float(((p_ov > 0.5) == ov).mean()),
                                  nad_pod_iste=dict(n=int(conf.sum()), uspesnost=float(((p_ov > 0.5) == ov)[conf].mean())),
                                  vacsina_strana=float(max(ov.mean(), 1 - ov.mean())),
                                  v_80_intervale=float(((tot >= q10) & (tot <= q90)).mean()),
                                  priem_chyba=float(np.mean(np.abs(tot - pred))), priemer=float(tot.mean()))

    # ----------------------------------------------------------- strelci
    lam = {e: (a, b, dd, c) for e, a, b, dd, c in zip(w.espn_id.values[ze["rows"]], W.blend(ze["lh"], ze["eh"], bw),
                                                    W.blend(ze["la"], ze["ea"], bw), w.date.values[ze["rows"]], w.comp.values[ze["rows"]])}
    sp = json.loads((RES / "scorers_backtest.json").read_text())["best"]
    _, df, _ = S.backtest(w, lam, S.load_lineups(), sp, eval_from="2019-01-01", collect=True)
    top = df.loc[df.groupby(["id", "side"]).p.idxmax()]
    big = df[df.p >= 0.30]
    out["strelci"] = dict(n_timov=int(len(top)), najpravdepodobnejsi_skoruje=float(top.y.mean()), jeho_predpoved=float(top.p.mean()),
                          hraci_nad_30=dict(n=int(len(big)), predpoved=float(big.p.mean()), skutocnost=float(big.y.mean())),
                          bezny_hrac_zakladu=float(df.y.mean()))
    (RES / "success.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))

    # ----------------------------------------------------------- Markdown
    tp = out["tip"]
    L = [f"# Úspešnosť modelu spätne ({out['obdobie'][0]} – {out['obdobie'][1][:10]})", "",
         f"Súťažné zápasy európskych reprezentácií (Liga národov, kvalifikácie ME a MS, ME, MS): **{n} zápasov**, každý predpovedaný",
         "walk-forward – model videl len zápasy, ktoré sa odohrali pred ním.", "",
         "## Tip na výsledok (1 / X / 2)", "",
         "| | úspešnosť |", "|---|---|",
         f"| **prototyp** | **{pct(tp['prototyp'])}** |",
         f"| replika modelu webu | {pct(tp['web_like'])} |",
         f"| vždy tipovať domácich | {pct(tp['vzdy_domaci'])} |", "",
         f"Remízu model ako najpravdepodobnejší výsledok tipol {tp['tip_remiza_pocet']}-krát, skutočne skončilo remízou {pct(tp['skutocne_remizy'])} zápasov – "
         "remíza je takmer vždy menej pravdepodobná než výhra favorita, preto tip na ňu zriedka vychádza ako najlepší.", "",
         "**Podľa istoty tipu** (kalibrácia: predpovedaná šanca vs skutočná úspešnosť)", "",
         "| šanca favorita | zápasov | model hovoril | skutočne vyšlo |", "|---|---|---|---|"]
    for r in out["podla_istoty"]:
        lo, hi = (float(x) for x in r["kos"].strip("(]").split(", "))
        L.append(f"| {pct(lo, 0)} – {pct(hi, 0)} | {r['n']} | {pct(r['predpoved'])} | {pct(r['uspesnost'])} |")
    s = out["silne_tipy_70"]
    dcs = out["dvojita_sanca_favorita"]
    L += ["", f"Tipy so šancou aspoň 70 %: {s['n']} zápasov, vyšlo {pct(s['uspesnost'])} (model čakal {pct(s['predpoved'])}).",
          f"Favorit aspoň neprehrá (dvojitá šanca): vyšlo {pct(dcs['uspesnost'])} z {dcs['n']} (model čakal {pct(dcs['predpoved'])}).", "",
          "**Podľa roka**", "", "| rok | zápasov | úspešnosť |", "|---|---|---|"]
    for y, r in out["podla_roka"].items():
        L.append(f"| {y} | {r['n']} | {pct(r['uspesnost'])} |")
    L += ["", "**Podľa súťaže**", "", "| súťaž | zápasov | úspešnosť |", "|---|---|---|"]
    for c, r in sorted(out["podla_sutaze"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {c} | {r['n']} | {pct(r['uspesnost'])} |")
    k = out["s_kurzami"]
    L += ["", f"**Zápasy so záverečnými kurzmi** ({k['n']}): prototyp {pct(k['prototyp'])} · trh (stávkové kancelárie) {pct(k['trh'])} · "
          f"finálna zmes 85 % trh + 15 % model {pct(k['finalna_zmes'])} · replika webu {pct(k['web_like'])}.", "",
          "## Skóre a góly", "", "| tip | úspešnosť |", "|---|---|"]
    sk, g = out["skore"], out["goly"]
    L += [f"| presné skóre (najpravdepodobnejšie) | {pct(sk['presne_top1'])} (model čakal {pct(sk['top1_predpoved'])}) |",
          f"| skutočné skóre medzi 3 najpravdepodobnejšími | {pct(sk['v_top3'])} |",
          f"| skutočné skóre medzi 5 najpravdepodobnejšími | {pct(sk['v_top5'])} |",
          f"| nad / pod 2,5 gólu | {pct(g['nad_pod_25'])} (vždy „nad“ by dalo {pct(max(g['nad_25_skutocne'], 1 - g['nad_25_skutocne']))}) |",
          f"| nad / pod 2,5, len isté tipy (pod 38 % alebo nad 62 %, {g['nad_pod_25_iste']['n']} zápasov) | {pct(g['nad_pod_25_iste']['uspesnost'])} |",
          f"| nad / pod 1,5 gólu | {pct(g['nad_15'])} |",
          f"| obaja skórujú áno / nie | {pct(g['obaja_skoruju'])} (vždy „nie“ by dalo {pct(max(g['obaja_skutocne'], 1 - g['obaja_skutocne']))}) |",
          f"| priemer gólov na zápas | predpoveď {g['priemer_golov_predpoved']:.2f} · skutočnosť {g['priemer_golov_skutocny']:.2f} |".replace(".", ","),
          "", "## Karty, strely, rohy", "",
          "| | zápasov | nad / pod | isté tipy (šanca < 40 % alebo > 60 %) | vždy väčšinová strana | v 80 % intervale | priem. chyba |",
          "|---|---|---|---|---|---|---|"]
    for name, r in out["pocty"].items():
        L.append(f"| {name} (čiara {str(r['ciara']).replace('.', ',')}) | {r['n']} | {pct(r['nad_pod'])} | {pct(r['nad_pod_iste']['uspesnost'])} ({r['nad_pod_iste']['n']}) | "
                 f"{pct(r['vacsina_strana'])} | {pct(r['v_80_intervale'])} | {str(round(r['priem_chyba'], 2)).replace('.', ',')} |")
    st = out["strelci"]
    L += ["", "## Strelci", "",
          f"- najpravdepodobnejší strelec tímu skóroval v **{pct(st['najpravdepodobnejsi_skoruje'])}** prípadov (model čakal {pct(st['jeho_predpoved'])}; {st['n_timov']} tímov v zápasoch so známou zostavou);",
          f"- hráči, ktorým model dal aspoň 30 %: {st['hraci_nad_30']['n']}, model čakal {pct(st['hraci_nad_30']['predpoved'])}, skórovalo {pct(st['hraci_nad_30']['skutocnost'])};",
          f"- bežný hráč základnej jedenástky skóruje v {pct(st['bezny_hrac_zakladu'])} zápasov.", "",
          f"## Najnovšie zápasy 14.–17. 11. 2026 ({out['najnovsie']['n']}, tip vyšiel v {pct(out['najnovsie']['uspesnost'])})", "",
          "| dátum | zápas | 1 / X / 2 | tip | výsledok | |", "|---|---|---|---|---|---|"]
    for r in out["najnovsie"]["zapasy"]:
        L.append(f"| {r['datum'][5:10]} | {r['dom']} – {r['host']} | {' / '.join(pct(x, 0) for x in r['p'])} | {r['tip']} | {r['skore']} | {'✓' if r['trafene'] else '✗'} |")
    (RES / "success.md").write_text("\n".join(L))
    print("\n".join(L))
