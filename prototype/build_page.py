"""Zostaví jednoduchú statickú stránku s predpoveďami: site/index.html (jeden súbor, bez závislostí).

    python3 build_page.py 2026-11-14 [ďalšie id zápasov mimo webu, napr. 401929710]
"""
from __future__ import annotations

import html
import json
import sys

import pandas as pd

import engine as E
from report import SK, num, pct

RES = E.ROOT / "results"
OUT = E.ROOT / "site"

CSS = """
:root{--bg:#f6f5f1;--card:#fff;--ink:#17191c;--mut:#687078;--line:#e2dfd7;--acc:#b7791f;--good:#1f7a4d;--bad:#b3382c;--bar1:#d99a2b;--barx:#a9adb3;--bar2:#3f4750}
@media (prefers-color-scheme:dark){:root{--bg:#121416;--card:#1b1e21;--ink:#eceae4;--mut:#99a1a8;--line:#2c3035;--acc:#e0a94a;--good:#58c08a;--bad:#ee7c6e;--bar1:#e0a94a;--barx:#6d747b;--bar2:#c9cdd2}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:0}
.sub{color:var(--mut);font-size:13px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 14px}
.head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
.meta{color:var(--mut);font-size:13px}
.tip{margin:10px 0 6px;font-size:16px}.tip b{color:var(--acc)}
.bar{display:flex;height:26px;border-radius:7px;overflow:hidden;font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
.bar span{display:flex;align-items:center;justify-content:center;white-space:nowrap;min-width:0;overflow:hidden}
.b1{background:var(--bar1);color:#1b1500}.bx{background:var(--barx);color:#111}.b2{background:var(--bar2);color:var(--bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.kpi{border:1px solid var(--line);border-radius:9px;padding:9px 11px}
.kpi small{display:block;color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.kpi b{font-size:18px;font-variant-numeric:tabular-nums}.kpi span{display:block;color:var(--mut);font-size:12.5px}
.real{margin-top:12px;padding:9px 11px;border-radius:9px;border:1px dashed var(--line);font-size:14px}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
details{margin-top:10px}summary{cursor:pointer;color:var(--mut);font-size:13px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:8px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);font-weight:500;font-size:12px}
.note{color:var(--mut);font-size:13px}
"""


def e(x) -> str:
    return html.escape(str(x))


def name(t: str) -> str:
    return SK.get(t, t)


def outcome_block(act: dict | None, f: dict, y: dict, sh: dict, top: list) -> str:
    if not act or act.get("state") not in ("post", "in"):
        return ""
    hg, ag = act["hg"], act["ag"]
    live = act["state"] == "in"
    o = 0 if hg > ag else 1 if hg == ag else 2
    tip = max(range(3), key=lambda i: [f["p1"], f["px"], f["p2"]][i])
    bits = [f"<b>{'Priebežne' if live else 'Skutočnosť'}: {hg}:{ag}</b>"]
    if not live:
        bits.append(f"tip <span class='{'ok' if tip == o else 'no'}'>{'trafený' if tip == o else 'netrafený'}</span>")
        bits.append("skóre " + ("<span class='ok'>presne</span>" if (f["scores"][0][0], f["scores"][0][1]) == (hg, ag) else "<span class='no'>iné</span>"))
    if act.get("hy") is not None and act.get("hsh") is not None:
        yy, ss = int(act["hy"] + act["ay"]), int(act["hsh"] + act["ash"])
        cls = lambda v, c: "ok" if c["q10"] <= v <= c["q90"] else "no"
        bits.append(f"žlté <span class='{cls(yy, y) if not live else ''}'>{yy}</span> (čakané {num(y['total'])})")
        bits.append(f"strely <span class='{cls(ss, sh) if not live else ''}'>{ss}</span> (čakané {num(sh['total'])})")
    sc = sorted({n for _, n, _ in act.get("scorers", []) if n})
    if sc:
        bits.append("strelci: " + e(", ".join(sc)))
    return f"<div class='real'>{' · '.join(bits)}</div>"


def card(r: dict, act: dict | None, when: str | None) -> str:
    f, m, c = r["final"], r["model"], r["counts"]
    h, a = name(r["home"]), name(r["away"])
    probs = [f["p1"], f["px"], f["p2"]]
    fav = max(zip(probs, (h, "remíza", a)))
    y, sh, sot, cor = c["yellow"], c["shots"], c["sot"], c["corners"]
    top = [(p["name"], p["p_score"]) for s in ("h", "a") for p in r["scorers"].get(s, {}).get("players", [])[:1]]
    bar = "".join(f"<span class='{cl}' style='flex:{max(p, 0.04):.3f}'>{lab} {pct(p)}</span>" for p, cl, lab in zip(probs, ("b1", "bx", "b2"), ("1", "X", "2")))
    scores = " · ".join(f"{i}:{j} {pct(p, 1)}" for i, j, p in f["scores"][:5])
    rows = ""
    for s in ("h", "a"):
        sc = r["scorers"].get(s)
        if not sc:
            continue
        for p in sc["players"][:6]:
            if p["p_score"] < 0.02:
                continue
            rows += f"<tr><td>{e(p['name'])}</td><td>{name(sc['team'])}</td><td><b>{pct(p['p_score'])}</b></td><td>{pct(p['p_2plus'], 1)}</td><td>{p.get('odds') or '–'}</td></tr>"
    src = f"<tr><td>len model</td><td>{pct(m['p1'])}</td><td>{pct(m['px'])}</td><td>{pct(m['p2'])}</td><td>{num(m['lam'][0], 2)} : {num(m['lam'][1], 2)}</td></tr>"
    if r.get("market"):
        k = r["market"]
        src += f"<tr><td>trh (kurzy bez marže)</td><td>{pct(k['p1'])}</td><td>{pct(k['px'])}</td><td>{pct(k['p2'])}</td><td>{num(k['lam'][0], 2)} : {num(k['lam'][1], 2)}</td></tr>"
    if r.get("site"):
        s_ = [x for x in r["site"]["factors"] if x["key"] != "market"]
        if s_:
            src += f"<tr><td>web – len model</td><td>{pct(s_[-1]['p1'])}</td><td>{pct(s_[-1]['px'])}</td><td>{pct(s_[-1]['p2'])}</td><td>{num(r['site']['lam_model'][0], 2)} : {num(r['site']['lam_model'][1], 2)}</td></tr>"
    lineup = "potvrdené zostavy" if r.get("lineup_confirmed") else "predpokladané zostavy"
    stamp = f" · vypočítané {e(when)} UTC" if when else ""
    return f"""<section class="card">
<div class="head"><h2>{e(h)} – {e(a)}</h2><span class="meta">{e(r.get('comp') or '')} · výkop {e(r['date'][11:16])} UTC{stamp}</span></div>
<p class="tip">Tip: <b>{e(fav[1])} ({pct(fav[0])})</b> · najpravdepodobnejšie skóre <b>{f['scores'][0][0]}:{f['scores'][0][1]}</b></p>
<div class="bar">{bar}</div>
<div class="grid">
<div class="kpi"><small>Očakávané góly</small><b>{num(f['lam'][0], 2)} : {num(f['lam'][1], 2)}</b><span>nad 2,5: {pct(f['over25'])} · obaja: {pct(f['btts'])}</span></div>
<div class="kpi"><small>Žlté karty</small><b>{num(y['total'])}</b><span>{num(y['home'])} + {num(y['away'])} · interval {y['q10']}–{y['q90']}</span></div>
<div class="kpi"><small>Strely</small><b>{num(sh['total'])}</b><span>{num(sh['home'])} + {num(sh['away'])} · interval {sh['q10']}–{sh['q90']}</span></div>
<div class="kpi"><small>Strely na bránu</small><b>{num(sot['total'])}</b><span>{num(sot['home'])} + {num(sot['away'])}</span></div>
<div class="kpi"><small>Rohy</small><b>{num(cor['total'])}</b><span>{num(cor['home'])} + {num(cor['away'])}</span></div>
<div class="kpi"><small>Najpravdep. strelci</small><b style="font-size:14px">{e('; '.join(f'{n} {pct(p)}' for n, p in top)) or '–'}</b><span>{lineup}</span></div>
</div>
{outcome_block(act, f, y, sh, top)}
<details><summary>Detail: skóre, zdroje, strelci</summary>
<p class="note">Skóre: {scores}</p>
<div class="tw"><table><tr><th>zdroj</th><th>1</th><th>X</th><th>2</th><th>oč. góly</th></tr>{src}
<tr><td><b>finálna ({pct(f['market_weight'])} trh)</b></td><td><b>{pct(f['p1'])}</b></td><td><b>{pct(f['px'])}</b></td><td><b>{pct(f['p2'])}</b></td><td><b>{num(f['lam'][0], 2)} : {num(f['lam'][1], 2)}</b></td></tr></table></div>
<div class="tw"><table><tr><th>hráč</th><th>tím</th><th>gól</th><th>2+ góly</th><th>kurz</th></tr>{rows}</table></div>
<p class="note">Žlté karty nad 2,5: {pct(y['over']['2.5'])} · nad 3,5: {pct(y['over']['3.5'])} · nad 4,5: {pct(y['over']['4.5'])} · červená karta v zápase: {pct(c['red']['p_any'])}</p>
</details></section>"""


if __name__ == "__main__":
    day = sys.argv[1]
    extra = sys.argv[2:]
    today = json.loads((RES / f"today_{day}.json").read_text())
    cards, done = [], []
    for eid in extra:
        r = json.loads((RES / f"prediction_{eid}.json").read_text())
        act = None
        f = RES / f"actual_{eid}.json"
        if f.exists():
            act = json.loads(f.read_text())
        cards.append((r["date"], card(r, act, (r.get("generated_utc") or "")[11:] or None)))
    for t in today:
        r = json.loads((RES / f"prediction_{t['id']}.json").read_text())
        cards.append((r["date"], card(r, t.get("actual"), "16:04")))
        if (t.get("actual") or {}).get("state") == "post":
            done.append(t)
    cards.sort(key=lambda x: x[0], reverse=True)
    summary = ""
    if done:
        hit = sum(max(range(3), key=lambda i: [t["final"]["p1"], t["final"]["px"], t["final"]["p2"]][i]) ==
                  (0 if t["actual"]["hg"] > t["actual"]["ag"] else 1 if t["actual"]["hg"] == t["actual"]["ag"] else 2) for t in done)
        summary = (f"<section class='card'><b>Priebežné vyhodnotenie:</b> tip na výsledok trafený v {hit} z {len(done)} dohraných zápasov. "
                   f"<span class='note'>Pár zápasov nič nedokazuje – spoľahlivosť ukazuje backtest na 1 998 zápasoch (log-loss 0,805 na teste, trh 0,764).</span></section>")
    now = pd.Timestamp.now("UTC").strftime("%d. %m. %Y %H:%M")
    page = f"""<!doctype html><html lang="sk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>Predpovede zápasov – prototyp</title><style>{CSS}</style></head><body><main>
<h1>Predpovede zápasov</h1>
<p class="sub">Prototyp: 85 % trh + 15 % vlastný model (sila tímov + Elo, kalibrované). Stránka zostavená {now} UTC. Štatistický odhad, nie istota.</p>
{summary}{''.join(c for _, c in cards)}
<p class="note">Karty, strely a rohy sú stredné hodnoty so 80 % intervalom; strelci = šanca aspoň jedného gólu. Predpovede zápasov Ligy národov boli vypočítané 14. 11. o 16:04 UTC, pred výkopmi.</p>
</main></body></html>"""
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(page)
    print("site/index.html", len(page), "B,", len(cards), "zápasov")
