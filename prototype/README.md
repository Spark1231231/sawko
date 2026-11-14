# Prototyp predpovede zápasov

Prototyp spoľahlivejšieho predikčného mechanizmu pre web **Stavková pravdepodobnosť**
(`stavkova-pravdepodobnost.vercel.app`). Vznikol v troch krokoch:

1. rozobratie toho, ako pravdepodobnosti počíta web,
2. iteračná slučka „hypotéza → walk-forward backtest → prijať / zamietnuť“,
3. ostrá predpoveď dnešného zápasu (víťaz, skóre, karty, strely, strelci).

Všetko je v Pythone (numpy/pandas/scipy), reprezentačný futbal, dáta z rovnakého zdroja ako web (verejný
feed ESPN) + svetová databáza medzištátnych výsledkov.

---

## 1. Ako to počíta web (stav 14. 11. 2026)

Zistené zo zdrojákov nasadenia (`lib/model/*`, `scripts/*`) a z dát, ktoré stránka zápasu posiela prehliadaču.

| krok | čo web robí |
|---|---|
| dáta | ESPN (zápasy, štatistiky, karty, zostavy) od 2016; kluby top 5 líg + európske poháre (24 028 zápasov), reprezentácie: Liga národov, kvalifikácie, ME, MS, prípravné (3 356) |
| sila tímov | Poissonova regresia: útok, obrana, vlastná domáca výhoda tímu; normálne priory zvlášť na útok a obranu; exponenciálny časový útlm; góly zmiešané so strelami na bránu a xG (`GOAL_MIX`) |
| kalibrácia | „stretch“: rozdiel síl sa dodatočne násobí `k = 1,26` (kluby) a `k = 1,46` (reprezentácie), `g` určuje, čo sa stane so súčtom gólov |
| skóre | nezávislé Poissony + Dixon–Coles `ρ = −0,06` (európske poháre od 2024/25 `+0,06`), mriežka do 10 gólov |
| úpravy | vzájomné zápasy (H2H, strop ±10 %), marodka (útok × max(0,72; 1 − 0,55·stratený podiel)), počasie, rozhodca (karty) |
| trh | λ spätne odvodené z kurzov bez marže; geometrický mix **85 % trh + 15 % model** |
| počty | rohy, karty, strely: ten istý fit útok/obrana, záporné binomické rozdelenie |
| hráči | góly/90 z klubu aj reprezentácie, forma, minúty, dostupnosť |

Vlastné backtesty webu (2022-08 … 2026-11, 6 772 zápasov): samotný model proti kurzom Fortuny prerába
(ROI −9,6 % až −12,8 %, záporné CLV) a web to aj priznáva („model sám trh neporazí“). Pri počtoch je model
len tesne nad konštantou: rohy 9,5 log-loss 0,6855 vs 0,6911, karty 4,5 0,6692 vs 0,6835.

### Slabé miesta, ktoré sa dali zmerať

1. **Stretch 1,46 je príznak, nie liek.** Model je príliš zmrštený k priemeru a potom sa plošne roztiahne.
   Pri replike architektúry webu vyjde optimum roztiahnutia 1,46 presne vtedy, keď je prior na útok/obranu
   okolo σ ≈ 0,45; s užším priorom je model výrazne horší (log-loss 0,83–0,88).
2. **Jeden prior na dve rôzne veci.** Výsledok 1/X/2 určuje rozdiel *síl*, počet gólov *štýl*. Každé potrebuje
   iné zmrštenie (sila takmer žiadne, štýl silné).
3. **Favorit je systematicky podhodnotený** – outsider strieľa menej gólov, než hovorí log-lineárny model, a remíz
   je pri nerovných súperoch menej, než dávajú nezávislé Poissony. Plošný stretch to opraví len sčasti.
4. **H2H faktor je šum.** Odchýlky v predchádzajúcich vzájomných zápasoch nepredpovedajú odchýlku v ďalšom:
   r = +0,007 (súčet gólov) a +0,003 (gólový rozdiel), n = 8 043, intervaly obsahujú nulu
   (`results/h2h_test.json`). Pri Portugalsko – Dánsko pritom web kvôli 3 zápasom zvýšil oba útoky o 10 %.
5. **Počty bez „papierovej sily“.** Fit útok/obrana pri strelách a kartách nevie, že favorit strieľa viac a outsider
   fauluje viac – práve tam je väčšina predpovedateľnosti.
6. **Úzke dáta.** Len zápasy z ESPN za 6 rokov; svetová databáza (50 285 zápasov od 1872) dá stabilnejšie Elo.

---

## 2. Matematika prototypu

### 2.1 Góly ako Poissonov proces

Počet gólov tímu v zápase je `K ~ Poisson(λ)`, `P(K=k) = e^{−λ} λ^k / k!`. Intenzity:

```
log λ_dom  = μ + γ·[hrá doma] + att[dom]  − def[host]
log λ_host = μ                + att[host] − def[dom]
```

Odhad je **MAP** (maximum vierohodnosti s priorom) na všetkých medzištátnych zápasoch pred dňom predpovede:

```
L = Σ_z w_z · [ g_dom·log λ_dom − λ_dom + g_host·log λ_host − λ_host ]  −  Σ_tím ( s² / 2σ_s² + t² / 2σ_t² )

s = (att + def)/2   celková SILA tímu         σ_s = 1,5   (takmer voľná)
t = (att − def)/2   gólový ŠTÝL tímu          σ_t = 0,3   (silno zmrštený)
w_z = dôležitosť(súťaž) · 2^(−vek_z / 650 dní)       prípravný zápas 0,4, menší turnaj 0,8, ostatné 1,0
```

Cieľ `g` nie je čistý počet gólov: `g = 0,5·góly + 0,5·(strely na bránu × konverzia)` tam, kde štatistika
existuje (tlmí náhodu jednotlivých gólov), debakle sú orezané na 8.

Riešenie je blokový Newton – v bloku „útoky“ (resp. „obrany“) je Hessián diagonálny, lebo v každom λ
vystupuje len jeden útok a jedna obrana, takže krok `Δ = gradient / hessián` je presný. Jeden fit ≈ 10 ms.

### 2.2 Elo

`R' = R + K·G·(W − W_e)`, `W_e = 1/(1 + 10^(−(R_dom − R_host + 50)/400))`, `K` = 17/25,5/34/47 podľa súťaže,
`G` = násobok za gólový rozdiel (1; 1,5; 1,75 + (n−3)/8). Rozdiel ratingov → góly cez Poissonovu regresiu
`log λ = a + b·Δ + c·Δ² + d·[doma] + e·[vonku]`, `Δ = (R_vlastný − R_súper)/400`.

### 2.3 Stacking a kalibrácia (učené len z minulosti)

Z out-of-sample predpovedí príbuzných zápasov (súťažné zápasy zóny UEFA + MS/ME, 3 400+ zápasov) sa každý
rok nanovo naučí:

```
stack:  log λ* = a + b·log λ_sila + c·log λ_sila,súper + d·log λ_elo + e·log λ_elo,súper + f·[doma]
        (aktuálne  b = 0,94  c = −0,17  d = 0,38  e = 0,35)

recal:  log(p1/px)* = α1 + β1·log(p1/px)        β1 = 1,11
        log(p2/px)* = α2 + β2·log(p2/px)        β2 = 1,24
```

`recal` je zovšeobecnené Plattovo škálovanie – robí to isté čo stretch webu, ale je naučené na správnej
populácii, zvlášť pre domácich a hostí, a nezasahuje do očakávaných gólov.

### 2.4 Matica skóre

`P(i, j) = Pois(i; λ_dom) · Pois(j; λ_host) · τ_ρ(i, j)` s Dixon–Colesovou korekciou `ρ = −0,06` (vyšla rovnako
ako na webe). Po rekalibrácii sa oblasti matice (výhra domácich / remíza / výhra hostí) prenásobia tak, aby
sedeli s 1/X/2 – je to riešenie s najmenšou KL-divergenciou od pôvodnej matice, tvar rozdelenia vnútri
oblasti ostáva.

### 2.5 Trh

Kurzy → pravdepodobnosti mocninovou metódou (`p_i = q_i^k`, `Σ p_i = 1`), z nich a z over/under spätne
`(λ_dom, λ_host)`. Finálna predpoveď: `λ = λ_trh^0,85 · λ_model^0,15`, `P ∝ P_trh^0,85 · P_model^0,15`.

### 2.6 Počty (strely, strely na bránu, rohy, žlté karty, fauly)

```
E[počet tímu] = exp( x·β + A[tím] − D[súper] )
x = [1, doma, vonku, s, s², prípravný, turnaj],   s = log(λ_vlastné / λ_súper)
```

`x·β` je „papierová sila“ (Poissonova regresia, učená po rokoch), `A`, `D` sú tímové odchýlky (rovnaký MAP
fit ako pri góloch, s časovým útlmom). Rozptyl: záporné binomické rozdelenie `Var = μ + μ²/r`
(gamma-Poissonova zmes), `r` z minulých out-of-sample odchýlok – zvlášť pre tím a pre súčet zápasu.
Karty navyše násobí efekt rozhodcu `(skutočné + 80)/(predpovedané + 80)` z jeho minulých zápasov.

### 2.7 Strelci

Riedenie Poissonovho procesu: každý gól tímu pripadne hráčovi `i` s pravdepodobnosťou `θ_i`, takže

```
góly hráča ~ Poisson(λ_tím · θ_i · m_i),      P(skóruje) = 1 − exp(−λ_tím · θ_i · m_i)
θ_i = (G_i + κ·θ_post) / (X_i + κ),           X_i = Σ λ_tím·m (očakávané góly tímu počas jeho minút), κ = 25
```

`θ_post` je priemer postu (11 skupín, krídla zvlášť), 40 % váhy má podiel hráča na strelách na bránu.
Podiely základnej jedenástky sa normujú na obvyklý podiel gólov, ktoré dá základ (≈ 78 %). V ostrej
predpovedi sa kombinujú tri zdroje (geometricky): reprezentačná história 25 %, klubové góly/90 z webu 25 %,
kurzy na strelca 50 %.

---

## 3. Iteračná slučka

Pravidlá: každá predpoveď je walk-forward (model vidí len zápasy pred daným dňom); nastavenia sa vyberajú na
období **ladenie** (2018-08 … 2023-12, 1 324 súťažných zápasov), obdobie **test** (2024-01 … 2026-11, 674
zápasov) slúži len na záverečné porovnanie. Metrika: log-loss 1/X/2 (nižšie = lepšie; konštantný tip 1,056).

| # | hypotéza | výsledok (ladenie) | verdikt |
|---|---|---|---|
| 1 | Elo → góly ako východisko | 0,8353 | základ |
| 2 | Poissonov MAP na svetových dátach (σ = 0,5) | 0,8293 | lepšie |
| 3 | menej zmrštenia (σ 0,25 → 1,5) | 0,8567 → 0,8220 | **kľúčové** – web je prezmrštený |
| 4 | oddelený prior sila / štýl (úzky prior na štýl) | 1/X/2 bez zmeny, over/under 0,6685 → 0,6662, presné skóre 2,7540 → 2,7513 | prijaté |
| 5 | mix gólov so strelami na bránu | presné skóre 2,7526 → 2,7472 | prijaté |
| 6 | kratší polčas útlmu (650 dní) | 0,8195 | prijaté |
| 7 | diagnóza: favorit podhodnotený (stretch 1,15 pomáha) | 0,8195 → 0,8133 | vedie k #8 |
| 8 | stacking sila + Elo a rekalibrácia 1/X/2 učená z minulosti | **0,8153** (test **0,8049**) | prijaté |
| 9 | kvadratické členy rozdielu síl v stackingu | 0,8151 | zamietnuté (bez zisku) |
| 10 | kalibrácia učená na celom svete namiesto zóny UEFA | 0,8223 | zamietnuté |
| 11 | H2H ako signál | r = 0,007 | zamietnuté |
| 12 | počty: pridať regresiu na papierovú silu | karty 3,5: 0,6957 → 0,6579 | prijaté |
| 13 | karty: efekt rozhodcu (κ = 80) | 2,1417 → 2,1348 | prijaté (malý zisk) |
| 14 | strelci: góly z udalostí zápasu namiesto hráčskych štatistík | pri predpovedi 24 % sa trafilo 32 % → po oprave 26 % | **oprava chyby v dátach** |
| 15 | strelci: jemnejšie posty + empirický Bayes | 0,2542 → 0,2497 | prijaté |
| 16 | váha trhu 0,5 / 0,7 / 0,85 | 0,85–1,0 najlepšie | potvrdené 0,85 |
| 17 | karty: predpovedaná „faulovosť“ tímov ako ďalší príznak | súčet NLL 2,0945 → 2,0905, ale test 2,1300 → 2,1322 | zamietnuté (nekonzistentné) |
| 18 | góly: zloženie základnej jedenástky (koľko streleckej sily a stálych hráčov nastúpi) | koeficienty nevýznamné (z < 1,8), test NLL horší | zamietnuté – úpravu „marodka“ sa na reprezentačných dátach nepodarilo potvrdiť |
| 19 | kalibrácia podľa populácie: dva európske tímy „uefa“, inak „world“ | mimo Európy sklony ≈ 1,0 a 1,1 (favorit tam podhodnotený nie je) | prijaté |

---

## 4. Výsledky backtestu

### 4.1 Výsledok zápasu (log-loss 1/X/2, walk-forward)

| model | ladenie (1 324) | test (674) | presnosť tipu (test) | RPS (test) |
|---|---|---|---|---|
| konštantný tip | 1,0563 | 1,0638 | 44 % | – |
| Elo | 0,8333 | 0,8257 | 62,5 % | 0,1580 |
| replika prístupu webu (ESPN 6 r., nezávislé priory, stretch 1,46) | 0,8240 | 0,8102 | 62,9 % | 0,1527 |
| **prototyp (len model)** | **0,8153** | **0,8049** | **64,2 %** | **0,1508** |

Párový rozdiel prototyp − replika webu na všetkých 1 998 zápasoch: **−0,0075**, 95 % bootstrap interval
[−0,0138; −0,0010] – zlepšenie je štatisticky významné, ale malé. Replika je moja rekonštrukcia architektúry
webu s najpriaznivejším nastavením, nie jeho presný kód.

Kalibrácia prototypu (všetky tri výsledky, 1 998 zápasov): keď model povie 55 %, stane sa to v 57 %;
65 % → 67 %; 75 % → 78 %; 85 % → 84 %; 95 % → 96 %.

### 4.2 Proti trhu (zápasy so záverečnými kurzmi DraftKings z ESPN)

| | ladenie (261) | test (574) |
|---|---|---|
| trh (kurzy bez marže) | **0,7480** | 0,7643 |
| prototyp (len model) | 0,7658 | 0,7815 |
| replika webu (len model) | 0,7746 | 0,7900 |
| 85 % trh + 15 % prototyp | 0,7489 | **0,7639** |

Trh je stále lepší než akýkoľvek model z výsledkov (vie o zostavách, zraneniach, forme hráčov v kluboch).
Prototyp zmazal asi tretinu odstupu modelu od trhu (0,0257 → 0,0172). Mix s trhom je na úrovni samotného trhu –
preto je finálna predpoveď 85 % trh.

### 4.3 Priame porovnanie s modelom webu (33 nadchádzajúcich zápasov, 14. 11. 2026)

Vzdialenosť „len modelu“ od trhu na tých istých zápasoch (`results/site_comparison.json`):

| | web | prototyp |
|---|---|---|
| KL-divergencia od trhu (1/X/2) | 0,0221 | **0,0170** |
| priem. chyba λ voči trhu | 0,239 | **0,154** |
| priem. súčet gólov (trh 2,79) | 2,52 | 2,66 |
| bližšie k trhu v zápasoch | 11 | **22** |

### 4.4 Počty (test 2024–2026, súčet za zápas; model / najlepšia možná konštanta)

| | NLL súčtu | over/under log-loss | MAE | len útok/obrana (ako web) |
|---|---|---|---|---|
| žlté karty | 2,130 / 2,181 | 3,5: **0,664** / 0,693 · 4,5: **0,619** / 0,649 | 1,70 / 1,78 | 3,5: 0,687 · 4,5: 0,648 |
| strely | 3,223 / 3,243 | 24,5: **0,683** / 0,693 | 4,86 / 5,02 | 24,5: 0,711 (horšie než konštanta) |
| strely na bránu | 2,541 / 2,572 | 8,5: **0,673** / 0,693 | 2,48 / 2,57 | 8,5: 0,688 |
| rohy | 2,593 / 2,617 | 9,5: **0,673** / 0,688 | 2,63 / 2,72 | 9,5: 0,689 |

Počty sú z princípu málo predpovedateľné (korelácia predpovede so skutočnosťou 0,24–0,31), ale prototyp má
proti konštante zisk 1,5–4,5 %, kým samotný fit útok/obrana 0–1 %.

### 4.5 Strelci (hráči základných jedenástok, súťažné zápasy 2023–2026, n = 19 877)

Log-loss „hráč skóruje“: **0,2497** vs 0,2542 (len post). Najpravdepodobnejší strelec tímu skóruje v 27 %
zápasov (model hovorí 26 %). Reprezentačná história sama veľa nepridá – preto ostrá predpoveď pridáva klubové
čísla z webu a kurzy.

---

## 5. Čo z toho preniesť do webu

1. Prior na **silu a štýl** namiesto útoku a obrany; silu takmer nezmršťovať.
2. Plošný stretch nahradiť **rekalibráciou 1/X/2** učenou rolling-origin na príbuzných zápasoch
   (`calib.py`, 4 čísla).
3. Pridať **svetové výsledky a Elo** ako druhý vstup (stacking).
4. **Vypnúť H2H faktor** – nenesie informáciu.
5. Pri počtoch pridať **regresiu na rozdiel síl** (`counts.py`) – najväčší relatívny zisk.
6. Strelcov počítať riedením s normovaním podielov a **góly brať z udalostí zápasu**.
7. Váhu trhu 85 % ponechať – backtest ju potvrdil aj pre reprezentácie.

Limity: reprezentácie hrajú ~10 zápasov ročne, takže neistota sily je veľká; model nevie o zostavách ani
zraneniach (to nesie trh); klubový futbal tu testovaný nebol.

Mimo Európy (všetky súťažné zápasy sveta od 2018-08, n = 5 231, bez kalibrácie): Elo 0,8367, model sily 0,8281,
mix 0,8253 (konštanta 1,0512) – jadro modelu teda funguje aj mimo zóny UEFA.

---

## 6. Spustenie

```bash
pip install numpy pandas scipy requests
python3 fetch_data.py all          # ESPN + svetové výsledky (≈ 5 min, treba prístup na site.api.espn.com)
python3 build_dataset.py           # data/processed/*.csv.gz
python3 tune.py                    # ladenie modelu sily a Elo
python3 wide_walk.py && python3 espn_walk.py   # walk-forward λ pre kalibráciu a počty
python3 final_goals.py             # záverečný backtest gólov + results/calibration.json
python3 counts_eval.py             # backtest počtov
python3 scorers_eval.py            # backtest strelcov
python3 site_snapshot.py           # čo práve predpovedá web (λ, kurzy, zostavy)
python3 compare_site.py            # prototyp vs web vs trh
python3 predict.py 401861167 && python3 report.py 401861167     # predpoveď zápasu
python3 today.py 2026-11-14        # všetky zápasy dňa (+ `check` po zápasoch)
python3 predict.py 401929710 --comp=concacaf.nations.league    # zápas, ktorý web nepokrýva (vstupy priamo z ESPN)
```

`data/processed/` obsahuje aj hotové walk-forward λ (`wide_walk.npz`, `espn_walk.npz`) a snapshot webu, takže
`predict.py` beží z čistého klonu bez prepočítavania celej histórie.

| súbor | obsah |
|---|---|
| `engine.py` | Elo, Poissonov MAP fit, matica skóre, metriky, odstránenie marže |
| `walk.py` | walk-forward predpovede |
| `calib.py` | stacking, rekalibrácia, prenásobenie matice |
| `counts.py` | modely počtov (NB) |
| `scorers.py` | model strelcov |
| `predict.py`, `report.py`, `today.py` | ostré predpovede |
| `results/` | výsledky backtestov a predpovede |
