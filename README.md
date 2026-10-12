# Proof

**A bakery production planner that decides how much of each item to bake each morning, and when to discount what's left.**

Built in one night for the CODE 2026 hackathon (Bakery Production Planner brief). It runs on 600 trading days of real transaction data from a French bakery, replays two 30-day windows, and measures the result in rupees.

| Window | `weekday_avg` baseline | **Proof** | Difference |
|---|---|---|---|
| Apr 27 – May 27, 2022 (a normal month) | ₹250,315 | **₹279,694** | **+₹29,379** |
| Aug 30 – Sep 30, 2022 (season turns) | ₹183,614 | ₹274,292 | +₹90,678 |

The spring window is the defensible number. The September gain is mostly the baseline over-baking after the tourist season ended, so it is reported as a stress case, not the headline.

---

## The idea

Two things drive everything.

**1. The average forecast is the wrong number to bake.**
Running out of a croissant costs ₹60 of lost margin. One unsold croissant costs ₹8 after markdown. The right amount to bake is a high quantile of the demand distribution, not the mean. This is the newsvendor critical fractile:

```
Cu   = price - unit_cost           margin lost when you sell out
Co   = unit_cost - salvage_value   loss on one unsold unit
tau* = Cu / (Cu + Co)
q*   = forecast_quantile(item, date, tau*)
```

| Item | Price | Cost | Running out costs | One unsold costs | Bake at |
|---|---|---|---|---|---|
| Croissant | ₹90 | ₹30 | ₹60 | ₹8 | ~88th percentile |
| Traditional baguette | ₹45 | ₹23 | ₹22 | ₹23 | ~49th percentile |

Pastry is deliberately over-baked and bread is not. That is why the model forecasts nine quantiles, not one number.

**2. Sales data is censored.**
If croissants sell out at 11:20, the register understates demand. Training on sales teaches the model to bake less, which causes more sell-outs. Proof detects sell-outs from gaps in the transaction log and reconstructs the demand that was never recorded, before any model is trained.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  EDGE         src/agents/override.py   (optional LLM)   │
│  Turns "kal 20 croissant ka order hai" into structured  │
│  JSON. A human approves before any quantity changes.    │
└───────────────────────────┬────────────────────────────┘
                            │  Pydantic contract
┌───────────────────────────▼────────────────────────────┐
│  CORE         src/core/   pandas · LightGBM · numpy     │
│  Imputation → quantile forecast → newsvendor → backtest │
│  Pure, offline, reproducible. No LLM client imported.   │
└────────────────────────────────────────────────────────┘
```

**No number a baker acts on is produced by a language model.** The model may propose an adjustment; the quantity comes from `apply_adjustment`, which is arithmetic. The core never imports an LLM client, so the backtest is reproducible and the demo runs with the network off.

The UI is a single static page (`frontend/app.html`): no framework, no build step, no WebGL, no scroll-driven animation, and nothing that fetches a URL.

More detail: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) (core pipeline, results, limitations) and [docs/APP.md](docs/APP.md) (the UI, the override agent, cold start).

---

## Quick start

### See the tool

No install needed. The plan and backtest results are committed.

Open `frontend/app.html` in a browser, fullscreen (F11). It works from `file://`.

Keys `1` `2` `3` jump between the three panels. Type `kal 20 croissant ka order hai` in the header box and press Approve to see the override flow (croissant 51 → 71).

### Rebuild everything from raw data

Python 3.10+.

```bash
pip install -r requirements.txt
```

Download the Kaggle *French bakery sales* dataset and save it as `data/raw/Bakery sales.csv` (it is not committed). Then:

```bash
python src/core/01_load.py                      # keep the top 10 items by volume
python src/core/02_curves.py                    # intraday curves, stockout detection, imputation
python src/core/03_forecast.py 2022-05-27       # nine LightGBM quantile models, held-out tail
python src/core/04_decide.py 2022-05-27 2022-05-21   # newsvendor quantities for the demo day
python src/core/05_backtest.py 2022-05-27       # three-policy replay  <- the key output
python src/core/06_plan.py 2022-05-27 2022-05-21     # writes output/plan.json + frontend/plan-data.js
```

With no date argument, stages 03–06 use the September window instead. The September outputs are also needed by `06_plan.py`, so run 03 and 05 once with no argument too (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md#reproducing-both-windows)).

Optional agent and cold-start scripts:

```bash
python src/agents/override.py             # self-check, seed the cache, export frontend/overrides.js
python src/core/cold_start.py             # self-check, then the Swiss cake example
```

Parsing a new phrase with a real model needs `pip install anthropic` and `ANTHROPIC_API_KEY`. Without them the demo uses the cached phrases.

---

## Repository layout

```
data/economics.csv     price / unit cost / salvage per item (human-maintained, INR)
data/raw/              dataset goes here (not committed)
data/processed/        intermediate parquet and models (regenerated, not committed)
src/core/              01_load … 06_plan, cold_start — deterministic, offline
src/agents/override.py free-text override parser (the only LLM touchpoint)
output/                decision, backtest and plan JSON
frontend/app.html      the tool
frontend/shared.css    design tokens
frontend/plan-data.js  plan as a script, so the page works from file://
frontend/overrides.js  cached override phrases, same reason
.cache/override/       cached model responses for the offline demo
docs/                  methodology and app notes
```

---

## How it is measured

The metric is **rupees**, never an accuracy percentage. Forecast quality is pinball loss; decision quality is profit delta. Three policies are replayed walk-forward over 30 trading days, never with a random split:

- `yesterday` — bake what sold yesterday
- `weekday_avg` — mean of the last four same weekdays (**the real baseline**)
- `proof` — the newsvendor quantity from the quantile forecast

Every policy is scored against reconstructed `true_demand`, and `proof` is also tested at 28%, 33% and 40% unit cost (the cost figures are estimates). It wins at every level in both windows.

## Honest limitations

- **Costs and prices are estimates.** The dataset has prices in euros; the rupee economics in `data/economics.csv` are ours. The patterns are real, the money is assumed, hence the sensitivity sweep.
- **Waste is simulated.** No point-of-sale system records what was thrown away, so each historical day is replayed under each policy against reconstructed demand. That is a replay, not a measurement.
- **Bread barely moves.** With no salvage value its two costs are nearly equal, so the optimal bake sits close to the median. The gain comes from pastry (36% of the spring gain is the newsvendor tilt, all of it on pastry) and from the forecast itself.
- **The spring window was chosen** as the most stable 30 trading days of spring 2022, by a rule applied before looking at results. See the methodology for the details and the other window.
- **Quantile models are noisy.** 81% of tail forecasts crossed before sorting, and the lower half runs slightly high. Models are untuned LightGBM defaults, to avoid leaking the tail.
- **Override parsing is demo-grade.** The three cached phrases are hand-seeded, not model output, and the card says so.
- **Desktop only**, presented fullscreen. No phone layout.

The full list is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md#limitations-and-open-issues).

## Not built, on purpose

Weather and festival features, two-wave baking, substitution modelling, oven-capacity allocation, POS integration, any database or backend server, a phone layout, and the cinematic landing page. They are the roadmap, not gaps in the thinking.

## Data

Transaction data: Kaggle *French bakery daily sales* (check its licence before redistributing). It is the only external dataset used.
