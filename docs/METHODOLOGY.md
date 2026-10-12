# Methodology

How Proof turns a transaction log into a bake list and a rupee figure. Every stage is a script in `src/core/` that runs start to finish, prints sanity checks, and writes a file for the next stage. No stage imports an LLM client.

```
01_load → 02_curves → 03_forecast → 04_decide → 05_backtest → 06_plan
```

## 1. Data (`01_load.py`)

Source: Kaggle *French bakery sales*, 234,005 rows, 2021-01-02 → 2022-09-30 (600 trading days of 637).

- `COUPE` (a €0.15 bread-slicing charge) and `FORMULE SANDWICH` (a €6.50 meal deal) are dropped before ranking: neither is a baked item.
- The top 10 items by units are kept (73.7% of units, 141,218 rows): 7 breads, 3 pastries.
- Refund rows (negative quantity) are kept. They void a sale rung minutes earlier, so daily sums net to what really sold.

`data/economics.csv` holds price, unit cost and salvage value per item in rupees. It is human-maintained. Salvage is 0 for bread and a markdown price for pastry; the code reads `salvage_value == 0` as "bread".

The dataset provides the *patterns*; the rupee figures are our estimates.

## 2. Stockout detection and imputation (`02_curves.py`)

Register sales are censored when an item sells out. We recover true demand in three steps.

1. **Intraday curve.** For each item and weekday, the mean share of the day's sales made before each hour (`fraction_sold_by`).
2. **Stockout flag.** An item-day is flagged when the silence before closing is more than 6× the day's median gap between sales. Gaps are measured in share of a normal day's sales, not clock minutes, because the shop closes 13:00–16:00 and on Sunday afternoons. Days with fewer than 3 sale times are never flagged.
3. **Imputation.** For a flagged day, `true_demand = observed_sales / fraction_sold_by(last sale)`, capped at 3× observed.

The multiple (6×) was chosen from 4×/5×/6× by widest separation from chance. The chance rate is computed in closed form (Rényi representation of exponential order statistics), not simulated:

| Multiple | Flagged | Chance alone | Ratio |
|---|---|---|---|
| 4× | 17.4% | 8.8% | 2.0× |
| 5× | 13.9% | 5.6% | 2.5× |
| 6× | 11.5% | 3.7% | 3.1× |

Items flagged within 2 points of chance are not imputed and their flags are cleared. This excludes `TARTELETTE` (5.1% vs 4.4%).

Result: 11.0% of item-days flagged after the noise floor, mean demand +4.6% overall (+17–97% on stockout days), 31 days hit the 3× cap. Item-days with no sales at all (67) get no row rather than a zero.

## 3. Quantile forecast (`03_forecast.py`)

One LightGBM model per quantile (τ = 0.1 … 0.9), trained on `true_demand` only. The loader reads that single column, so `observed_sales` cannot leak into training.

Features: item, weekday, day of month, month, weekend flag, lags at 1/7/14 days, rolling mean and std over 7 and 28 days. Lags and rolling windows use calendar days, so `lag_7` is always the same weekday; closed days stay `NaN`. LightGBM defaults, no tuning: tuning on the tail would leak into the backtest.

The last 30 trading days are held out. Models are fit once on everything before; each held-out day is forecast from history up to the day before. Predicted quantiles are sorted per row to remove crossing. `forecast_quantile(forecast, item, date, tau)` interpolates linearly between the nine points.

Held-out pinball loss in the September window: **3.14 units** overall (0.68 `CAMPAGNE` → 8.69 `TRADITIONAL BAGUETTE`).

## 4. Newsvendor decision (`04_decide.py`)

```
Cu   = price - unit_cost
Co   = max(unit_cost - salvage_value, 0.5)    # floor keeps tau < 1
tau* = Cu / (Cu + Co)
q*   = round(forecast_quantile(item, date, tau*))
```

Output: quantity, τ*, low/median/high and plain-language drivers per item. The script exits non-zero if the contrast fails (every bread at or below its median, every pastry above), so the check gates the pipeline. Today: bread τ* = 0.475–0.491, pastry 0.882–0.890.

Bread sits near its median under any plausible cost, because with no salvage its under- and over-bake costs are nearly equal. The directional claim is real but small (at 60% food cost `TRADITIONAL BAGUETTE` moves 1.7 units on a base of 152).

## 5. Backtest (`05_backtest.py`)

Three policies are replayed walk-forward over 30 trading days:

| Policy | Bake quantity |
|---|---|
| `yesterday` | previous trading day's observed sales |
| `weekday_avg` | mean of the last four same weekdays' observed sales (**the real baseline**) |
| `proof` | `q*` from the quantile forecast |

Baselines only see `observed_sales`, as a baker with a till would. Every policy is scored against `true_demand`:

```
sold     = min(quantity, true_demand)
leftover = max(quantity - true_demand, 0)
profit   = sold * price + leftover * salvage - quantity * unit_cost
```

A stockout item-day is any unmet demand, however small. Because nothing records waste, this is a **replay against reconstructed demand**, not a measurement.

A sensitivity sweep resets every item's unit cost to 28%, 33% and 40% of price (salvage unchanged).

### Results

| | `yesterday` | `weekday_avg` | `proof` |
|---|---|---|---|
| **Spring** 2022-04-27 → 05-27, profit | ₹217,128 | ₹250,315 | **₹279,694** |
| waste | 18.8% | 14.5% | 16.4% |
| stockout item-days (of 298) | 163 | 132 | 102 |
| **September** 2022-08-30 → 09-30, profit | ₹216,123 | ₹183,614 | **₹274,292** |
| waste | 17.6% | 41.2% | 19.5% |
| stockout item-days (of 297) | 155 | 51 | 79 |

`proof` vs `weekday_avg`: **+₹29,379** (spring), +₹90,678 (September).

Sensitivity, `proof` minus `weekday_avg`:

| Unit cost | 28% | 33% | 40% |
|---|---|---|---|
| Spring | +₹40,215 | +₹32,203 | +₹23,921 |
| September | +₹39,579 | +₹50,165 | +₹70,498 |

`proof` wins at every level in both windows.

### What the gain is made of

- **Spring:** the forecast contributes +₹18,826 and the newsvendor tilt +₹10,553 (36%). All of the tilt is on pastry (+₹10,600, stockout item-days 35 → 13); bread's tilt is −₹47, as expected at τ* ≈ 0.48. One item accounts for 45% of the gain.
- **September:** the tilt is only 7%. The window sits on a seasonal cliff: `TRADITIONAL BAGUETTE` ran 406/day in August and 157/day in September, so a four-week average over-bakes 79% (294 vs 165). That inflates the gain and is why September is the stress case, not the headline.
- Proof wastes slightly *more* than `weekday_avg` in spring (16.4% vs 14.5%). That is the pastry over-bake, bought deliberately to avoid sell-outs; bread waste falls from 13.0% to 7.7%.

### Why the spring window

Chosen as the most stable 30 trading days of spring 2022 by scanning every candidate window March–June on level shift and drift (`TRADITIONAL BAGUETTE`: 174/day against 177/day the four weeks before, −1.6%; September's shift is −60.9%). The rule was applied before looking at results. The demo day, Saturday 2022-05-21, was fixed the same way (last Saturday in the window; the window's final day is the Ascension bridge, when demand ran 2–4× the forecast).

### Reproducing both windows

The app reads both windows, so run the pipeline twice for the forecast and backtest.

```bash
python src/core/01_load.py
python src/core/02_curves.py

python src/core/03_forecast.py                  # September window (default)
python src/core/04_decide.py
python src/core/05_backtest.py                  # -> output/backtest.json

python src/core/03_forecast.py 2022-05-27       # spring window
python src/core/04_decide.py 2022-05-27 2022-05-21
python src/core/05_backtest.py 2022-05-27       # -> output/backtest_2022-05-27.json

python src/core/06_plan.py 2022-05-27 2022-05-21
```

## 6. Markdown schedule (`06_plan.py`)

The intraday curve's second job is projecting the shelf through the day. For each item and hour (08:00–19:00), projected stock is compared with the rest of the day's median demand. When the shelf exceeds it by at least one unit, a markdown applies: 20% off at 12:00, 30% at 17:00, 50% at 18:00. The deepest cut matches the 50% markdown behind the salvage figures. Only the three pastries trigger it.

"Recovered at closing" is projected closing stock × salvage price over the marked items. It is what the markdown saves against dumping, not a second claim on the backtest gain. The header's "protected a day" figure is the 30-day delta ÷ 30, labelled as an average day.

## Limitations and open issues

- **Estimated economics.** Prices and costs are ours; the sweep brackets the pastry convention (33% ± 20%) but at those levels bread's τ* rises to 0.60–0.72, so it does not bracket bread's real ~51%.
- **Simulated waste.** See above: replay, not measurement.
- **Quantile noise.** 81% of tail forecasts crossed before sorting; the median moves 0.95 units on sorting, `TRADITIONAL BAGUETTE` worst. Regularisation tuned on a pre-tail window is the lever.
- **Calibration.** In the tail, 18% of days fall at or below q0.1 and 58% at or below q0.5, while the top is on target (89% at q0.9). With 30 days this is partly noise, but bread baked near the median may run high.
- **Seasonal curves.** One intraday curve per weekday, averaged over the year, cannot follow seasonal shifts in purchase timing. The flag rate doubles in Jul–Aug 2022 (21–25% vs 11–13% a year earlier). A residual third of flags (3.7 of 11.5 points) could still be chance.
- **Mild leakage.** The curve and the 6× rule were fit on all 600 days, tail included. The tail is about 5% of the curve's data.
- **Cold start logic lives twice** (Python canonical, ~12 mirrored lines in `app.html`). See [APP.md](APP.md).
