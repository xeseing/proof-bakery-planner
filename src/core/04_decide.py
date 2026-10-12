"""Phase 4: newsvendor bake quantities from the quantile forecast.

In:  data/economics.csv                  price / cost / salvage per item (human-maintained)
     data/processed/forecast.parquet     q0.1 ... q0.9 per item-day
     data/processed/daily_demand.parquet history behind the drivers
Out: output/decision.json                {item, quantity, tau, low, high, median, drivers}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
TAIL_END = sys.argv[1] if len(sys.argv) > 1 else None    # the window 03_forecast was run for
DECIDE_FOR = sys.argv[2] if len(sys.argv) > 2 else None  # a day inside it; default is its last
WINDOW = f"_{TAIL_END}" if TAIL_END else ""
OUT = ROOT / f"output/decision{f'_{DECIDE_FOR or TAIL_END}' if TAIL_END else ''}.json"
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{tau}" for tau in TAUS]
CO_FLOOR = 0.5  # keeps tau under 1 for an item that costs nothing to waste


def quantile(forecast_row, tau):
    """The day's forecast at `tau`, straight line between the 0.1 ... 0.9 grid.

    Same reading as 03_forecast.forecast_quantile; numbered scripts can't import each other.
    """
    return float(np.interp(tau, TAUS, forecast_row[QCOLS].to_numpy(dtype=float)))


economics = pd.read_csv(ROOT / "data/economics.csv")
forecast = pd.read_parquet(PROCESSED / f"forecast{WINDOW}.parquet")
demand = pd.read_parquet(PROCESSED / "daily_demand.parquet")
assert set(economics["item"]) == set(forecast["item"]), "economics.csv and the forecast disagree"

date = pd.Timestamp(DECIDE_FOR) if DECIDE_FOR else forecast["date"].max()
today = forecast[forecast["date"] == date].set_index("item")
history = demand[demand["date"] < date]  # only what the baker knows at 5am
print(f"bake plan for {date:%A %d %B %Y}\n")

rows, plan = [], []
for e in economics.itertuples():
    cu = int(e.price - e.unit_cost)                         # margin lost when you sell out
    co = max(int(e.unit_cost - e.salvage_value), CO_FLOOR)  # loss on one unsold unit
    tau = cu / (cu + co)
    f = today.loc[e.item]
    bake, median = quantile(f, tau), quantile(f, 0.5)

    h = history[history["item"] == e.item]
    same_weekday = h[h["date"].dt.dayofweek == date.dayofweek]
    weekday_lift = same_weekday.tail(8)["true_demand"].mean() / h.tail(56)["true_demand"].mean() - 1
    trend = h.tail(7)["true_demand"].mean() / h.tail(28)["true_demand"].mean() - 1
    sold_out = int(same_weekday.tail(4)["was_stockout"].sum())

    plan.append({
        "item": e.item,
        "quantity": round(bake),
        "tau": round(tau, 3),
        "low": round(quantile(f, 0.2)),
        "high": round(quantile(f, 0.8)),
        "median": round(median),
        # detail reads as a phrase for the Phase 7 explainer; value lets the UI decide
        # which drivers are worth showing at all.
        "drivers": [
            {"factor": "weekday", "detail": f"{date:%A}, {weekday_lift:+.0%} vs its average day",
             "value": round(float(weekday_lift), 3)},
            {"factor": "trend", "detail": f"last 7 days {trend:+.0%} vs the last 28",
             "value": round(float(trend), 3)},
            {"factor": "stockouts", "detail": f"sold out {sold_out} of the last 4 {date:%A}s",
             "value": sold_out},
        ],
    })
    rows.append({
        "item": e.item, "price": e.price, "cost": e.unit_cost, "salvage": e.salvage_value,
        "Cu": cu, "Co": co, "tau": tau, "median": median, "bake": bake,
        "vs_median": "above" if bake > median else "below" if bake < median else "at",
        # This table's convention: bread carries no salvage, pastry sells on at a markdown.
        "bread": e.salvage_value == 0,
    })

table = pd.DataFrame(rows).set_index("item")
print(table.round({"tau": 3, "median": 1, "bake": 1}).to_string())
print(f"\ndrivers, {plan[0]['item']}: {json.dumps(plan[0]['drivers'])}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"date": f"{date:%Y-%m-%d}", "bake_list": plan}, indent=2))
print(f"saved {len(plan)} decisions -> {OUT.relative_to(ROOT).as_posix()}")

bread, pastry = table[table["bread"]], table[~table["bread"]]
wrong = pd.concat([bread[bread["vs_median"] != "below"], pastry[pastry["vs_median"] != "above"]])
print(f"\nsanity check: all {len(bread)} bread below the median forecast, "
      f"all {len(pastry)} pastry above?")
if len(wrong):
    raise SystemExit(
        "STOP: " + "; ".join(f"{i} lands {r.vs_median} its median (tau {r.tau:.3f})"
                             for i, r in wrong.iterrows())
        + ".\nA tau of 0.500 means Cu == Co, so the median IS the newsvendor answer. Check "
          "data/economics.csv: with no salvage, bread needs unit_cost above half of price.")
print("PASS: every bread below its median, every pastry above.")
