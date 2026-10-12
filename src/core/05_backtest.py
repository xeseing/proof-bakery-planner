"""Phase 5: walk-forward backtest of three bake policies over the final 30 days.

In:  data/processed/daily_demand.parquet  what sold, and what demand really was
     data/processed/forecast.parquet      q0.1 ... q0.9 per item-day, out-of-sample
     data/economics.csv                   price / cost / salvage per item
Out: output/backtest.json                 per policy, split by kind, plus the cost sweep
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
TAIL_END = sys.argv[1] if len(sys.argv) > 1 else None  # the window 03_forecast was run for
SUFFIX = f"_{TAIL_END}" if TAIL_END else ""
OUT = ROOT / f"output/backtest{SUFFIX}.json"
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{tau}" for tau in TAUS]
CO_FLOOR = 0.5           # keeps tau under 1 for an item that costs nothing to waste
SWEEP = [0.28, 0.33, 0.40]  # unit cost as a share of price: our estimate, give or take 20%
POLICIES = ["yesterday", "weekday_avg", "proof"]


def quantile(forecast_row, tau):
    """The day's forecast at `tau`, straight line between the 0.1 ... 0.9 grid."""
    return float(np.interp(tau, TAUS, forecast_row[QCOLS].to_numpy(dtype=float)))


def critical_fractile(econ):
    """tau* per item, the share of days the bake should cover demand."""
    return {r.item: (r.price - r.unit_cost)
                    / ((r.price - r.unit_cost) + max(r.unit_cost - r.salvage_value, CO_FLOOR))
            for r in econ.itertuples()}


def replay(econ, flat_tau=None):
    """Score every policy over the window against what demand really was.

    `flat_tau` swaps the newsvendor fractile for one quantile on every item, which
    is how the median model (tau 0.5, the forecast with no tilt) gets scored.
    """
    tau = {r.item: flat_tau for r in econ.itertuples()} if flat_tau else critical_fractile(econ)
    proof = forecast.assign(quantity=[round(quantile(row, tau[row["item"]]))
                                      for _, row in forecast.iterrows()])
    plans = pd.concat([
        window.assign(policy="yesterday", quantity=window["yesterday"].round()),
        window.assign(policy="weekday_avg", quantity=window["weekday_avg"].round()),
        window.merge(proof[["item", "date", "quantity"]], on=["item", "date"]).assign(policy="proof"),
    ]).merge(econ, on="item")
    assert plans["quantity"].notna().all()

    q, d = plans["quantity"], plans["true_demand"]
    plans["sold"] = np.minimum(q, d)
    plans["leftover"] = (q - d).clip(lower=0)
    plans["stockout"] = (d - q) > 0
    plans["profit"] = (plans["sold"] * plans["price"] + plans["leftover"] * plans["salvage_value"]
                       - q * plans["unit_cost"])
    plans["policy"] = pd.Categorical(plans["policy"], POLICIES, ordered=True)
    plans["kind"] = np.where(plans["salvage_value"] == 0, "bread", "pastry")
    return plans


def totals(plans, keys=("policy",)):
    """Profit, waste and stockouts per policy (optionally split further)."""
    g = plans.groupby(list(keys), observed=True)
    return pd.DataFrame({
        "profit": g["profit"].sum().round(),
        "waste_units": g["leftover"].sum().round(),
        "waste_%": g["leftover"].sum() / g["quantity"].sum() * 100,
        "stockout_item_days": g["stockout"].sum(),
    })


demand = pd.read_parquet(PROCESSED / "daily_demand.parquet").sort_values(["item", "date"])
forecast = pd.read_parquet(PROCESSED / f"forecast{SUFFIX}.parquet")
economics = pd.read_csv(ROOT / "data/economics.csv")

# What the baker could bake from the till alone: yesterday's sales, and the same
# weekday averaged over the last 4 weeks. Our imputed demand is not theirs to use.
demand["yesterday"] = demand.groupby("item")["observed_sales"].shift(1)
demand["weekday_avg"] = (demand.groupby(["item", demand["date"].dt.dayofweek])["observed_sales"]
                         .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()))
window = demand[demand["date"].isin(forecast["date"].unique())]
days = window["date"].nunique()
print(f"window: {window['date'].min():%Y-%m-%d} to {window['date'].max():%Y-%m-%d}, "
      f"{days} trading days, {len(window):,} item-days, scored against true_demand")

plans = replay(economics)
table = totals(plans)
delta = table.loc["proof", "profit"] - table.loc["weekday_avg", "profit"]
print("\nthree policies over the window:")
print(table.round(1).to_string())
print(f"\nproof vs weekday_avg: Rs {delta:+,.0f} over {days} trading days "
      f"(Rs {delta / days:+,.0f} a day, Rs {delta / days * 30:+,.0f} a month)")

split = totals(plans, ["kind", "policy"])
print("\nwhere it comes from:")
print(split.round(1).to_string())

# Where the delta comes from: the model's median is the forecast with no tilt, so
# median - weekday_avg is what the forecast bought, and proof - median is the tilt.
median_profit = (replay(economics, flat_tau=0.5).query("policy == 'proof'")
                 .groupby("kind", observed=True)["profit"].sum())
kind_profit = plans.groupby(["kind", "policy"], observed=True)["profit"].sum().unstack()
decomposition = pd.DataFrame({"weekday_avg": kind_profit["weekday_avg"],
                              "median_model": median_profit, "proof": kind_profit["proof"]})
decomposition.loc["TOTAL"] = decomposition.sum()
decomposition["forecast_adds"] = decomposition["median_model"] - decomposition["weekday_avg"]
decomposition["tilt_adds"] = decomposition["proof"] - decomposition["median_model"]
print("\nforecast or newsvendor tilt?")
print(decomposition.round().to_string())

per_item = totals(plans, ["item", "policy"])["profit"].unstack()
per_item["proof - weekday_avg"] = per_item["proof"] - per_item["weekday_avg"]
print("\nprofit per item:")
print(per_item.sort_values("proof - weekday_avg", ascending=False).round().to_string())
top = per_item["proof - weekday_avg"].idxmax()
top_gain = per_item.loc[top, "proof - weekday_avg"]
print(f"largest single item: {top}, Rs {top_gain:+,.0f} of the Rs {delta:+,.0f} delta "
      f"({top_gain / delta:.0%})")

sweep = {}
for pct in SWEEP:
    swept = totals(replay(economics.assign(unit_cost=(economics["price"] * pct).round(2))))
    sweep[f"{pct:.0%}"] = {"profit": swept["profit"].to_dict(),
                           "delta": swept.loc["proof", "profit"] - swept.loc["weekday_avg", "profit"]}
print(f"\nsensitivity, unit cost at {', '.join(f'{p:.0%}' for p in SWEEP)} of price:")
print(pd.DataFrame({level: {**s["profit"], "proof - weekday_avg": s["delta"]}
                    for level, s in sweep.items()}).round().to_string())
wins = all(s["delta"] > 0 for s in sweep.values()) and delta > 0
print(f"proof beats weekday_avg at every cost level: {wins}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "window": {"start": f"{window['date'].min():%Y-%m-%d}", "end": f"{window['date'].max():%Y-%m-%d}",
               "trading_days": days, "item_days": len(window)},
    "policies": table.round(1).to_dict(orient="index"),
    "proof_vs_weekday_avg": {"rupees": delta, "per_day": delta / days, "per_month": delta / days * 30},
    "by_kind": split.round(1).reset_index().to_dict(orient="records"),
    "decomposition": decomposition.round().reset_index(names="kind").to_dict(orient="records"),
    "largest_item": {"item": top, "rupees": top_gain, "share_of_delta": top_gain / delta},
    "by_item": per_item.round().reset_index().to_dict(orient="records"),
    "sensitivity": sweep,
    "proof_wins_at_every_cost_level": wins,
}, indent=2, default=float))
print(f"saved -> {OUT.relative_to(ROOT).as_posix()}")
