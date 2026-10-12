"""Phase 3: quantile demand forecast, one LightGBM model per quantile.

In:  data/processed/daily_demand.parquet        (target: true_demand)
Out: data/processed/models/q0.1.txt ... q0.9.txt  the nine models
     data/processed/forecast.parquet              item, date, q0.1 ... q0.9 for the held-out tail
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{tau}" for tau in TAUS]
TAIL_DAYS = 30  # held out: the same days the Phase 5 backtest replays
TAIL_END = sys.argv[1] if len(sys.argv) > 1 else None  # a date, or the last trading day
SUFFIX = f"_{TAIL_END}" if TAIL_END else ""
MODELS = PROCESSED / f"models{SUFFIX}"
FEATURES = ["item", "weekday", "day_of_month", "month", "is_weekend",
            "lag_1", "lag_7", "lag_14", "mean_7", "std_7", "mean_28", "std_28"]


def forecast_quantile(forecast, item, date, tau):
    """Demand quantile `tau` for `item` on `date`, straight line between the 0.1 ... 0.9 grid.

    Outside 0.1-0.9 it returns the nearest end of the grid.
    """
    row = forecast[(forecast["item"] == item) & (forecast["date"] == pd.Timestamp(date))]
    return float(np.interp(tau, TAUS, row[QCOLS].iloc[0]))


# Only the target is loaded, so observed_sales cannot leak into training.
demand = pd.read_parquet(PROCESSED / "daily_demand.parquet", columns=["date", "item", "true_demand"])
print(f"target:        true_demand, {len(demand):,} item-days (observed_sales not loaded)")

# History features on a full calendar, so lag_7 is the same weekday last week.
# Closed days and days an item wasn't offered stay NaN; LightGBM handles those.
wide = demand.pivot(index="date", columns="item", values="true_demand")
wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
history = {
    "lag_1": wide.shift(1),
    "lag_7": wide.shift(7),
    "lag_14": wide.shift(14),
    "mean_7": wide.rolling(7, min_periods=1).mean().shift(1),  # the 7 days before, never today
    "std_7": wide.rolling(7, min_periods=2).std().shift(1),
    "mean_28": wide.rolling(28, min_periods=1).mean().shift(1),
    "std_28": wide.rolling(28, min_periods=2).std().shift(1),
}
history = pd.concat({name: f.stack(future_stack=True) for name, f in history.items()}, axis=1)
history.index.names = ["date", "item"]

data = demand.join(history, on=["date", "item"])
data["item"] = data["item"].astype("category")
data["weekday"] = data["date"].dt.dayofweek.astype("category")
data["day_of_month"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["is_weekend"] = (data["date"].dt.dayofweek >= 5).astype(int)

# Walk-forward: fit on everything before the tail, then forecast each tail day
# from the history up to the day before, as the baker would at 5am.
dates = pd.DatetimeIndex(np.sort(data["date"].unique()))
if TAIL_END:  # an earlier window: everything after it is the future and is dropped
    dates = dates[dates <= pd.Timestamp(TAIL_END)]
tail_start, tail_end = dates[-TAIL_DAYS], dates[-1]
train = data[data["date"] < tail_start]
tail = data[data["date"].between(tail_start, tail_end)]
print(f"train:         {len(train):,} item-days, to {train['date'].max():%Y-%m-%d}")
print(f"held-out tail: {len(tail):,} item-days, {tail_start:%Y-%m-%d} to {tail['date'].max():%Y-%m-%d}")

MODELS.mkdir(parents=True, exist_ok=True)
models = {}
for tau in TAUS:
    models[tau] = lgb.LGBMRegressor(objective="quantile", alpha=tau, random_state=0, verbose=-1)
    models[tau].fit(train[FEATURES], train["true_demand"])
    models[tau].booster_.save_model(MODELS / f"q{tau}.txt")

raw = np.column_stack([models[tau].predict(tail[FEATURES]) for tau in TAUS])
crossed = np.diff(raw, axis=1) < 0
print(f"\nquantiles crossed before sorting: {crossed.any(axis=1).mean():.1%} of item-day "
      f"forecasts, {crossed.mean():.1%} of adjacent pairs. Sorted.")
pred = np.sort(raw, axis=1)

taus = np.array(TAUS)
y = tail["true_demand"].to_numpy()[:, None]
pinball = np.maximum(taus * (y - pred), (taus - 1) * (y - pred)).mean(axis=1)
scores = pd.DataFrame({"item": tail["item"].astype(str).to_numpy(), "pinball_loss": pinball,
                       "mean_true_demand": tail["true_demand"].to_numpy()})
print("\npinball loss on the held-out tail (units, mean over the 9 quantiles):")
print(scores.groupby("item")[["pinball_loss", "mean_true_demand"]].mean().round(2).to_string())
print(f"overall: {pinball.mean():.2f}")
print("share of tail days at or below each quantile (should sit near tau):")
print("  " + "  ".join(f"{tau}: {share:.2f}" for tau, share in zip(TAUS, (y <= pred).mean(axis=0))))

gain = pd.Series(models[0.5].booster_.feature_importance("gain"), index=FEATURES)
print("\nmedian model (tau 0.5), share of total gain by feature (%):")
print((gain / gain.sum() * 100).sort_values(ascending=False).round(1).to_string())

forecast = pd.DataFrame(pred, columns=QCOLS)
forecast.insert(0, "date", tail["date"].to_numpy())
forecast.insert(0, "item", tail["item"].astype(str).to_numpy())
forecast.to_parquet(PROCESSED / f"forecast{SUFFIX}.parquet", index=False)
print(f"\nsaved 9 models -> data/processed/models{SUFFIX}/, {len(forecast):,} forecasts "
      f"-> data/processed/forecast{SUFFIX}.parquet")

last = forecast.iloc[-1]
q = forecast_quantile(forecast, last["item"], last["date"], 0.85)
assert last["q0.8"] <= q <= last["q0.9"]
print(f"forecast_quantile({last['item']!r}, {last['date']:%Y-%m-%d}, 0.85) = {q:.1f}, "
      f"between q0.8 {last['q0.8']:.1f} and q0.9 {last['q0.9']:.1f}")
