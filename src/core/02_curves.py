"""Phase 2: intraday sales curve, stockout detection, demand imputation.

In:  data/processed/transactions.parquet
Out: data/processed/intraday_curve.parquet  item, weekday, hour, fraction_sold_by
     data/processed/daily_demand.parquet    date, item, observed_sales, was_stockout,
                                            stockout_time, true_demand
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
GAP_MULTIPLES = [4, 5, 6]  # sold out = closing silence > this many typical gaps; best one kept
MIN_SALES = 3              # with fewer sale times than this, a day's rhythm can't be judged
NOISE_FLOOR = 2            # an item flagged within 2 points of chance is not imputed at all
MAX_UPLIFT = 3             # imputation never scales a day's sales up more than 3x


def build_curve(sales):
    """Mean share of the day's sales made before each whole hour, per (item, weekday)."""
    hourly = (sales.assign(hour=sales["datetime"].dt.hour)
              .pivot_table(index=["item", "weekday", "date"], columns="hour",
                           values="quantity", aggfunc="sum", fill_value=0)
              .reindex(columns=range(24), fill_value=0))
    share = hourly.cumsum(axis=1).div(hourly.sum(axis=1), axis=0)
    share.columns = share.columns + 1  # sold in hours 0..h = sold before (h+1):00
    share.insert(0, 0, 0.0)            # nothing sold before 00:00
    curve = share.groupby(level=["item", "weekday"]).mean().reset_index()
    return (curve.melt(id_vars=["item", "weekday"], var_name="hour", value_name="fraction_sold_by")
            .astype({"hour": int})
            .sort_values(["item", "weekday", "hour"], ignore_index=True))


def fraction_sold_by(curve, item, weekday, hour):
    """Share of a normal day's sales of `item` made before clock time `hour` (11.5 = 11:30).

    Straight line between whole hours. Takes one hour or an array of them.
    """
    c = curve[(curve["item"] == item) & (curve["weekday"] == weekday)]
    return np.interp(hour, c["hour"], c["fraction_sold_by"])


def fraction_sold_by_rows(curve, rows, clock):
    """fraction_sold_by for every row of `rows` (needs item, weekday) at the given clock times."""
    out = pd.Series(np.nan, index=rows.index)
    for (item, weekday), g in rows.groupby(["item", "weekday"]):
        out.loc[g.index] = fraction_sold_by(curve, item, weekday, clock.loc[g.index])
    return out


def chance_of_flag(sale_times, multiple):
    """Chance the rule flags a day with NO stockout, given its number of sale times.

    With no stockout, sales fall at random in share-of-day time, so the gaps are
    independent exponentials. This is the exact P(closing silence > multiple x
    median gap), via Renyi's representation of exponential order statistics.
    Matches a brute-force simulation.
    """
    k = sale_times - 1                    # gaps between sales
    rates = k - np.arange((k + 1) // 2)  # one factor per order statistic up to the median
    p = np.prod(rates / (rates + multiple))
    if k % 2 == 0:                        # even count: median is the mean of the middle two
        p *= k / (k + multiple)
    return p


def flag_rates(days, multiple):
    """Per item: % of item-days flagged at `multiple`, and % that chance alone would flag."""
    rates = pd.DataFrame({
        "flag%": days["judged"] & (days["silence"] > multiple * days["median_gap"]),
        "chance%": days["sale_times"].map(lambda n: chance_of_flag(n, multiple))
                   .where(days["judged"], 0),
    })
    rates = pd.concat([rates.groupby(days["item"]).mean(),
                       rates.mean().to_frame("ALL ITEMS").T]) * 100
    rates["ratio"] = rates["flag%"] / rates["chance%"]
    return rates


def hhmm(hour):
    return f"{int(hour):02d}:{int(hour % 1 * 60):02d}"


tx = pd.read_parquet(PROCESSED / "transactions.parquet")
tx["date"] = tx["datetime"].dt.normalize()
tx["weekday"] = tx["datetime"].dt.dayofweek                          # 0 = Monday
tx["clock"] = tx["datetime"].dt.hour + tx["datetime"].dt.minute / 60  # 11:20 -> 11.33
sales = tx[tx["quantity"] > 0]                                       # refunds are not sale events
print(f"transactions:  {len(tx):,} rows, {tx['item'].nunique()} items, "
      f"{tx['date'].nunique()} trading days")

# (a) Pass 1: the curve from every day.
curve1 = build_curve(sales)

# (b) Stockout detection. A silence only means something if customers normally buy
# in that window, so gaps are measured in share of a normal day's sales, not minutes:
# the 8am rush counts for a lot, the 13:00-16:00 lunch closure for almost nothing.
times = (sales.drop_duplicates(["item", "date", "clock"])  # same-minute tickets count once
         .sort_values(["item", "date", "clock"]))
times["sold_by"] = fraction_sold_by_rows(curve1, times, times["clock"])
times["gap"] = times.groupby(["item", "date"])["sold_by"].diff()
times["gap_minutes"] = times.groupby(["item", "date"])["clock"].diff() * 60

days = (times.groupby(["item", "date"])
        .agg(weekday=("weekday", "first"), sale_times=("clock", "size"),
             stockout_time=("datetime", "max"), last_clock=("clock", "max"),
             last_sold_by=("sold_by", "max"), median_gap=("gap", "median"))
        .reset_index())
days["close"] = days["date"].map(sales.groupby("date")["clock"].max())  # shop's last sale that day
days["silence"] = fraction_sold_by_rows(curve1, days, days["close"]) - days["last_sold_by"]
days["judged"] = days["sale_times"] >= MIN_SALES

comparison = pd.concat({m: flag_rates(days, m) for m in GAP_MULTIPLES}, axis=1)
multiple = max(GAP_MULTIPLES, key=lambda m: comparison.loc["ALL ITEMS", (m, "ratio")])
print("\n% of item-days flagged vs % that chance alone would flag, per gap multiple:")
print(comparison.round(1).to_string())
print(f"-> keeping {multiple}x: widest separation from chance")
days["was_stockout"] = days["judged"] & (days["silence"] > multiple * days["median_gap"])

by_item = days.groupby("item")
gaps = pd.DataFrame({
    "sales/day": by_item["sale_times"].median(),
    "gap_min": times.groupby("item")["gap_minutes"].median(),
    "gap%_p10": by_item["median_gap"].quantile(0.1) * 100,
    "gap%_p50": by_item["median_gap"].median() * 100,
    "gap%_p90": by_item["median_gap"].quantile(0.9) * 100,
})
gaps["threshold%"] = multiple * gaps["gap%_p50"]
print("\ngaps between sales. gap_min = minutes; gap% = share of a normal day's sales inside")
print(f"the gap (each day's median, spread across days). Flag: closing silence > {multiple}x gap%.")
print(gaps.round(1).to_string())

chosen = comparison[multiple].drop("ALL ITEMS")
above_chance = chosen["flag%"] - chosen["chance%"]
flags = pd.DataFrame({
    "item_days": by_item.size(),
    "judged": by_item["judged"].sum(),
    "flagged": by_item["was_stockout"].sum(),
    "points_above_chance": above_chance,
    "imputed": above_chance > NOISE_FLOOR,
    "median_stockout": days[days["was_stockout"]].groupby("item")["last_clock"].median()
                       .map(hhmm, na_action="ignore"),
})
print(f"\nstockout flags at {multiple}x: {days['was_stockout'].mean():.1%} of {len(days):,} item-days")
print(flags.round(1).to_string())

out_of_range = (chosen["flag%"] < 2) | (chosen["flag%"] > 40)
if out_of_range.sum() > len(chosen) / 2:
    raise SystemExit(f"\nSTOP: flag rate outside 2-40% for {out_of_range.sum()} of {len(chosen)} "
                     "items. The threshold is wrong; see the gap distribution above.")

# Noise floor: an item flagged barely more often than chance would flag it has no
# stockout signal, so none of its days count as stockouts and none get imputed.
noisy = flags.index[~flags["imputed"]].tolist()
days.loc[days["item"].isin(noisy), "was_stockout"] = False
print(f"noise floor: not imputing {', '.join(noisy) or 'no items'} "
      f"(within {NOISE_FLOOR} points of chance)")

# (a) Pass 2: rebuild the curve without stockout days, whose curves are cut short.
sales = sales.merge(days[["item", "date", "was_stockout"]], on=["item", "date"])
curve2 = build_curve(sales[~sales["was_stockout"]])
assert np.allclose(curve2.loc[curve2["hour"] == 24, "fraction_sold_by"], 1)
curve2.to_parquet(PROCESSED / "intraday_curve.parquet", index=False)
print(f"\ncurve rebuilt without {days['was_stockout'].sum():,} stockout days "
      f"-> data/processed/intraday_curve.parquet")

# (c) Imputation: a day that sold out at 11:20 saw only the demand up to 11:20.
daily = days.join(tx.groupby(["item", "date"])["quantity"].sum().rename("observed_sales"),
                  on=["item", "date"])
assert (daily["observed_sales"] > 0).all()
f = fraction_sold_by_rows(curve2, daily, daily["last_clock"])
multiplier = (1 / f).clip(1, MAX_UPLIFT).where(daily["was_stockout"], 1)
daily["true_demand"] = daily["observed_sales"] * multiplier
daily["stockout_time"] = daily["stockout_time"].where(daily["was_stockout"])
daily["capped"] = daily["was_stockout"] & (1 / f > MAX_UPLIFT)

by_item = daily.groupby("item")
flagged = daily[daily["was_stockout"]].groupby("item")[["observed_sales", "true_demand"]].sum()
uplift = pd.DataFrame({
    "observed_mean": by_item["observed_sales"].mean(),
    "true_demand_mean": by_item["true_demand"].mean(),
})
uplift["uplift%"] = (uplift["true_demand_mean"] / uplift["observed_mean"] - 1) * 100
uplift["uplift%_stockout_days"] = (flagged["true_demand"] / flagged["observed_sales"] - 1) * 100
uplift["capped_3x"] = by_item["capped"].sum()
print("\nimputed demand per item-day:")
print(uplift.round(1).to_string())
print(f"\noverall: true_demand {daily['true_demand'].sum() / daily['observed_sales'].sum() - 1:+.1%} "
      f"vs observed_sales")

no_sales = tx["date"].nunique() * tx["item"].nunique() - len(daily)
print(f"item-days with no sales at all: {no_sales} (treated as not offered, no row)")

cols = ["date", "item", "observed_sales", "was_stockout", "stockout_time", "true_demand"]
daily[cols].to_parquet(PROCESSED / "daily_demand.parquet", index=False)
print(f"saved {len(daily):,} item-days -> data/processed/daily_demand.parquet")
