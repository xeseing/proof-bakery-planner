"""Phase 1: load the bakery transaction log and keep the top 10 items by volume.

In:  data/raw/Bakery sales.csv  (Kaggle "French bakery sales")
Out: data/processed/transactions.parquet  (datetime, transaction_id, item, quantity)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/Bakery sales.csv"
OUT = ROOT / "data/processed/transactions.parquet"
# Not bakes, dropped before ranking: COUPE is the 0.15 EUR bread-slicing charge,
# FORMULE SANDWICH a 6.50 EUR lunch meal deal that would swamp every rupee total.
NOT_BAKED = ["COUPE", "FORMULE SANDWICH"]

raw = pd.read_csv(RAW)
print(f"raw rows:      {len(raw):,}")

df = pd.DataFrame({
    "datetime": pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y-%m-%d %H:%M"),
    "transaction_id": raw["ticket_number"].astype(int),
    "item": raw["article"],
    "quantity": raw["Quantity"].astype(int),
}).sort_values(["datetime", "transaction_id"])  # the raw file is not in time order

# Refunds are separate tickets with negative quantity, almost all voiding a sale
# rung a minute earlier. Keeping them makes daily sums net to what really sold.
refunds = df[df["quantity"] < 0]
print(f"refund rows:   {len(refunds):,} kept ({refunds['quantity'].sum():,} units)")

not_baked = df["item"].isin(NOT_BAKED)
print(f"not baked:     {not_baked.sum():,} rows dropped ({', '.join(NOT_BAKED)})")
df = df[~not_baked]

days = df["datetime"].dt.normalize()
print(f"date range:    {days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}")
print(f"trading days:  {days.nunique()} of {(days.max() - days.min()).days + 1} calendar days")
print(f"unique items:  {df['item'].nunique()}")

volume = (df.groupby("item")["quantity"].agg(units="sum", rows="size")
          .sort_values("units", ascending=False))
volume["share"] = volume["units"] / volume["units"].sum()
top10 = volume.head(10)
print("\ntop 10 items by volume:")
print(top10.to_string(formatters={"share": "{:.1%}".format}))
print(f"\ntop 10 = {top10['share'].sum():.1%} of units; "
      f"dropping the other {len(volume) - 10} items")

df = df[df["item"].isin(top10.index)]
assert df["item"].nunique() == 10

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\nsaved {len(df):,} rows -> {OUT.relative_to(ROOT)}")
