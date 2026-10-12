"""Phase 6.1: assemble the plan the app renders, for one demo day.

In:  output/decision<window>.json           bake quantities for the day
     output/backtest<window>.json           the replay that leads the report
     output/backtest.json                   the September replay, kept as the stress case
     data/processed/intraday_curve.parquet  when the day's sales actually happen
Out: output/plan.json                       the plan
     frontend/plan-data.js                  the same, as a script app.html can load from file://
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
OUTPUT = ROOT / "output"
WINDOW = sys.argv[1] if len(sys.argv) > 1 else None  # which backtest window leads the report
DAY = sys.argv[2] if len(sys.argv) > 2 else None     # which day in it the plan is for
SUFFIX = f"_{WINDOW}" if WINDOW else ""
DAY_SUFFIX = f"_{DAY or WINDOW}" if WINDOW else ""
QCOLS = [f"q{tau}" for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
CLOCK_HOURS = range(8, 20)                 # the shelf, hour by hour, to closing
MARKDOWN_HOURS = {12: 20, 17: 30, 18: 50}  # hour -> % off, deepening towards closing
MIN_SURPLUS = 1                            # never mark down over a fraction of an item


def report(backtest, note):
    """The handful of backtest numbers the report panel shows."""
    return {
        "start": backtest["window"]["start"],
        "end": backtest["window"]["end"],
        "trading_days": backtest["window"]["trading_days"],
        "note": note,
        "profit": {p: v["profit"] for p, v in backtest["policies"].items()},
        "waste_pct": {p: v["waste_%"] for p, v in backtest["policies"].items()},
        "stockout_item_days": {p: v["stockout_item_days"] for p, v in backtest["policies"].items()},
        "delta": backtest["proof_vs_weekday_avg"]["rupees"],
        "per_month": backtest["proof_vs_weekday_avg"]["per_month"],
        "sensitivity": {level: s["delta"] for level, s in backtest["sensitivity"].items()},
    }


decision = json.loads((OUTPUT / f"decision{DAY_SUFFIX}.json").read_text())
headline = json.loads((OUTPUT / f"backtest{SUFFIX}.json").read_text())
stress = json.loads((OUTPUT / "backtest.json").read_text())
curve = pd.read_parquet(PROCESSED / "intraday_curve.parquet")

date = pd.Timestamp(decision["date"])
# Same split the backtest uses: bread carries no salvage, pastry sells on at a markdown.
economics = pd.read_csv(ROOT / "data/economics.csv")
kind = {r.item: ("bread" if r.salvage_value == 0 else "pastry") for r in economics.itertuples()}
salvage = {r.item: r.salvage_value for r in economics.itertuples()}
price = {r.item: r.price for r in economics.itertuples()}

# The nine quantiles ride along per item: a new product with no history borrows the
# shape of the nearest one it can find.
forecast = pd.read_parquet(PROCESSED / f"forecast{SUFFIX}.parquet")
shape = forecast[forecast["date"] == pd.Timestamp(decision["date"])].set_index("item")
bake_list = [{**item, "kind": kind[item["item"]], "price": int(price[item["item"]]),
              "quantiles": [round(float(v), 1) for v in shape.loc[item["item"], QCOLS]],
              "reason": ""}  # Phase 7 writes reason
             for item in decision["bake_list"]]

# The intraday curve's second job: project the shelf through the day. Whatever the
# rest of the day's demand will not clear is what gets marked down, and the later
# it is, the deeper the cut.
schedule = []
for item in bake_list:
    c = curve[(curve["item"] == item["item"]) & (curve["weekday"] == date.dayofweek)]
    for hour in CLOCK_HOURS:
        sold_by = float(np.interp(hour, c["hour"], c["fraction_sold_by"]))
        demand_left = item["median"] * (1 - sold_by)
        on_shelf = item["quantity"] - min(item["quantity"], item["median"] * sold_by)
        surplus = on_shelf - demand_left
        schedule.append({
            "hour": hour,
            "item": item["item"],
            "remaining": round(on_shelf, 1),
            "expected_remaining_demand": round(demand_left, 1),
            "discount_pct": MARKDOWN_HOURS.get(hour, 0) if surplus >= MIN_SURPLUS else 0,
        })

# What the markdown is worth: whatever is still on the shelf at closing comes back
# at the salvage price instead of going in the bin.
closing = {r["item"]: r["remaining"] for r in schedule if r["hour"] == max(CLOCK_HOURS)}
marked_items = {r["item"] for r in schedule if r["discount_pct"]}

plan = {
    "date": decision["date"],
    "protected_rupees": round(headline["proof_vs_weekday_avg"]["rupees"]
                              / headline["window"]["trading_days"]),
    "recovered_rupees": round(sum(closing[i] * salvage[i] for i in marked_items)),
    "bake_list": bake_list,
    "markdown_schedule": schedule,
    "backtest": {"headline": report(headline, "a normal month"),
                 "stress": report(stress, "when the season turns")},
}

(OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))
(ROOT / "frontend").mkdir(exist_ok=True)
(ROOT / "frontend/plan-data.js").write_text(f"window.PLAN = {json.dumps(plan, indent=2)};\n")

marked = sorted({row["item"] for row in schedule if row["discount_pct"]})
print(f"plan for {date:%A %d %B %Y}: {len(bake_list)} items, "
      f"Rs {plan['protected_rupees']:,} protected a day")
for name, r in plan["backtest"].items():
    print(f"{name:9s} {r['start']} to {r['end']}, Rs {r['delta']:+,.0f}")
print(f"markdowns: {', '.join(marked) if marked else 'none'}, "
      f"Rs {plan['recovered_rupees']:,} recovered at closing")
print("saved -> output/plan.json and frontend/plan-data.js")
