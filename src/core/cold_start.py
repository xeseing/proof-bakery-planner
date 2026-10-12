"""A bake quantity for an item with no sales history.

Borrow the demand shape of the most similar item the bakery already sells, scale it
to the volume the owner expects, and run the same newsvendor on the new item's own
economics. No model is trained and no LLM is involved: similarity here is category
first, then nearest price, because that is what a baker would say too.

    py -3.13 src/core/cold_start.py            self-check, then the Swiss cake example
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CO_FLOOR = 0.5  # same floor the newsvendor uses elsewhere


def most_similar(kind, price, catalogue):
    """The item to borrow from: same category, then closest price."""
    pool = [c for c in catalogue if c["kind"] == kind]
    if not pool:
        raise ValueError(f"nothing on the menu is a {kind}")
    match = min(pool, key=lambda c: abs(c["price"] - price))
    return match, f"same category, closest price (Rs {match['price']} against Rs {price})"


def at_tau(quantiles, tau):
    """Read the nine-point curve at `tau`, straight line between points, ends clamped."""
    x = min(TAUS[-1], max(TAUS[0], tau))
    i = min(len(TAUS) - 2, int((x - TAUS[0]) / 0.1))
    return quantiles[i] + ((x - TAUS[i]) / 0.1) * (quantiles[i + 1] - quantiles[i])


def borrowed_plan(match, daily_volume, price, unit_cost, salvage_value):
    """Scale the borrowed curve to the expected volume, then bake at this item's tau*."""
    if price <= unit_cost:
        raise ValueError("price must be above unit cost")
    if daily_volume <= 0:
        raise ValueError("expected daily volume must be above zero")
    scale = daily_volume / match["quantiles"][4]         # index 4 is the middle, tau 0.5
    quantiles = [q * scale for q in match["quantiles"]]
    cu = price - unit_cost
    co = max(unit_cost - salvage_value, CO_FLOOR)
    tau = cu / (cu + co)
    return {"tau": round(tau, 3), "quantiles": [round(q, 1) for q in quantiles],
            "quantity": round(at_tau(quantiles, tau)), "median": round(at_tau(quantiles, 0.5)),
            "low": round(at_tau(quantiles, 0.2)), "high": round(at_tau(quantiles, 0.8))}


def demo():
    menu = [{"item": "CROISSANT", "kind": "pastry", "price": 90, "quantiles": [1, 2, 3, 4, 10, 6, 7, 8, 9]},
            {"item": "TARTELETTE", "kind": "pastry", "price": 110, "quantiles": [4, 4, 5, 6, 10, 8, 9, 10, 13]},
            {"item": "BAGUETTE", "kind": "bread", "price": 35, "quantiles": [2, 2, 3, 3, 10, 4, 4, 5, 6]}]
    match, why = most_similar("pastry", 120, menu)
    assert match["item"] == "TARTELETTE" and "closest price" in why       # 110 beats 90
    assert most_similar("bread", 120, menu)[0]["item"] == "BAGUETTE"      # category wins over price
    assert at_tau([0, 1, 2, 3, 4, 5, 6, 7, 8], 0.5) == 4                  # reads the middle
    assert at_tau([0, 1, 2, 3, 4, 5, 6, 7, 8], 0.45) == 3.5               # halfway between points
    assert at_tau([0, 1, 2, 3, 4, 5, 6, 7, 8], 0.99) == 8                 # clamped at the end

    plan = borrowed_plan(match, daily_volume=20, price=120, unit_cost=40, salvage_value=30)
    assert plan["median"] == 20                     # scaled so the middle is the stated volume
    assert plan["tau"] == 0.889                     # 80 / (80 + 10)
    assert plan["quantity"] > plan["median"]        # high margin: bake above the middle
    bread = borrowed_plan(menu[2], 20, price=35, unit_cost=18, salvage_value=0)
    assert bread["quantity"] <= bread["median"]     # bread sits at or under the middle
    for bad in [lambda: borrowed_plan(match, 20, 40, 40, 0), lambda: borrowed_plan(match, 0, 120, 40, 30)]:
        try:
            bad()
            raise AssertionError("bad input should not produce a quantity")
        except ValueError:
            pass
    print("self-check ok")


if __name__ == "__main__":
    demo()
    menu = json.loads((ROOT / "output/plan.json").read_text())["bake_list"]
    match, why = most_similar("pastry", 120, menu)
    plan = borrowed_plan(match, daily_volume=12, price=120, unit_cost=40, salvage_value=30)
    print(f"Swiss cake (pastry, Rs 120, sells about 12 a day)")
    print(f"  modelled on {match['item']} — {why}")
    print(f"  bake {plan['quantity']}, middle {plan['median']}, "
          f"usual range {plan['low']}-{plan['high']}, tau {plan['tau']}")
