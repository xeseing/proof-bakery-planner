# The app, the override agent, and cold start

## `frontend/app.html` — the tool

One static file: vanilla HTML/CSS/JS, no framework, no build step, no webfonts, nothing that fetches a URL. It reads two scripts that sit next to it, so it works opened straight from disk with the network off.

| File | Written by | Holds |
|---|---|---|
| `frontend/plan-data.js` | `src/core/06_plan.py` | the plan (same content as `output/plan.json`) |
| `frontend/overrides.js` | `src/agents/override.py` | cached override phrases |
| `frontend/shared.css` | by hand | design tokens |

A `file://` page cannot `fetch()` JSON but can load a script tag, which is why the data ships as `.js`. If you regenerate the plan, regenerate both outputs (`06_plan.py` writes both).

**Three panels, all visible at once, no scrolling model.** Press `1`, `2` or `3` to jump to one. `Esc` closes the override card.

1. **Bake list** — pastry first, in large numerals with the delta against a usual day; bread folded into one summary line you can open. Each block's heading states the economics ("running out costs 8× an unsold one"). A reason appears only when it is not obvious.
2. **Report** — the backtest: profit by policy, the headline gain, the sensitivity sweep. The spring window leads; September is the stress row.
3. **Markdown clock** — the discount schedule by hour with the cuts annotated once on the shared axis, and the rupees recovered at closing.

Design rules: no WebGL, no scroll-driven animation, motion only when an override is approved. The palette comes from a bakery at 5 a.m.: ink `#0B1017`, panel `#141B25`, ember `#FF8A3D` (the single accent, reserved for pastry numbers), flour `#F2EDE4`, loss `#E5484D`, gain `#3DD68C`.

Present fullscreen (F11). The layout is built for 1024×768 and 1280×800; in a window, browser chrome pushes the markdown clock below the fold. Desktop only.

## `src/agents/override.py` — the one LLM touchpoint

A baker types a message like `kal 20 croissant ka order hai`. The agent turns it into a structured proposal; a human approves; only then does a number change.

```
message ──► parse_override ──► ProposedAdjustment ──► approval card ──► apply_adjustment
            (cache, model,      {item, delta_units,     (Approve /       (arithmetic,
             or fallback)        reason, confidence}      Reject)          floored at 0)
```

- **The model never computes a quantity.** `apply_adjustment(bake_list, item, delta)` adds the delta to one item and floors at zero. The parsed output is a Pydantic model; confidence outside 0–1 fails validation.
- **Cache first, model second, documented failure third.** Responses are stored in `.cache/override/` keyed on the prompt hash. On a miss with the network off, or on unusable output, the node returns an empty proposal and reports `degraded: override_parse`.
- **The browser never calls a model.** The script exports the cache to `frontend/overrides.js` and the page only reads that. The in-page logic mirrors `apply_adjustment`; a quantity (1–500) is read arithmetically from the message, so any number works with no cached entry. The number in the message must match exactly; the item name matches fuzzily.
- **The three cached phrases are hand-seeded**, not model output, and the card says "cached (seeded example)". Setting `ANTHROPIC_API_KEY`, installing `anthropic`, and running `python src/agents/override.py "<phrase>"` writes a real response in their place; the UI changes nothing. Model: `claude-sonnet-5`.

Test phrases: `kal 80 puff ka order hai` (maps to croissant, confidence 0.55 — a fuzzy mapping a human should approve), `shaadi hai Saturday, thoda zyada` (no number, so no adjustment), `party order — 40 cream rolls Friday`, `school trip cancelled, kam bana lena`, `bread ka supply nahi aayega kal`.

Known gap: Approve repaints the bake row only. It does not re-solve the markdown schedule or the protected figure.

There is deliberately no LangGraph. With one node and no branches, a state graph would add ceremony and no behaviour.

## `src/core/cold_start.py` — a product with no history

A new item has no sales to forecast from. The planner borrows the demand shape of the most similar item and runs the same newsvendor on the new item's own economics:

1. **Most similar:** same category (bread/pastry), then the nearest price.
2. **Scale** the borrowed nine-point quantile curve so its median equals the owner's expected daily volume.
3. **Bake at τ\*** computed from the new item's price, cost and salvage.

No model is trained and no LLM is involved. Example, a Swiss cake (pastry, ₹120, cost ₹40, salvage ₹30, about 12 a day), run with `python src/core/cold_start.py`.

The app exposes this as a "new item" form that takes a name, category, price, cost, salvage and expected volume.

The page mirrors these two rules in about 12 lines of JavaScript because it has no server to ask. Python is canonical and self-checked; if one changes, change both.
