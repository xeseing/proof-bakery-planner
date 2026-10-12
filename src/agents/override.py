"""Phase 7 (scoped): turn a baker's message into a proposed bake adjustment.

The model only proposes. The number the baker acts on comes out of
apply_adjustment, which is arithmetic — no prompt is anywhere near it.

    py -3.13 src/agents/override.py                      self-check, seed, export
    py -3.13 src/agents/override.py "kal 80 puff ka order hai"   parse one message

Every response is cached under .cache/override/ keyed on the prompt hash, and
exported to frontend/overrides.js so the demo runs with the wifi off.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/override"
MODEL = "claude-sonnet-5"

# Verbatim from SPECS 7.4. {item_list} is substituted, not .format()ed: the prompt
# is full of JSON braces.
PROMPT = """You convert a bakery owner's message into a structured demand
adjustment. Messages are often code-mixed Hindi and English, informal
and abbreviated.

You are given the items the bakery plans to bake. Map the message to ONE
of those exact item names. If it maps to none, return delta_units 0 and
confidence 0.

Rules:
- delta_units is ADDITIONAL units on top of the existing plan, not a
  replacement total, unless the message clearly states a replacement.
- If no number is stated, do not guess. Return delta_units 0, put what
  you understood in reason, confidence below 0.5.
- Never adjust more than one item per message. If several are mentioned,
  take the one with an explicit number.
- You propose. A human approves. Never phrase output as a completed
  action.

Available items: {item_list}

Output JSON only:
{"item":"...","delta_units":0,"reason":"...","confidence":0.0}

"kal 80 puff ka order hai"
 -> {"item":"puff","delta_units":80,
     "reason":"Catering order for 80 puffs tomorrow","confidence":0.93}
"shaadi hai Saturday, thoda zyada bana lena"
 -> {"item":"cake_slice","delta_units":0,
     "reason":"Wedding Saturday, quantity not specified","confidence":0.4}

On failure return
{"item":"","delta_units":0,"reason":"could not parse","confidence":0.0}

Message: """

# No API key here tonight, so these three stand in for model output. Running this
# file with a phrase and a key replaces them with the real thing; nothing else changes.
# The card shows which of the two it is.
SEEDS = {
    "kal 20 croissant ka order hai": {                       # the demo phrase
        "item": "CROISSANT", "delta_units": 20, "confidence": 0.95,
        "reason": "Catering order for 20 croissants tomorrow"},
    "kal 80 puff ka order hai": {                            # the alias still resolves
        "item": "CROISSANT", "delta_units": 80, "confidence": 0.55,
        "reason": "Catering order for 80 puffs tomorrow; nearest item on this menu is croissant"},
    "shaadi hai Saturday, thoda zyada bana lena": {          # a number nobody stated
        "item": "", "delta_units": 0, "confidence": 0.4,
        "reason": "Wedding Saturday, quantity not specified"},
}
FALLBACK = "kal 20 croissant ka order hai"  # what the UI shows when a message cannot be parsed


class ProposedAdjustment(BaseModel):
    item: str
    delta_units: int
    reason: str
    confidence: float = Field(ge=0, le=1)


def apply_adjustment(bake_list, item, delta_units):
    """Add delta to one item's quantity. Deterministic, and never below zero."""
    return [b if b["item"] != item else {**b, "quantity": max(0, b["quantity"] + delta_units)}
            for b in bake_list]


def normalise(text):
    """Loose key for looking a message up: case, spacing and punctuation don't matter."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def build_prompt(text, items):
    return PROMPT.replace("{item_list}", ", ".join(items)) + text


def ask_model(prompt):
    """The one network call in the whole project."""
    from anthropic import Anthropic  # imported late: the cached demo never needs it
    reply = Anthropic().messages.create(model=MODEL, max_tokens=300,
                                        messages=[{"role": "user", "content": prompt}])
    return reply.content[0].text


def parse_override(text, items, allow_network=True):
    """Cache first, model second, documented failure third.

    Returns (ProposedAdjustment, degraded) where degraded names this node if the
    proposal did not come from a real answer.
    """
    prompt = build_prompt(text, items)
    cached = CACHE / f"{hashlib.sha1(prompt.encode()).hexdigest()[:16]}.json"
    if cached.exists():
        return ProposedAdjustment(**json.loads(cached.read_text())["proposal"]), None
    if not allow_network:
        return ProposedAdjustment(item="", delta_units=0, reason="could not parse",
                                  confidence=0.0), "override_parse"
    try:
        proposal = ProposedAdjustment.model_validate_json(ask_model(prompt))
    except Exception as e:                      # no key, no network, or unusable JSON
        print(f"[degraded] override_parse: {e}")
        return ProposedAdjustment(item="", delta_units=0, reason="could not parse",
                                  confidence=0.0), "override_parse"
    write_cache(text, prompt, proposal.model_dump(), MODEL)
    return proposal, None


def write_cache(text, prompt, proposal, source):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(prompt.encode()).hexdigest()[:16]
    (CACHE / f"{key}.json").write_text(json.dumps(
        {"text": text, "source": source, "proposal": proposal}, indent=2))


def export(items):
    """Cache -> frontend/overrides.js, so app.html can answer offline from a file://."""
    phrases = {}
    for hit in sorted(CACHE.glob("*.json")):
        entry = json.loads(hit.read_text())
        phrases[normalise(entry["text"])] = {**entry["proposal"], "source": entry["source"]}
    out = {"phrases": phrases, "fallback": normalise(FALLBACK)}
    (ROOT / "frontend/overrides.js").write_text(f"window.OVERRIDES = {json.dumps(out, indent=2)};\n")
    return phrases


def demo():
    plan = [{"item": "CROISSANT", "quantity": 51}, {"item": "BAGUETTE", "quantity": 36}]
    assert apply_adjustment(plan, "CROISSANT", 80)[0]["quantity"] == 131
    assert apply_adjustment(plan, "CROISSANT", -99)[0]["quantity"] == 0        # never negative
    assert apply_adjustment(plan, "TARTELETTE", 5) == plan                     # unknown item: no-op
    assert apply_adjustment(plan, "CROISSANT", 80)[1] == plan[1]               # touches one item
    assert normalise("Kal 80 PUFF ka order hai!") == "kal 80 puff ka order hai"
    try:
        ProposedAdjustment(item="X", delta_units=1, reason="r", confidence=1.4)
        raise AssertionError("confidence above 1 should not validate")
    except ValidationError:
        pass
    print("self-check ok")


if __name__ == "__main__":
    demo()
    items = [b["item"] for b in json.loads((ROOT / "output/plan.json").read_text())["bake_list"]]
    for text, proposal in SEEDS.items():                 # only fills what is not cached already
        prompt = build_prompt(text, items)
        if not (CACHE / f"{hashlib.sha1(prompt.encode()).hexdigest()[:16]}.json").exists():
            write_cache(text, prompt, proposal, "seeded")
    if len(sys.argv) > 1:
        proposal, degraded = parse_override(sys.argv[1], items)
        print(f"{proposal.model_dump()}  degraded={degraded}")
    print(f"{len(export(items))} phrases -> frontend/overrides.js")
