"""writes the recommendation rationale in plain english.

important: this never does the math. the decision and every number are already
fixed by the constraint solver. the generator only phrases them, so a small
model (or the template fallback) is perfectly safe here.

uses flan-t5-base if transformers is installed, otherwise a deterministic
template. either way it gets the same structured facts.

todo: if we move to a bigger local model (llama-3 / mistral via ollama), keep
this same generate(context) interface so nothing else changes.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .router import ModelRouter


class RationaleGenerator:
    def __init__(self):
        self.backend = "template"
        self.model = "template"
        self._pipe = None
        try:
            from transformers import pipeline
            self._pipe = pipeline("text2text-generation", model="google/flan-t5-base")
            self.backend = "flan-t5"
            self.model = "flan-t5-base"
        except Exception:
            # no transformers (or model download blocked). template it is.
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def _template(self, ctx: Dict[str, Any]) -> str:
        drivers = " and ".join(ctx["drivers"]) or "its configuration"
        why = f" {ctx['why']}." if ctx.get("why") else ""
        return (f"{ctx['brand']}'s {ctx['competitor']} is gaining share at a stable "
                f"price on {drivers}.{why} It maps to your {ctx['own_sku']} "
                f"({ctx['own_model']}). Recommended: {ctx['action']}. {ctx['numbers']}. "
                f"This respects your constraints ({ctx['constraints']}).")

    def _prompt(self, ctx: Dict[str, Any]) -> str:
        drivers = " and ".join(ctx["drivers"])
        return (
            "Write a two-sentence recommendation for a product line manager. "
            "Be concrete and cite the numbers.\n"
            f"Competitor move: {ctx['brand']} {ctx['competitor']} is gaining at a "
            f"stable price on {drivers}.\n"
            f"Why buyers like it: {ctx.get('why', 'n/a')}.\n"
            f"Maps to our SKU: {ctx['own_sku']} ({ctx['own_model']}).\n"
            f"Decision: {ctx['action']} ({ctx['numbers']}).\n"
            f"Constraints respected: {ctx['constraints']}."
        )

    def generate(self, ctx: Dict[str, Any], router: ModelRouter) -> str:
        use = router.choose("rationale", "template", "flan-t5", self.available)
        t0 = time.time()
        if use == "flan-t5":
            out = self._pipe(self._prompt(ctx), max_new_tokens=90)[0]["generated_text"]
            text = out.strip()
        else:
            text = self._template(ctx)
        router.log("rationale", use, self.model if use == "flan-t5" else "template",
                   (time.time() - t0) * 1000)
        return text
