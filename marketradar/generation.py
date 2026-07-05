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
from functools import lru_cache
from typing import Any, Dict

from .router import ModelRouter


@lru_cache(maxsize=1)
def _load_flan_t5():
    """Load flan-t5-base (tokenizer + model) once per process and reuse it.

    The ~1gb weights are the single biggest cost in quality mode, so caching
    them here is what turns a repeated multi-second stall into a one-time load."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl


class RationaleGenerator:
    def __init__(self):
        self.model = "flan-t5-base"
        self._tok = None
        self._mdl = None
        # only check that transformers imports here. the ~1gb model download is
        # deferred to first use, so cheap mode never pays for a model it won't run.
        try:
            import transformers  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_model(self):
        # flan-t5 is seq2seq. transformers 5.x dropped the text2text pipeline
        # task, so we load the model directly and call generate ourselves.
        if self._mdl is None:
            self._tok, self._mdl = _load_flan_t5()
        return self._tok, self._mdl

    def _template(self, ctx: Dict[str, Any]) -> str:
        drivers = " and ".join(ctx["drivers"]) or "its configuration"
        why = f" {ctx['why']}." if ctx.get("why") else ""
        return (f"{ctx['brand']}'s {ctx['competitor']} is gaining share at a stable "
                f"price on {drivers}.{why} It maps to your {ctx['own_sku']} "
                f"({ctx['own_model']}). Recommended: {ctx['action']}. {ctx['numbers']}. "
                f"This respects your constraints ({ctx['constraints']}).")

    def _prompt(self, ctx: Dict[str, Any]) -> str:
        # flowing text with no "Label:" markers, so flan-t5 rephrases instead of
        # echoing the labels back.
        drivers = " and ".join(ctx["drivers"])
        why = f" {ctx['why'][0].upper() + ctx['why'][1:]}." if ctx.get("why") else ""
        return (
            "Rewrite the following as one concise recommendation for a product "
            f"manager. {ctx['brand']}'s {ctx['competitor']} is gaining share at a "
            f"stable price on {drivers}.{why} It matches our {ctx['own_sku']}, so we "
            f"should {ctx['action'].lower()}, which keeps us within {ctx['numbers']} "
            f"and {ctx['constraints']}.")

    def generate(self, ctx: Dict[str, Any], router: ModelRouter) -> str:
        use = router.choose("rationale", "template", "flan-t5", self.available)
        t0 = time.time()
        if use == "flan-t5":
            try:
                tok, mdl = self._ensure_model()
                ids = tok(self._prompt(ctx), return_tensors="pt",
                          truncation=True, max_length=512)
                out = mdl.generate(**ids, max_new_tokens=140)
                text = tok.decode(out[0], skip_special_tokens=True).strip()
                if not text:                       # guard against an empty decode
                    use, text = "template", self._template(ctx)
            except Exception:
                # download or inference failed. fall back so we still answer.
                use, text = "template", self._template(ctx)
        else:
            text = self._template(ctx)
        router.log("rationale", use, "flan-t5-base" if use == "flan-t5" else "template",
                   (time.time() - t0) * 1000)
        return text
