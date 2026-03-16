"""
watermark/embedder.py
---------------------
Injects an invisible cryptographic watermark into LLM text responses.

HOW IT WORKS
------------
English has thousands of near-synonym pairs:
  "however" ↔ "nevertheless", "but", "yet"
  "important" ↔ "significant", "crucial", "key"
  "shows" ↔ "demonstrates", "reveals", "indicates"

For each such "soft" word in the response, we have a choice: keep it or
swap it for a synonym. The HMAC of (secret_key + surrounding_context)
deterministically decides the choice — biasing toward "green list" synonyms.

Over a full response (~200+ words), this creates a statistically detectable
signal that survives copy-paste and minor edits, but is completely invisible
to any human reader. The text quality is unchanged.

PROXY-FRIENDLY
--------------
Unlike logit-level watermarking (which requires access to model internals),
this works entirely on the output text — perfect for a proxy like Aegis.
"""

import re
from dataclasses import dataclass
from typing import Optional
from .key import WatermarkKey


# ---------------------------------------------------------------------------
# Synonym map — the larger this is, the stronger the watermark signal.
# Each entry: canonical_word → [synonym_0, synonym_1, synonym_2, ...]
# We need at least 2 choices to make a binary green/red decision.
#
# Design rules:
#   - All synonyms must be genuinely interchangeable in context
#   - Include the original word as one option (so ~50% of words stay unchanged)
#   - Prefer common words — rare synonyms look odd
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, list[str]] = {
    # Connectives
    "however":       ["however", "nevertheless", "yet", "that said"],
    "therefore":     ["therefore", "thus", "hence", "consequently"],
    "furthermore":   ["furthermore", "moreover", "additionally", "also"],
    "although":      ["although", "though", "even though", "while"],
    "because":       ["because", "since", "as", "given that"],
    "but":           ["but", "however", "yet", "though"],
    "also":          ["also", "additionally", "as well", "too", "furthermore"],
    "so":            ["so", "therefore", "thus", "hence"],
    "while":         ["while", "whereas", "although", "even as"],
    "when":          ["when", "once", "as soon as", "whenever"],

    # Degree words
    "very":          ["very", "quite", "rather", "fairly", "particularly"],
    "extremely":     ["extremely", "highly", "exceptionally", "remarkably"],
    "often":         ["often", "frequently", "commonly", "regularly"],
    "sometimes":     ["sometimes", "occasionally", "at times", "periodically"],
    "always":        ["always", "consistently", "invariably", "continually"],
    "never":         ["never", "not once", "at no point"],
    "mostly":        ["mostly", "largely", "primarily", "mainly", "chiefly"],
    "generally":     ["generally", "typically", "usually", "normally", "broadly"],
    "particularly":  ["particularly", "especially", "specifically", "notably"],

    # Verbs
    "shows":         ["shows", "demonstrates", "reveals", "indicates", "illustrates"],
    "uses":          ["uses", "utilizes", "employs", "leverages", "applies"],
    "helps":         ["helps", "assists", "aids", "supports", "enables"],
    "allows":        ["allows", "enables", "permits", "lets", "facilitates"],
    "requires":      ["requires", "needs", "demands", "necessitates"],
    "provides":      ["provides", "offers", "delivers", "supplies", "gives"],
    "includes":      ["includes", "contains", "encompasses", "comprises", "covers"],
    "ensures":       ["ensures", "guarantees", "assures", "confirms"],
    "creates":       ["creates", "generates", "produces", "builds", "constructs"],
    "improves":      ["improves", "enhances", "strengthens", "boosts", "optimizes"],
    "reduces":       ["reduces", "decreases", "lowers", "minimizes", "limits"],
    "increases":     ["increases", "raises", "boosts", "elevates", "amplifies"],
    "makes":         ["makes", "renders", "turns", "causes"],
    "means":         ["means", "implies", "indicates", "suggests", "signifies"],
    "works":         ["works", "functions", "operates", "performs", "runs"],
    "needs":         ["needs", "requires", "demands", "calls for"],
    "gets":          ["gets", "obtains", "receives", "acquires", "gains"],
    "gives":         ["gives", "provides", "offers", "supplies", "delivers"],
    "takes":         ["takes", "requires", "needs", "demands"],
    "keeps":         ["keeps", "maintains", "retains", "preserves"],
    "runs":          ["runs", "executes", "operates", "functions", "performs"],
    "calls":         ["calls", "invokes", "triggers", "initiates"],
    "handles":       ["handles", "manages", "processes", "deals with"],
    "stores":        ["stores", "saves", "persists", "retains", "holds"],
    "checks":        ["checks", "verifies", "validates", "confirms", "tests"],
    "returns":       ["returns", "yields", "gives back", "produces", "outputs"],
    "supports":      ["supports", "enables", "facilitates", "assists", "backs"],

    # Adjectives
    "important":     ["important", "significant", "crucial", "key", "essential"],
    "useful":        ["useful", "helpful", "valuable", "practical", "beneficial"],
    "simple":        ["simple", "straightforward", "easy", "basic", "clear"],
    "complex":       ["complex", "complicated", "intricate", "sophisticated"],
    "different":     ["different", "distinct", "varied", "diverse", "various"],
    "specific":      ["specific", "particular", "precise", "exact", "certain"],
    "common":        ["common", "frequent", "typical", "standard", "usual"],
    "large":         ["large", "significant", "substantial", "considerable", "major"],
    "small":         ["small", "minor", "minimal", "limited", "slight"],
    "new":           ["new", "novel", "fresh", "modern", "recent"],
    "main":          ["main", "primary", "principal", "central", "core"],
    "key":           ["key", "critical", "essential", "fundamental", "vital"],
    "good":          ["good", "effective", "solid", "strong", "sound"],
    "better":        ["better", "superior", "improved", "more effective"],
    "best":          ["best", "optimal", "ideal", "most effective"],
    "fast":          ["fast", "quick", "rapid", "swift", "efficient"],
    "easy":          ["easy", "simple", "straightforward", "effortless"],
    "clear":         ["clear", "obvious", "evident", "apparent", "plain"],
    "full":          ["full", "complete", "entire", "comprehensive", "total"],
    "real":          ["real", "actual", "genuine", "true", "authentic"],
    "similar":       ["similar", "comparable", "equivalent", "analogous", "like"],
    "available":     ["available", "accessible", "obtainable", "usable"],
    "possible":      ["possible", "feasible", "achievable", "viable", "doable"],
    "necessary":     ["necessary", "required", "essential", "needed", "mandatory"],
    "correct":       ["correct", "accurate", "proper", "right", "valid"],
    "current":       ["current", "present", "existing", "active", "live"],
    "previous":      ["previous", "prior", "earlier", "former", "past"],
    "original":      ["original", "initial", "first", "base", "source"],
    "multiple":      ["multiple", "several", "various", "numerous", "many"],
    "single":        ["single", "individual", "one", "sole", "lone"],
    "additional":    ["additional", "extra", "further", "more", "supplementary"],

    # Nouns
    "approach":      ["approach", "method", "technique", "strategy", "way"],
    "issue":         ["issue", "problem", "challenge", "concern", "matter"],
    "result":        ["result", "outcome", "output", "effect", "consequence"],
    "example":       ["example", "instance", "case", "illustration"],
    "feature":       ["feature", "characteristic", "property", "attribute", "aspect"],
    "process":       ["process", "procedure", "workflow", "mechanism", "operation"],
    "system":        ["system", "framework", "structure", "platform", "setup"],
    "data":          ["data", "information", "details", "records", "values"],
    "way":           ["way", "method", "approach", "manner", "means"],
    "part":          ["part", "component", "element", "piece", "section"],
    "type":          ["type", "kind", "sort", "category", "form"],
    "case":          ["case", "instance", "scenario", "situation", "example"],
    "value":         ["value", "benefit", "advantage", "merit", "worth"],
    "level":         ["level", "degree", "extent", "amount", "measure"],
    "number":        ["number", "count", "quantity", "amount", "total"],
    "point":         ["point", "aspect", "detail", "element", "factor"],
    "step":          ["step", "stage", "phase", "action", "move"],
    "set":           ["set", "collection", "group", "series", "range"],
    "list":          ["list", "collection", "set", "series", "array"],
    "version":       ["version", "variant", "edition", "release", "iteration"],
    "option":        ["option", "choice", "alternative", "possibility"],
    "request":       ["request", "query", "call", "invocation", "message"],
    "response":      ["response", "reply", "output", "answer", "result"],
    "error":         ["error", "fault", "failure", "issue", "problem"],
    "change":        ["change", "modification", "update", "alteration", "revision"],
    "access":        ["access", "entry", "permission", "reach", "availability"],
    "impact":        ["impact", "effect", "influence", "consequence", "result"],
    "goal":          ["goal", "objective", "aim", "target", "purpose"],
    "task":          ["task", "job", "operation", "activity", "action"],
    "user":          ["user", "caller", "client", "consumer", "requester"],

    # Adverbs
    "quickly":       ["quickly", "rapidly", "swiftly", "promptly", "efficiently"],
    "easily":        ["easily", "readily", "simply", "straightforwardly"],
    "clearly":       ["clearly", "evidently", "obviously", "plainly"],
    "directly":      ["directly", "immediately", "straight", "explicitly"],
    "simply":        ["simply", "just", "merely", "only", "purely"],
    "currently":     ["currently", "presently", "now", "at present", "today"],
    "typically":     ["typically", "generally", "usually", "normally", "commonly"],
    "effectively":   ["effectively", "efficiently", "successfully", "well"],
    "automatically": ["automatically", "programmatically", "dynamically"],
    "properly":      ["properly", "correctly", "accurately", "appropriately"],
    "efficiently":   ["efficiently", "effectively", "optimally", "well"],
    "significantly": ["significantly", "substantially", "considerably", "notably"],
    "essentially":   ["essentially", "fundamentally", "basically", "primarily"],
    "actually":      ["actually", "in fact", "indeed", "in practice"],
    "already":       ["already", "previously", "beforehand", "prior to this"],
}

# Build a reverse lookup: synonym → canonical word
# e.g. "demonstrates" → "shows"
_REVERSE_MAP: dict[str, str] = {}
for _canonical, _synonyms in SYNONYM_MAP.items():
    for _syn in _synonyms:
        _REVERSE_MAP[_syn.lower()] = _canonical


@dataclass
class EmbedResult:
    original_text: str
    watermarked_text: str
    substitutions_made: int    # How many swaps were performed
    eligible_words: int        # How many words were candidates
    key_version: int


class WatermarkEmbedder:
    """
    Embeds a cryptographic watermark into LLM response text.

    Usage:
        key = WatermarkKey.generate()
        embedder = WatermarkEmbedder(key)
        result = embedder.embed("The model shows important results...")
        # → "The model demonstrates significant results..."  (or not — key decides)
    """

    def __init__(self, key: WatermarkKey):
        self.key = key

    def embed(self, text: str) -> EmbedResult:
        """
        Watermark a piece of text.
        Returns the (possibly modified) text + metadata.
        """
        tokens = self._tokenize(text)
        output_tokens = []
        substitutions = 0
        eligible = 0

        for i, token in enumerate(tokens):
            word_lower = token["word"].lower()
            canonical = _REVERSE_MAP.get(word_lower)

            if canonical is None or canonical not in SYNONYM_MAP:
                # Not a watermarkable word — keep as-is
                output_tokens.append(token["original"])
                continue

            synonyms = SYNONYM_MAP[canonical]
            eligible += 1

            # Build context using CANONICAL forms — critical for embedder/detector sync.
            # If a previous word was already swapped (e.g. "leverages" replacing "uses"),
            # we normalize it back to its canonical form so both sides always hash the same string.
            context_words = [
                _REVERSE_MAP.get(t["word"].lower(), t["word"].lower())
                for t in tokens[max(0, i-3):i] + tokens[i+1:i+4]
                if t["is_word"]
            ]
            context = f"{canonical}|{'_'.join(context_words)}"

            # Deterministically select the ONE green synonym for this position
            # using HMAC(key, context) — this is the watermark signal
            seed = self.key.derive_token_seed(context)
            green_synonym = synonyms[seed % len(synonyms)]

            # ALWAYS use the green synonym — this is what creates the detectable signal
            output_word = self._match_case(token["word"], green_synonym)

            if output_word.lower() != token["word"].lower():
                substitutions += 1

            output_tokens.append(
                token["original"].replace(token["word"], output_word, 1)
            )

        watermarked = "".join(output_tokens)

        return EmbedResult(
            original_text=text,
            watermarked_text=watermarked,
            substitutions_made=substitutions,
            eligible_words=eligible,
            key_version=self.key.version,
        )

    def _tokenize(self, text: str) -> list[dict]:
        """
        Split text into tokens preserving whitespace and punctuation.
        Each token is {"original": str, "word": str, "is_word": bool}
        """
        tokens = []
        # Split on word boundaries, keeping delimiters
        parts = re.split(r"(\b\w+\b)", text)
        for part in parts:
            if re.match(r"^\w+$", part):
                tokens.append({"original": part, "word": part, "is_word": True})
            else:
                tokens.append({"original": part, "word": part, "is_word": False})
        return tokens

    def _match_case(self, original: str, replacement: str) -> str:
        """Preserve the capitalisation of the original word."""
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
