"""
watermark/detector.py
---------------------
Detects whether a piece of text carries the Aegis watermark.
Returns a z-score, p-value, and a human-readable verdict.

THE MATH
--------
Null hypothesis H₀: The text has NO watermark.
  → Word choices are random → ~50% should land on "green" synonyms by chance.

Alternative hypothesis H₁: The text IS watermarked.
  → The embedder biased choices toward green → significantly more than 50% green.

We use a one-sided z-test on the proportion of green choices:

  z = (observed_green - expected_green) / sqrt(n * p * (1-p))

Where:
  n = number of watermarkable words found
  p = key.green_fraction (default 0.5)
  observed_green = count of words that matched their green assignment

At z > 4.0, p < 0.00003 — strong forensic-grade evidence.

IMPORTANT: you need ~50+ watermarkable words for reliable detection.
A 200-word response typically has 30-60 eligible words.

KEY ROTATION
------------
Pass a KeyStore to try all key versions. Detection succeeds if any version
gives a significant z-score.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
import re
import random

from .key import WatermarkKey, KeyStore
from .embedder import SYNONYM_MAP, _REVERSE_MAP


# Z-score thresholds — tune based on your false-positive tolerance
Z_THRESHOLD_WEAK   = 1.5   # p ≈ 0.067 — suspicious, needs more text
Z_THRESHOLD_MEDIUM = 2.5   # p ≈ 0.006 — likely watermarked
Z_THRESHOLD_STRONG = 4.0   # p ≈ 0.00003 — forensic-grade evidence


@dataclass
class TokenAnalysis:
    """Analysis of a single watermarkable word found in the text."""
    position: int
    original_word: str
    canonical: str
    synonyms: list[str]
    context: str
    is_green: bool       # Did this word land on its green assignment?
    chosen_index: int    # Which synonym index was used


@dataclass
class DetectionResult:
    """
    Full detection report. This is your evidence artifact.

    If watermark_detected=True, this object contains everything needed to
    make a legal/technical case that a text was produced by (or derived from)
    a model protected by Aegis.
    """
    # Verdict
    watermark_detected: bool
    confidence: str            # "none" | "weak" | "medium" | "strong"
    key_version_matched: Optional[int]

    # Statistics
    z_score: float
    p_value: float
    green_count: int           # Words that matched green assignment
    total_eligible: int        # Total watermarkable words found
    green_fraction: float      # green_count / total_eligible

    # Evidence trail
    token_analyses: list[TokenAnalysis] = field(default_factory=list)

    # Human-readable summary
    summary: str = ""

    def to_report(self) -> str:
        """Generate a human-readable forensic report."""
        lines = [
            "=" * 60,
            "AEGIS WATERMARK DETECTION REPORT",
            "=" * 60,
            f"Verdict:          {'WATERMARK DETECTED' if self.watermark_detected else 'NO WATERMARK FOUND'}",
            f"Confidence:       {self.confidence.upper()}",
            f"Key version:      {self.key_version_matched or 'N/A'}",
            "",
            "STATISTICS",
            f"  Z-score:        {self.z_score:.4f}",
            f"  P-value:        {self.p_value:.6f}",
            f"  Green tokens:   {self.green_count} / {self.total_eligible}",
            f"  Green fraction: {self.green_fraction:.1%} (expected ~50%)",
            "",
            "INTERPRETATION",
        ]

        if self.watermark_detected:
            lines.append(
                f"  The text shows a statistically significant excess of 'green' "
                f"word choices (z={self.z_score:.2f}, p={self.p_value:.6f}). "
                f"This is consistent with the Aegis watermarking scheme using key v{self.key_version_matched}."
            )
            lines.append(
                f"  The probability of this occurring by chance is 1 in {int(1/max(self.p_value, 1e-10)):,}."
            )
        else:
            lines.append(
                f"  Insufficient signal to confirm watermark (z={self.z_score:.2f}). "
                f"This could mean: (a) text is not watermarked, (b) text is too short, "
                f"or (c) significant paraphrasing was applied."
            )

        lines += ["", "EVIDENCE SAMPLE (first 10 green tokens)"]
        green_tokens = [t for t in self.token_analyses if t.is_green][:10]
        for t in green_tokens:
            lines.append(f"  pos={t.position:4d}  '{t.original_word}' (canonical: '{t.canonical}')")

        lines.append("=" * 60)
        return "\n".join(lines)


class WatermarkDetector:
    """
    Detects the Aegis watermark in a piece of text.

    Usage (single key):
        key = WatermarkKey.from_hex("...")
        detector = WatermarkDetector(key)
        result = detector.detect("The model demonstrates significant results...")
        print(result.to_report())

    Usage (key rotation — checks all versions):
        store = KeyStore.from_env()
        detector = WatermarkDetector.from_keystore(store)
        result = detector.detect(text)
    """

    def __init__(self, key: WatermarkKey):
        self.key = key
        self._keystore: Optional[KeyStore] = None

    @classmethod
    def from_keystore(cls, store: KeyStore) -> "WatermarkDetector":
        """Create detector that tries all key versions."""
        detector = cls(store.latest())
        detector._keystore = store
        return detector

    def detect(self, text: str) -> DetectionResult:
        """
        Analyse text and return a full detection report.
        If a KeyStore was provided, tries all versions and returns
        the strongest signal found.
        """
        if self._keystore:
            best: Optional[DetectionResult] = None
            for version in self._keystore.all_versions():
                key = self._keystore.get(version)
                result = self._detect_with_key(text, key)
                if best is None or result.z_score > best.z_score:
                    best = result
            return best
        return self._detect_with_key(text, self.key)

    def _detect_with_key(self, text: str, key: WatermarkKey) -> DetectionResult:
        """Run detection with a specific key."""
        tokens = self._tokenize(text)
        analyses: list[TokenAnalysis] = []
        green_count = 0

        for i, token in enumerate(tokens):
            word_lower = token["word"].lower()
            canonical = _REVERSE_MAP.get(word_lower)

            if canonical is None or canonical not in SYNONYM_MAP:
                continue

            synonyms = SYNONYM_MAP[canonical]
            if len(synonyms) < 2:
                continue

            # Canonicalize context words — same logic as the embedder.
            # Synonyms in surrounding positions must be normalized back to their
            # canonical form so the HMAC seed matches what the embedder computed.
            context_words = [
                _REVERSE_MAP.get(t["word"].lower(), t["word"].lower())
                for t in tokens[max(0, i-3):i] + tokens[i+1:i+4]
                if t["is_word"]
            ]
            context = f"{canonical}|{'_'.join(context_words)}"

            # Replay the embedder's deterministic selection
            seed = key.derive_token_seed(context)
            green_synonym = synonyms[seed % len(synonyms)]

            # The word is "green" if it matches what our embedder would have chosen
            is_green = word_lower == green_synonym.lower()

            # Did the actual word in the text match the green choice?
            if is_green:
                green_count += 1

            # Find the chosen index for the report
            chosen_index = next(
                (j for j, s in enumerate(synonyms) if s.lower() == word_lower), 0
            )

            analyses.append(TokenAnalysis(
                position=i,
                original_word=token["word"],
                canonical=canonical,
                synonyms=synonyms,
                context=context,
                is_green=is_green,
                chosen_index=chosen_index,
            ))

        n = len(analyses)

        if n < 10:
            # Not enough data for reliable statistics
            return DetectionResult(
                watermark_detected=False,
                confidence="none",
                key_version_matched=None,
                z_score=0.0,
                p_value=1.0,
                green_count=green_count,
                total_eligible=n,
                green_fraction=green_count / n if n > 0 else 0.0,
                token_analyses=analyses,
                summary=f"Too few watermarkable tokens ({n} found, need ≥10).",
            )

        # One-sided z-test: H₀ p=green_fraction, H₁ p > green_fraction
        p_null = key.green_fraction
        z = (green_count - n * p_null) / math.sqrt(n * p_null * (1 - p_null))
        p_value = _z_to_p(z)

        # Determine confidence level
        if z >= Z_THRESHOLD_STRONG:
            confidence = "strong"
            detected = True
        elif z >= Z_THRESHOLD_MEDIUM:
            confidence = "medium"
            detected = True
        elif z >= Z_THRESHOLD_WEAK:
            confidence = "weak"
            detected = False   # Weak signal alone isn't enough to flag
        else:
            confidence = "none"
            detected = False

        observed_fraction = green_count / n

        return DetectionResult(
            watermark_detected=detected,
            confidence=confidence,
            key_version_matched=key.version if detected else None,
            z_score=z,
            p_value=p_value,
            green_count=green_count,
            total_eligible=n,
            green_fraction=observed_fraction,
            token_analyses=analyses,
            summary=(
                f"{'DETECTED' if detected else 'NOT DETECTED'} — "
                f"z={z:.2f}, p={p_value:.4f}, "
                f"{green_count}/{n} green ({observed_fraction:.0%})"
            ),
        )

    def _tokenize(self, text: str) -> list[dict]:
        tokens = []
        parts = re.split(r"(\b\w+\b)", text)
        for part in parts:
            if re.match(r"^\w+$", part):
                tokens.append({"word": part, "is_word": True})
            else:
                tokens.append({"word": part, "is_word": False})
        return tokens


def _z_to_p(z: float) -> float:
    """
    Convert a z-score to a one-sided p-value.
    Uses the complementary error function (erfc) for accuracy in the tails.
    """
    # P(Z > z) for standard normal using erfc
    return 0.5 * math.erfc(z / math.sqrt(2))
