"""
tests/test_watermark.py
-----------------------
Full test suite for the Aegis watermarking system.

Run with:  python tests/test_watermark.py

Tests cover:
  - Key generation and derivation
  - Embedding: text quality preserved, substitutions made
  - Embedding: deterministic (same key → same output)
  - Embedding: different keys → different outputs
  - Detection: watermarked text is detected
  - Detection: unwatermarked text is NOT detected (no false positives)
  - Detection: key rotation works
  - Detection: partial paraphrasing still detects
  - Detection: too-short text handled gracefully
  - End-to-end: full pipeline
"""

import sys
import os
import random

# Add parent directory to path so we can import the watermark package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark.key import WatermarkKey, KeyStore
from watermark.embedder import WatermarkEmbedder, SYNONYM_MAP
from watermark.detector import WatermarkDetector


# ── Test helpers ────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            msg = f"  ❌ {name}"
            if detail:
                msg += f"\n     {detail}"
            print(msg)
            self.errors.append(name)

    def section(self, title: str):
        print(f"\n── {title} {'─' * (50 - len(title))}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 55}")
        print(f"Results: {self.passed} passed, {self.failed} failed / {total} total")
        if self.failed == 0:
            print("🛡️  All watermark tests passing")
        else:
            print(f"Failed tests: {', '.join(self.errors)}")
        print('=' * 55)
        return self.failed == 0


# ── Sample texts ─────────────────────────────────────────────────────────────

SHORT_TEXT = "This is a simple test."

MEDIUM_TEXT = """
The system uses a key to generate watermarks. This approach is very important
because it ensures that the output is always consistent. The process requires
a specific algorithm that provides reliable results. However, the method also
allows for key rotation, which is useful when security is a primary concern.
The data shows that this technique creates a significant improvement over
previous approaches. Furthermore, the system reduces the risk of unauthorized
access while it increases the overall security of the platform.
"""

LONG_TEXT = """
Cryptographic watermarking is a key technique in modern AI security. The main
approach uses an HMAC function to generate a specific seed for each token
position. This seed currently determines which synonym to use from a list of
common alternatives. The system ensures that the process is always deterministic
given the same secret key.

Furthermore, the method allows for multiple key versions, which is very useful
for key rotation. When a new key is required, the old key remains in the
keystore so that previously watermarked text can still be detected. This
feature is particularly important for long-term forensic evidence.

The detection process uses a simple z-test to determine whether a specific
text shows a statistically significant excess of green token choices. The
result provides a p-value that clearly indicates the confidence level. When
the z-score is extremely high, it effectively demonstrates that the text
was watermarked using the Aegis system.

However, the approach sometimes requires a large amount of text to produce
reliable results. Very short texts typically do not include enough watermarkable
tokens for the statistical test to be significant. The system reduces false
positives by requiring at least ten eligible tokens before it provides a
confident verdict.
"""

HUMAN_TEXT = """
I think the weather today is really quite lovely. My dog keeps running around
the garden chasing butterflies, which always makes me laugh. Later I plan to
cook pasta for dinner — probably something with tomatoes and garlic. It should
be delicious. Tomorrow I have a meeting at nine in the morning, which is early
but manageable. I've been reading a fascinating book about the history of Rome
lately, and I can't seem to put it down.
"""


# ── Tests ────────────────────────────────────────────────────────────────────

def test_key_generation(t: TestRunner):
    t.section("Key Generation")

    k1 = WatermarkKey.generate()
    k2 = WatermarkKey.generate()
    t.check("Keys are 32 bytes", len(k1.secret) == 32)
    t.check("Two generated keys are different", k1.secret != k2.secret)
    t.check("Key version defaults to 1", k1.version == 1)

    hex_str = k1.to_hex()
    k_restored = WatermarkKey.from_hex(hex_str)
    t.check("Key survives hex round-trip", k_restored.secret == k1.secret)


def test_hmac_derivation(t: TestRunner):
    t.section("HMAC Derivation")

    key = WatermarkKey.from_hex("a" * 64, version=1)

    seed1 = key.derive_token_seed("shows|the_model_clear")
    seed2 = key.derive_token_seed("shows|the_model_clear")
    seed3 = key.derive_token_seed("shows|different_context")

    t.check("Same context → same seed (deterministic)", seed1 == seed2)
    t.check("Different context → different seed", seed1 != seed3)
    t.check("Seed is an integer", isinstance(seed1, int))


def test_embedding_basic(t: TestRunner):
    t.section("Embedding — Basic")

    key = WatermarkKey.generate()
    embedder = WatermarkEmbedder(key)
    result = embedder.embed(MEDIUM_TEXT)

    t.check(
        "Watermarked text is a non-empty string",
        isinstance(result.watermarked_text, str) and len(result.watermarked_text) > 0
    )
    t.check(
        "Some substitutions were made",
        result.substitutions_made > 0,
        f"Got {result.substitutions_made} substitutions"
    )
    t.check(
        "Eligible words found",
        result.eligible_words > 0,
        f"Got {result.eligible_words} eligible words"
    )
    t.check(
        "Key version recorded",
        result.key_version == key.version
    )
    # Text quality: length should be similar (synonyms are similar length words)
    len_ratio = len(result.watermarked_text) / len(MEDIUM_TEXT)
    t.check(
        "Text length preserved (within 30%)",
        0.7 < len_ratio < 1.3,
        f"Length ratio: {len_ratio:.2f}"
    )


def test_embedding_deterministic(t: TestRunner):
    t.section("Embedding — Determinism")

    key = WatermarkKey.from_hex("b" * 64)
    embedder = WatermarkEmbedder(key)

    result1 = embedder.embed(LONG_TEXT)
    result2 = embedder.embed(LONG_TEXT)

    t.check(
        "Same key + same text → identical output",
        result1.watermarked_text == result2.watermarked_text
    )
    t.check(
        "Same substitution count both times",
        result1.substitutions_made == result2.substitutions_made
    )


def test_embedding_key_sensitivity(t: TestRunner):
    t.section("Embedding — Key Sensitivity")

    key1 = WatermarkKey.from_hex("c" * 64)
    key2 = WatermarkKey.from_hex("d" * 64)

    result1 = WatermarkEmbedder(key1).embed(LONG_TEXT)
    result2 = WatermarkEmbedder(key2).embed(LONG_TEXT)

    t.check(
        "Different keys produce different watermarked text",
        result1.watermarked_text != result2.watermarked_text,
        "Two different keys produced identical output — very unlikely if working correctly"
    )


def test_detection_positive(t: TestRunner):
    t.section("Detection — True Positives")

    key = WatermarkKey.generate()
    embedder = WatermarkEmbedder(key)
    detector = WatermarkDetector(key)

    result_embed = embedder.embed(LONG_TEXT)
    result_detect = detector.detect(result_embed.watermarked_text)

    t.check(
        "Watermarked long text is detected",
        result_detect.watermark_detected,
        f"z={result_detect.z_score:.2f}, p={result_detect.p_value:.4f}, "
        f"{result_detect.green_count}/{result_detect.total_eligible} green"
    )
    t.check(
        "Z-score is positive",
        result_detect.z_score > 0,
        f"z={result_detect.z_score:.2f}"
    )
    t.check(
        "Green fraction is above 50%",
        result_detect.green_fraction > 0.5,
        f"green fraction = {result_detect.green_fraction:.1%}"
    )
    t.check(
        "Confidence is medium or strong",
        result_detect.confidence in ("medium", "strong"),
        f"Got confidence='{result_detect.confidence}'"
    )


def test_detection_negative(t: TestRunner):
    t.section("Detection — False Positive Rate")

    key = WatermarkKey.generate()
    detector = WatermarkDetector(key)

    # Human-written text should NOT trigger detection
    result = detector.detect(HUMAN_TEXT)
    t.check(
        "Natural human text is not flagged",
        not result.watermark_detected,
        f"z={result.z_score:.2f}, p={result.p_value:.4f}"
    )

    # AI text NOT watermarked with our key should not be detected
    # Simulate by watermarking with a DIFFERENT key
    other_key = WatermarkKey.generate()
    other_embedder = WatermarkEmbedder(other_key)
    other_result = other_embedder.embed(LONG_TEXT)

    result2 = detector.detect(other_result.watermarked_text)
    t.check(
        "Text watermarked with different key is not detected",
        not result2.watermark_detected,
        f"z={result2.z_score:.2f}"
    )


def test_detection_short_text(t: TestRunner):
    t.section("Detection — Short Text Handling")

    key = WatermarkKey.generate()
    embedder = WatermarkEmbedder(key)
    detector = WatermarkDetector(key)

    result = detector.detect(SHORT_TEXT)
    t.check(
        "Short text handled gracefully (no crash)",
        True  # If we got here, no exception was raised
    )
    t.check(
        "Short text returns not-detected (insufficient data)",
        not result.watermark_detected,
        f"Got detected=True on {len(SHORT_TEXT)}-char text"
    )
    t.check(
        "Short text result has a summary message",
        len(result.summary) > 0
    )


def test_key_rotation(t: TestRunner):
    t.section("Key Rotation")

    key_v1 = WatermarkKey.from_hex("e" * 64, version=1)
    key_v2 = WatermarkKey.from_hex("f" * 64, version=2)

    # Watermark text with v1
    embedder_v1 = WatermarkEmbedder(key_v1)
    watermarked_v1 = embedder_v1.embed(LONG_TEXT).watermarked_text

    # Watermark different text with v2
    embedder_v2 = WatermarkEmbedder(key_v2)
    watermarked_v2 = embedder_v2.embed(LONG_TEXT).watermarked_text

    # Build a keystore with both versions
    store = KeyStore()
    store.add(key_v1)
    store.add(key_v2)
    detector = WatermarkDetector.from_keystore(store)

    result_v1 = detector.detect(watermarked_v1)
    result_v2 = detector.detect(watermarked_v2)

    t.check(
        "KeyStore detects v1-watermarked text",
        result_v1.watermark_detected,
        f"z={result_v1.z_score:.2f}"
    )
    t.check(
        "KeyStore detects v2-watermarked text",
        result_v2.watermark_detected,
        f"z={result_v2.z_score:.2f}"
    )
    t.check(
        "Store has 2 key versions",
        len(store.all_versions()) == 2
    )
    t.check(
        "Latest key is v2",
        store.latest().version == 2
    )


def test_partial_paraphrase_resilience(t: TestRunner):
    t.section("Paraphrase Resilience")

    key = WatermarkKey.generate()
    embedder = WatermarkEmbedder(key)
    detector = WatermarkDetector(key)

    watermarked = embedder.embed(LONG_TEXT).watermarked_text

    # Simulate ~20% of words being changed by an attacker
    words = watermarked.split()
    num_to_change = len(words) // 5
    indices = random.sample(range(len(words)), num_to_change)
    for i in indices:
        words[i] = words[i][::-1]  # simple corruption (reverse the word)
    corrupted = " ".join(words)

    result = detector.detect(corrupted)
    # We can't guarantee detection after heavy corruption, but z-score should still be elevated
    t.check(
        "Z-score is still positive after 20% word corruption",
        result.z_score > 0,
        f"z={result.z_score:.2f}"
    )


def test_detection_report(t: TestRunner):
    t.section("Detection Report")

    key = WatermarkKey.generate()
    embedder = WatermarkEmbedder(key)
    detector = WatermarkDetector(key)

    result = detector.detect(embedder.embed(LONG_TEXT).watermarked_text)
    report = result.to_report()

    t.check("Report is a non-empty string", len(report) > 100)
    t.check("Report contains z-score", "Z-score" in report)
    t.check("Report contains p-value", "P-value" in report)
    t.check("Report contains verdict", "WATERMARK" in report)


def test_end_to_end(t: TestRunner):
    t.section("End-to-End Pipeline")

    # Simulate the full Aegis flow:
    # 1. Key is generated once at startup
    key = WatermarkKey.generate()

    # 2. Every LLM response passes through the embedder
    embedder = WatermarkEmbedder(key)
    llm_response = LONG_TEXT
    protected_response = embedder.embed(llm_response).watermarked_text

    # 3. Months later, a suspect clone model appears
    # We sample 5 outputs from it
    clone_outputs = [protected_response] * 5  # In reality these would vary

    # 4. Detector checks if outputs carry our watermark
    detector = WatermarkDetector(key)
    detections = [detector.detect(output) for output in clone_outputs]

    any_detected = any(d.watermark_detected for d in detections)
    t.check(
        "At least one clone output detected as watermarked",
        any_detected,
        f"Scores: {[round(d.z_score,2) for d in detections]}"
    )

    # Also confirm that another company's outputs don't trigger false positives
    other_key = WatermarkKey.generate()
    other_embedder = WatermarkEmbedder(other_key)
    third_party_output = other_embedder.embed(LONG_TEXT).watermarked_text

    result = detector.detect(third_party_output)
    t.check(
        "Third-party model output is not falsely flagged",
        not result.watermark_detected,
        f"z={result.z_score:.2f}"
    )

    print(f"\n  Sample report from clone detection:\n")
    best = max(detections, key=lambda d: d.z_score)
    for line in best.to_report().split("\n")[:20]:
        print(f"  {line}")
    print("  ...")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = TestRunner()

    test_key_generation(t)
    test_hmac_derivation(t)
    test_embedding_basic(t)
    test_embedding_deterministic(t)
    test_embedding_key_sensitivity(t)
    test_detection_positive(t)
    test_detection_negative(t)
    test_detection_short_text(t)
    test_key_rotation(t)
    test_partial_paraphrase_resilience(t)
    test_detection_report(t)
    test_end_to_end(t)

    success = t.summary()
    sys.exit(0 if success else 1)
