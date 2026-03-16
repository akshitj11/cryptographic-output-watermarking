# Aegis Watermark

Cryptographic output watermarking for LLM APIs.

## What it does

Every LLM response passing through Aegis gets an invisible statistical signature
embedded into it via synonym substitution. If a competitor trains a clone model
on stolen outputs, their model inherits the signature — and you can detect it
with forensic-grade statistical proof.

## How it works

1. **Embed** — For each "soft" word in a response (words with near-synonyms),
   `HMAC(secret_key, canonical_context)` deterministically selects which synonym
   to use. The choice is invisible to readers but statistically detectable.

2. **Detect** — Given a suspect text, replay the same HMAC for each eligible word.
   Count how many words match their "green" (key-assigned) synonym. Under the null
   hypothesis (no watermark), ~50% match by chance. A watermarked text matches
   ~90%+. A z-test converts this to a p-value.

## Files

```
watermark/
├── key.py          — Secret key generation, storage, rotation
├── embedder.py     — Injects watermark into text
├── detector.py     — Detects watermark, returns z-score + report
├── middleware.py   — FastAPI routes for Aegis scorer integration
└── __init__.py

tests/
└── test_watermark.py   — 35 tests, all passing
```

## Quick start

```python
from watermark.key import WatermarkKey
from watermark.embedder import WatermarkEmbedder
from watermark.detector import WatermarkDetector

# 1. Generate a key (do this once, store securely)
key = WatermarkKey.generate()
print("Save this key:", key.to_hex())

# 2. Embed watermark in every LLM response
embedder = WatermarkEmbedder(key)
protected = embedder.embed(llm_response).watermarked_text

# 3. Later, detect in a suspect clone's output
detector = WatermarkDetector(key)
result = detector.detect(suspect_text)
print(result.to_report())
```

## Plug into Aegis

In `src/scorer/scorer.py`:
```python
from watermark.middleware import add_watermark_routes
add_watermark_routes(app)
```

In `src/proxy/server.js`, after getting the LLM response:
```javascript
const wm = await fetch('http://localhost:8000/watermark/embed', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: llmResponse, session_id: sessionId })
}).then(r => r.json());

reply.send(wm.watermarked_text);  // send watermarked text to caller
```

## Environment setup

```bash
# Generate a key
python -c "import secrets; print(secrets.token_hex(32))"

# Set it (add to .env)
export AEGIS_WATERMARK_KEY=<your_hex_key>

# For key rotation, use versioned env vars:
export AEGIS_WATERMARK_KEY_v1=<old_key>
export AEGIS_WATERMARK_KEY_v2=<new_key>
```

## Run tests

```bash
python tests/test_watermark.py
```

Expected output:
```
Results: 35 passed, 0 failed / 35 total
🛡️  All watermark tests passing
```

## Detection thresholds

| Z-score | Confidence | p-value    | Meaning                    |
|---------|------------|------------|----------------------------|
| < 1.5   | none       | > 0.07     | No signal                  |
| 1.5–2.5 | weak       | 0.007–0.07 | Suspicious, needs more text|
| 2.5–4.0 | medium     | 0.00003–0.007 | Likely watermarked      |
| > 4.0   | strong     | < 0.00003  | Forensic-grade evidence    |

## Known limitations

- Requires ~50+ eligible words for reliable detection (~200 word response minimum)
- Heavy paraphrasing (>40% of words changed) degrades the signal
- Does not survive translation to another language
- Context-window synonym substitution is the proxy-compatible approach;
  logit-level watermarking (requires model access) would be stronger
