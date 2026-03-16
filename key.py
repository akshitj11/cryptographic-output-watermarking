"""
watermark/key.py
----------------
Secret key management for Aegis watermarking.

The key is the root of all security. If someone learns your key,
they can strip the watermark. Keep it in an env variable or secrets manager,
never hardcoded.

Key rotation: every key has a version. Detection tries all known versions,
so you can rotate without losing the ability to detect older watermarks.
"""

import os
import hmac
import hashlib
import secrets
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WatermarkKey:
    secret: bytes          # The raw secret bytes
    version: int = 1       # Key version — increment on rotation
    green_fraction: float = 0.5   # Fraction of synonym choices that are "green"
    delta: float = 2.0     # How strongly to bias toward green synonyms

    @classmethod
    def generate(cls, version: int = 1) -> "WatermarkKey":
        """Generate a cryptographically secure random key."""
        return cls(
            secret=secrets.token_bytes(32),  # 256-bit key
            version=version,
        )

    @classmethod
    def from_env(cls, env_var: str = "AEGIS_WATERMARK_KEY") -> "WatermarkKey":
        """
        Load key from environment variable.
        The env var should be a hex-encoded 32-byte secret.
        e.g. export AEGIS_WATERMARK_KEY=a3f1...
        """
        raw = os.environ.get(env_var)
        if not raw:
            raise ValueError(
                f"Environment variable {env_var} not set. "
                "Run: export AEGIS_WATERMARK_KEY=$(python -c \"import secrets; print(secrets.token_hex(32))\")"
            )
        return cls(secret=bytes.fromhex(raw))

    @classmethod
    def from_hex(cls, hex_str: str, version: int = 1) -> "WatermarkKey":
        """Load from a hex string (useful in tests)."""
        return cls(secret=bytes.fromhex(hex_str), version=version)

    def to_hex(self) -> str:
        """Export key as hex — store this securely."""
        return self.secret.hex()

    def derive_token_seed(self, context: str) -> int:
        """
        Core cryptographic primitive.

        Given a context string (surrounding words), derive a deterministic
        integer seed using HMAC-SHA256. This seed drives all synonym selection.

        Properties:
        - Deterministic: same context always gives same seed
        - Unpredictable: without the key, the seed looks random
        - Collision-resistant: different contexts give different seeds
        """
        h = hmac.new(
            self.secret,
            context.encode("utf-8"),
            hashlib.sha256
        ).digest()
        # Take first 8 bytes as a 64-bit integer for the RNG seed
        return int.from_bytes(h[:8], "big")

    def is_green(self, context: str, choice_index: int, num_choices: int) -> bool:
        """
        Decide if a particular synonym choice is "green" (watermark-preferred).

        The green/red partition is:
        - Derived from the secret key + context (so it's unpredictable)
        - Deterministic (same inputs → same answer always)
        - Balanced (roughly green_fraction of choices are green)
        """
        seed = self.derive_token_seed(context)
        import random
        rng = random.Random(seed)
        # Randomly assign each choice slot to green or red
        assignments = [rng.random() < self.green_fraction for _ in range(num_choices)]
        return assignments[choice_index % num_choices]


class KeyStore:
    """
    Manages multiple key versions for rotation.

    When you rotate keys:
    1. Generate a new key with version N+1
    2. Add it to the store
    3. New watermarks use the latest key
    4. Detection still checks all previous versions
    """

    def __init__(self):
        self._keys: dict[int, WatermarkKey] = {}

    def add(self, key: WatermarkKey):
        self._keys[key.version] = key

    def latest(self) -> WatermarkKey:
        if not self._keys:
            raise ValueError("No keys in store. Add a key first.")
        return self._keys[max(self._keys)]

    def get(self, version: int) -> Optional[WatermarkKey]:
        return self._keys.get(version)

    def all_versions(self) -> list[int]:
        return sorted(self._keys.keys())

    @classmethod
    def from_env(cls) -> "KeyStore":
        """
        Load all key versions from environment.
        Looks for AEGIS_WATERMARK_KEY_v1, AEGIS_WATERMARK_KEY_v2, etc.
        Falls back to AEGIS_WATERMARK_KEY for v1.
        """
        store = cls()
        # Try numbered versions first
        for v in range(1, 20):
            raw = os.environ.get(f"AEGIS_WATERMARK_KEY_v{v}")
            if raw:
                store.add(WatermarkKey.from_hex(raw, version=v))
        # Fallback: single key env var → version 1
        if not store._keys:
            raw = os.environ.get("AEGIS_WATERMARK_KEY")
            if raw:
                store.add(WatermarkKey.from_hex(raw, version=1))
        return store
