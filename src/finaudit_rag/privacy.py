"""Mask supported identifiers before queries or evidence leave the process."""

import re
from dataclasses import dataclass

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w-]|\.[\w-])", re.IGNORECASE)
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
AADHAAR = re.compile(r"(?<![\w\d])[2-9]\d{3}[- ]?\d{4}[- ]?\d{4}(?![\w\d])")
ACCOUNT = re.compile(r"(?i)(\b(?:bank )?account (?:number|no\.?|#)\s*[:#|–-]?\s*)(\d{6,18})(?!\w)")


@dataclass(frozen=True)
class MaskedText:
    text: str
    replacements: int


def mask_sensitive_text(text: str) -> MaskedText:
    """Mask supported identifiers while preserving the surrounding evidence."""
    text, accounts = ACCOUNT.subn(r"\1[ACCOUNT]", text)
    text, emails = EMAIL.subn("[EMAIL]", text)
    text, pans = PAN.subn("[PAN]", text)
    text, aadhaars = AADHAAR.subn("[AADHAAR]", text)
    return MaskedText(text, accounts + emails + pans + aadhaars)
