"""Mask supported identifiers before queries or evidence leave the process."""

import re
from dataclasses import dataclass

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w-]|\.[\w-])", re.IGNORECASE)
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
AADHAAR = re.compile(r"(?<!\w)[2-9]\d{3}[\s–-]*\d{4}[\s–-]*\d{4}(?!\w)")
ACCOUNT = re.compile(
    r"(?i)(\b(?:bank\s+)?(?:account|acct|a/c)\b"
    r"(?:\s+(?:number\b|no\b\.?)\s*[:#|–-]?|\s*[:#|])\s*)"
    r"(\d(?:[\s–-]*\d)*)(?!\w)"
)


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
