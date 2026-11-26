import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest

from finaudit_rag.corpus import load_manifest, verify_pdf

PDF_BYTES = b"%PDF-1.7\nfixture bytes"


@pytest.fixture
def write_manifest(tmp_path: Path) -> Callable[..., Path]:
    entry = {
        "id": "sample-filing",
        "title": "Example filing",
        "issuer": "Example issuer",
        "kind": "filing",
        "url": "https://example.org/filing.pdf",
        "sha256": sha256(PDF_BYTES).hexdigest(),
        "pages": 1,
    }

    def write(*, duplicate: bool = False, **changes: object) -> Path:
        document = {**entry, **changes}
        path = tmp_path / "corpus.json"
        path.write_text(
            json.dumps({"version": "v1", "documents": [document] * (2 if duplicate else 1)}),
            encoding="utf-8",
        )
        return path

    return write


def test_verify_pdf(write_manifest: Callable[..., Path], tmp_path: Path) -> None:
    (document,) = load_manifest(write_manifest())
    (tmp_path / "sample-filing.pdf").write_bytes(PDF_BYTES)
    verify_pdf(document, tmp_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"duplicate": True}, "duplicate ID"),
        ({"id": "../outside"}, "invalid or duplicate ID"),
        ({"pages": True}, "positive page count"),
        ({"sha256": "not-a-digest"}, "invalid or duplicate SHA-256"),
        ({"url": "http://example.org/insecure.pdf"}, "expected an HTTPS source"),
        ({"url": "https://:secret@example.org/filing.pdf"}, "expected an HTTPS source"),
    ],
)
def test_reject_invalid_entries(
    write_manifest: Callable[..., Path], changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest(**changes))


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"<html>error</html>", "missing PDF header"), (b"%PDF-modified", "SHA-256 differs")],
)
def test_reject_changed_pdf(
    write_manifest: Callable[..., Path], tmp_path: Path, content: bytes, message: str
) -> None:
    (document,) = load_manifest(write_manifest())
    (tmp_path / "sample-filing.pdf").write_bytes(content)
    with pytest.raises(ValueError, match=message):
        verify_pdf(document, tmp_path)
