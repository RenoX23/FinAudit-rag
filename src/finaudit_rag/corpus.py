"""Read the pinned corpus manifest and verify local PDF bytes."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Document:
    """One source snapshot with a stable ID and page count."""

    id: str
    title: str
    issuer: str
    kind: str
    url: str
    sha256: str
    pages: int


def load_manifest(path: Path) -> tuple[Document, ...]:
    """Validate and return the documents in a v1 corpus manifest."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != "v1":
        raise ValueError("expected a v1 corpus manifest")
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ValueError("expected a non-empty documents list")

    fields = ("id", "title", "issuer", "kind", "url", "sha256")
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != {*fields, "pages"}:
            raise ValueError(f"document {index}: unexpected fields")
        if any(not isinstance(entry[field], str) or not entry[field].strip() for field in fields):
            raise ValueError(f"document {index}: expected non-empty text fields")

        doc_id = entry["id"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", doc_id) or doc_id in seen_ids:
            raise ValueError(f"document {index}: invalid or duplicate ID")
        digest = entry["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in seen_hashes:
            raise ValueError(f"{doc_id}: invalid or duplicate SHA-256")
        url = urlsplit(entry["url"])
        if url.scheme != "https" or not url.hostname or url.username is not None or url.fragment:
            raise ValueError(
                f"{doc_id}: expected an HTTPS source URL without credentials or fragment"
            )
        if entry["kind"] not in {"filing", "regulatory"}:
            raise ValueError(f"{doc_id}: unknown document kind")
        if type(entry["pages"]) is not int or entry["pages"] < 1:
            raise ValueError(f"{doc_id}: expected a positive page count")

        documents.append(Document(**entry))
        seen_ids.add(doc_id)
        seen_hashes.add(digest)

    return tuple(documents)


def verify_pdf(document: Document, directory: Path) -> None:
    """Check a local file against its pinned SHA-256 without loading it into memory."""
    with (directory / f"{document.id}.pdf").open("rb") as pdf:
        if pdf.read(5) != b"%PDF-":
            raise ValueError(f"{document.id}: missing PDF header")
        pdf.seek(0)
        digest = hashlib.file_digest(pdf, "sha256").hexdigest()
    if digest != document.sha256:
        raise ValueError(f"{document.id}: SHA-256 differs from the manifest")


def main() -> None:
    """Validate a manifest and optionally its corresponding local files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    args = parser.parse_args()
    try:
        documents = load_manifest(args.manifest)
        if args.pdf_dir is not None:
            for document in documents:
                verify_pdf(document, args.pdf_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    pages = sum(doc.pages for doc in documents)
    status = " and verified PDF checksums" if args.pdf_dir is not None else ""
    print(f"Validated {len(documents)} documents ({pages} pages){status}.")


if __name__ == "__main__":
    main()
