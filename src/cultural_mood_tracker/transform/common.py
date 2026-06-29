from __future__ import annotations

import csv
import html
import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any


MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "â€™": "'",
    "\u00e2\u20ac\u02dc": "'",
    "â€˜": "'",
    "\u00e2\u20ac\u0153": '"',
    "â€œ": '"',
    "\u00e2\u20ac\ufffd": '"',
    "â€\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "â€“": "-",
    "\u00e2\u20ac\u201d": "-",
    "â€”": "-",
    "\u00e2\u20ac\u00a6": "...",
    "â€¦": "...",
    "\u00c2\u00a3": "\u00a3",
    "Â£": "£",
    "\u00c2": "",
    "Â": "",
}
MOJIBAKE_MARKERS = (
    "\u00e2",
    "\u00c3",
    "\u00d0",
    "\u00d9",
    "â€",
    "Ã",
    "Ð",
    "Ù",
)


def find_latest_run_id(root: Path) -> str:
    run_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not run_dirs:
        raise RuntimeError(f"No run directories found under {root}")
    return run_dirs[-1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def find_matching_file(directory: Path, prefix: str, suffix: str) -> Path | None:
    matches = sorted(directory.glob(f"{prefix}*{suffix}"))
    return matches[0] if matches else None


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    for _ in range(2):
        if any(token in text for token in MOJIBAKE_REPLACEMENTS):
            for bad, good in MOJIBAKE_REPLACEMENTS.items():
                text = text.replace(bad, good)
        if not any(marker in text for marker in MOJIBAKE_MARKERS):
            return text
        candidates: list[str] = []
        for codec in ("latin1", "cp1252"):
            try:
                candidates.append(text.encode(codec).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if not candidates:
            break
        # Prefer the candidate with fewer known mojibake markers.
        text = min(candidates, key=lambda value: sum(value.count(marker) for marker in MOJIBAKE_MARKERS))
    if any(token in text for token in MOJIBAKE_REPLACEMENTS):
        for bad, good in MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)
    return text


def clean_text(text: str, *, strip_urls: bool = False) -> str:
    if not text:
        return ""
    cleaned = repair_mojibake(text)
    cleaned = html.unescape(cleaned)
    if any(token in cleaned for token in MOJIBAKE_REPLACEMENTS):
        for bad, good in MOJIBAKE_REPLACEMENTS.items():
            cleaned = cleaned.replace(bad, good)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    if strip_urls:
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)
    return cleaned.strip()


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip().lower()


def normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_name(value)).strip()


def contains_encoding_noise(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS) or any(
        token in text for token in MOJIBAKE_REPLACEMENTS
    )


def count_normalized_phrase_occurrences(text: str, phrase: str) -> int:
    normalized_text = normalize_phrase(text)
    normalized_phrase = normalize_phrase(phrase)
    if not normalized_text or not normalized_phrase:
        return 0
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])"
    return len(re.findall(pattern, normalized_text))


def stable_text_hash(text: str) -> str:
    return sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def stable_value_hash(value: str) -> str:
    return sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(value) for key, value in row.items()})


def serialize_csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_wikipedia_timestamp(raw_value: str) -> str:
    if len(raw_value) != 10:
        return raw_value
    return f"{raw_value[0:4]}-{raw_value[4:6]}-{raw_value[6:8]}T00:00:00Z"


def quality_flags_for_text(text: str, *, min_length: int) -> list[str]:
    flags: list[str] = []
    if not text:
        flags.append("empty_text")
    if text and len(text) < min_length:
        flags.append("too_short")
    if contains_encoding_noise(text):
        flags.append("possible_encoding_noise")
    return flags
