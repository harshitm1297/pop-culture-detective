from __future__ import annotations

import csv
from datetime import UTC, datetime
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


def _apply_mojibake_replacements(text: str) -> str:
    cleaned = text
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


def _mojibake_score(text: str) -> tuple[int, int]:
    marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    replacement_count = sum(text.count(token) for token in MOJIBAKE_REPLACEMENTS)
    question_triplets = text.count("???")
    return marker_count + replacement_count + question_triplets, -len(text)


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
    best = _apply_mojibake_replacements(text)
    seen = {best}
    queue = [best]

    while queue:
        current = queue.pop(0)
        if not any(marker in current for marker in MOJIBAKE_MARKERS):
            if _mojibake_score(current) < _mojibake_score(best):
                best = current
            continue
        for codec in ("latin1", "cp1252"):
            try:
                candidate = current.encode(codec).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidate = _apply_mojibake_replacements(candidate)
            if candidate in seen:
                continue
            seen.add(candidate)
            queue.append(candidate)
            if _mojibake_score(candidate) < _mojibake_score(best):
                best = candidate

    return _apply_mojibake_replacements(best)


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


def strip_tmdb_review_boilerplate(text: str) -> str:
    if not text:
        return text
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    skip_next_url = False

    for line in lines:
        if not line:
            cleaned_lines.append("")
            skip_next_url = False
            continue
        if re.match(r"(?i)^check out my full review\b", line):
            skip_next_url = True
            continue
        if re.fullmatch(r"https?://\S+", line):
            if skip_next_url or len(line) <= 80:
                skip_next_url = False
                continue
        cleaned_lines.append(line)
        skip_next_url = False

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_source_boilerplate(text: str, *, source_name: str) -> str:
    if not text:
        return text

    cleaned = text

    if source_name == "vulture":
        start_markers = (
            "Things you buy through our links may earn",
            "movie review\n",
            "tv review\n",
            "endings\n",
            "spooky summer\n",
        )
        end_markers = (
            "\nSign up for\nThe Critics\n",
            "\nSign up for the Vulture Daily\n",
            "\nMore Movie Reviews\n",
            "\nMore TV Reviews\n",
            "\nTags:\n",
            "\nMost Viewed Stories\n",
            "\nLatest News from Vulture\n",
            "\nSign In To Continue Reading\n",
            "\nCreate Your Free Account\n",
            "\nAlready a subscriber?\nSign In\n",
            "\nHave an Account?\n",
        )
        for marker in start_markers:
            position = cleaned.find(marker)
            if position > 0:
                cleaned = cleaned[position:]
                break
        cut_positions = [cleaned.find(marker) for marker in end_markers if marker in cleaned]
        cut_positions = [position for position in cut_positions if position >= 0]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)]

    elif source_name == "indiewire":
        end_markers = (
            "\nRead More:\n",
            "\nDaily Headlines\n",
            "\nMore from IndieWire\n",
            "\nMust Read\n",
            "\nMore From IndieWire\n",
            "\nMost Popular\n",
            "\nYou may also like\n",
            "\nAbout\n",
            "\nNewsletter Sign Up\n",
            "\nHave a Tip?\n",
            "\nPMC Logo\n",
        )
        start_markers = (
            "\nIt’s crazy to think",
            "\nThere may be no ",
            "\nSteven Spielberg\nis returning",
            "\nInvoked several times",
        )
        for marker in start_markers:
            position = cleaned.find(marker)
            if position > 0:
                cleaned = cleaned[position + 1 :]
                break
        cut_positions = [cleaned.find(marker) for marker in end_markers if marker in cleaned]
        cut_positions = [position for position in cut_positions if position >= 0]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)]

    elif source_name == "slashfilm":
        end_markers = (
            "\nRecommended\n",
            "\nRecommended",
        )
        cut_positions = [cleaned.find(marker) for marker in end_markers if marker in cleaned]
        cut_positions = [position for position in cut_positions if position >= 0]
        if cut_positions:
            cleaned = cleaned[: min(cut_positions)]

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def flatten_text_for_storage(text: str) -> str:
    if not text:
        return text
    flattened = text.replace("\r", "\n")
    flattened = re.sub(r"\s*\n+\s*", " ", flattened)
    flattened = re.sub(r"[ \t]{2,}", " ", flattened)
    return flattened.strip()


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


def normalize_datetime(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T00:00:00Z"

    if re.fullmatch(r"\d{8}T\d{6}Z", value):
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return parsed.isoformat().replace("+00:00", "Z")

    if re.fullmatch(r"\d{14}", value):
        parsed = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        return parsed.isoformat().replace("+00:00", "Z")

    if re.fullmatch(r"\d{8}", value):
        parsed = datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
        return parsed.isoformat().replace("+00:00", "Z")

    iso_candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        return value

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed.isoformat().replace("+00:00", "Z")


def quality_flags_for_text(text: str, *, min_length: int) -> list[str]:
    flags: list[str] = []
    if not text:
        flags.append("empty_text")
    if text and len(text) < min_length:
        flags.append("too_short")
    if contains_encoding_noise(text):
        flags.append("possible_encoding_noise")
    return flags
