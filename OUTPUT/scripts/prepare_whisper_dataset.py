#!/usr/bin/env python3
"""
Pipeline for standardising the South African ASR corpora into the HuggingFace Whisper JSONL format.

Key features
------------
* Scans every `<LANG>-ZA` directory under a dataset root.
* Loads heterogeneous transcript sources (TSV, CSV, XML, Kaldi-style utt2text, plain TXT, *.data).
* Cleans text according to spec.md requirements (removes markup/timestamps, normalises casing).
* Resolves messy audio naming schemes by indexing all audio files per language.
* Optionally normalises audio to 16 kHz mono WAV via ffmpeg (`--normalize-audio`).
* Splits each language into `transcriptions/train.jsonl` and `transcriptions/val.jsonl` (90/10 default).
* Produces a multilingual `dataset.jsonl` plus a machine-readable summary report.

Usage
-----
    python scripts/prepare_whisper_dataset.py \
        --root /home/makhosi/Hdrive/SPEECH_AUDIO_DATA/OUTPUT \
        --lowercase \
        --train-ratio 0.9 \
        --normalize-audio

Run with `--dry-run` first to inspect what would happen without writing files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree


LANG_CODE_MAP: Dict[str, str] = {
    "af-ZA": "af",
    "en-ZA": "en",
    "nr-ZA": "nr",
    "nso-ZA": "nso",
    "ss-ZA": "ss",
    "st-ZA": "st",
    "tn-ZA": "tn",
    "ts-ZA": "ts",
    "ve-ZA": "ve",
    "xh-ZA": "xh",
    "zu-ZA": "zu",
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
TRANSCRIPT_SKIP_KEYWORDS = {
    "licence",
    "license",
    "readme",
    "metadata",
    "phoneset",
    "phonesetmap",
    "config",
    "manifest",
}

TEXT_HEADER_KEYWORDS = {
    "audio",
    "path",
    "file",
    "filename",
    "audio_file",
    "text",
    "transcript",
    "utterance",
    "utt",
    "utterance_id",
    "sentence",
}

CLEANING_PATTERNS: Sequence[Tuple[re.Pattern, str]] = [
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"\[[^\]]*\]"), " "),
    (re.compile(r"\b(?:SPEAKER|SPK|SPKR|MALE|FEMALE|HOST|ANNOUNCER)\s*\d*:?", re.IGNORECASE), " "),
    (re.compile(r"\b[A-Z]{2,}\d*:?"), " "),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), " "),
]

@dataclass
class Example:
    audio: str
    text: str
    language: str


@dataclass
class LanguageSummary:
    language_dir: str
    language_code: str
    total_examples: int
    train_examples: int
    val_examples: int
    skipped_examples: int
    duplicate_examples: int
    missing_audio: int
    transcript_files: int
    audio_files_indexed: int
    output_train: str
    output_val: str


class AudioIndex:
    """Indexes every audio file under `<lang>/audio` for fuzzy lookups."""

    def __init__(self, root: Path, lang_dir: Path) -> None:
        self.root = root
        self.lang_dir = lang_dir
        self.audio_dir = lang_dir / "audio"
        self.primary: Dict[str, Path] = {}
        self.suffix_map: Dict[str, List[Path]] = defaultdict(list)
        self.all_paths: List[Path] = []

    def build(self) -> None:
        if not self.audio_dir.exists():
            return
        for file_path in self.audio_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rel_path = file_path.relative_to(self.root)
            stem = file_path.stem.lower()
            self.primary.setdefault(stem, rel_path)
            self.suffix_map[self._suffix_key(stem)].append(rel_path)
            self.all_paths.append(rel_path)

    def resolve(self, value: str) -> Optional[Path]:
        value = (value or "").strip()
        if not value:
            return None
        candidate = Path(value)
        # Absolute path inside the dataset root?
        if candidate.is_absolute() and candidate.exists():
            try:
                return candidate.relative_to(self.root)
            except ValueError:
                pass
        # Try direct joins.
        for base in (self.lang_dir, self.audio_dir):
            attempt = (base / candidate).resolve()
            if attempt.exists():
                return attempt.relative_to(self.root)
        # Try only the filename appended to audio dir.
        name_only = candidate.name
        if name_only:
            attempt = (self.audio_dir / name_only).resolve()
            if attempt.exists():
                return attempt.relative_to(self.root)
        # Lookup by stems.
        stem = candidate.stem.lower()
        direct = self.primary.get(stem)
        if direct:
            return direct
        suffix = self._suffix_key(stem)
        candidates = self.suffix_map.get(suffix, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Choose the one whose stem endswith the requested stem.
            for rel in candidates:
                if rel.stem.lower().endswith(stem):
                    return rel
        # Final fallback: substring search.
        for rel in self.all_paths:
            if stem and stem in rel.stem.lower():
                return rel
        return None

    @staticmethod
    def _suffix_key(stem: str) -> str:
        parts = [p for p in stem.split("_") if p]
        if len(parts) >= 4:
            return "_".join(parts[-4:])
        if len(parts) >= 2:
            return "_".join(parts[-2:])
        return stem


class LanguageProcessor:
    def __init__(self, root: Path, lang_dir: Path, lang_code: str, args: argparse.Namespace) -> None:
        self.root = root
        self.lang_dir = lang_dir
        self.lang_code = lang_code
        self.args = args
        self.audio_dir = lang_dir / "audio"
        self.trans_dir = lang_dir / "transcriptions"
        self.audio_out_dir = lang_dir / args.audio_out if args.normalize_audio else None
        self.output_train = self.trans_dir / "train.jsonl"
        self.output_val = self.trans_dir / "val.jsonl"
        self.audio_index = AudioIndex(root, lang_dir)
        self.examples: Dict[str, Example] = {}
        self.skipped = 0
        self.missing_audio = 0
        self.duplicates = 0
        self.transcript_file_count = 0

    def run(self) -> LanguageSummary:
        if not self.trans_dir.exists():
            return self._empty_summary()
        self.audio_index.build()
        transcript_pairs = list(self._load_transcripts())
        for audio_ref, text in transcript_pairs:
            if not text:
                self.skipped += 1
                continue
            resolved = self.audio_index.resolve(audio_ref or "")
            if not resolved:
                resolved = self.audio_index.resolve(Path(audio_ref or "").stem)
            if not resolved:
                self.missing_audio += 1
                continue
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                self.skipped += 1
                continue
            final_audio_rel = self._prepare_audio_target(resolved)
            entry = Example(
                audio=final_audio_rel.as_posix(),
                text=cleaned_text if not self.args.lowercase else cleaned_text.lower(),
                language=self.lang_code,
            )
            if entry.audio in self.examples:
                self.duplicates += 1
                continue
            self.examples[entry.audio] = entry
        total_examples = len(self.examples)
        if not total_examples:
            return self._empty_summary()
        train_examples, val_examples = self._split_and_write()
        return LanguageSummary(
            language_dir=self.lang_dir.name,
            language_code=self.lang_code,
            total_examples=total_examples,
            train_examples=len(train_examples),
            val_examples=len(val_examples),
            skipped_examples=self.skipped,
            duplicate_examples=self.duplicates,
            missing_audio=self.missing_audio,
            transcript_files=self.transcript_file_count,
            audio_files_indexed=len(self.audio_index.all_paths),
            output_train=self.output_train.relative_to(self.root).as_posix(),
            output_val=self.output_val.relative_to(self.root).as_posix(),
        )

    def _split_and_write(self) -> Tuple[List[Example], List[Example]]:
        entries = sorted(self.examples.values(), key=lambda item: item.audio)
        rng = random.Random(f"{self.lang_dir.name}-{self.args.seed}")
        rng.shuffle(entries)
        split_idx = max(1, int(len(entries) * self.args.train_ratio))
        if split_idx >= len(entries):
            split_idx = len(entries) - 1
        train_entries = entries[:split_idx]
        val_entries = entries[split_idx:]
        if not self.args.dry_run:
            self._write_jsonl(self.output_train, train_entries)
            self._write_jsonl(self.output_val, val_entries)
        return train_entries, val_entries

    def _write_jsonl(self, path: Path, entries: Iterable[Example]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def _load_transcripts(self) -> Iterator[Tuple[str, str]]:
        for file_path in sorted(self.trans_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".tsv", ".csv", ".txt", ".xml", ".data"} and "utt2text" not in file_path.name:
                continue
            if any(keyword in file_path.name.lower() for keyword in TRANSCRIPT_SKIP_KEYWORDS):
                continue
            self.transcript_file_count += 1
            suffix = file_path.suffix.lower()
            if "utt2text" in file_path.name or suffix == ".tsv":
                yield from self._parse_two_column(file_path, delimiter="\t")
            elif suffix == ".csv":
                yield from self._parse_two_column(file_path, delimiter=",")
            elif suffix == ".xml":
                yield from self._parse_xml(file_path)
            elif suffix == ".data":
                yield from self._parse_data(file_path)
            elif suffix == ".txt":
                yield from self._parse_plain_text_file(file_path)

    def _parse_two_column(self, file_path: Path, delimiter: str) -> Iterator[Tuple[str, str]]:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header_detected = False
            header: List[str] = []
            for row in reader:
                if not row:
                    continue
                row = [cell.strip() for cell in row]
                if not header_detected and self._looks_like_header(row):
                    header = [cell.lower() for cell in row]
                    header_detected = True
                    continue
                audio_ref: Optional[str] = None
                text_value: Optional[str] = None
                if header_detected:
                    audio_ref = self._extract_by_header(row, header, ("audio", "audio_file", "filepath", "file", "filename", "utt", "utterance", "utterance_id"))
                    text_value = self._extract_by_header(row, header, ("text", "transcript", "utterance", "orth", "sentence"))
                if not audio_ref:
                    audio_ref = row[0]
                if not text_value:
                    text_value = row[-1]
                yield audio_ref, text_value

    def _parse_xml(self, file_path: Path) -> Iterator[Tuple[str, str]]:
        try:
            tree = ElementTree.parse(file_path)
        except ElementTree.ParseError:
            return
        for recording in tree.findall(".//recording"):
            audio_attr = recording.attrib.get("audio") or recording.attrib.get("file")
            if not audio_attr:
                continue
            orth = recording.find("orth")
            text_value = ""
            if orth is not None:
                text_value = "".join(orth.itertext())
            yield audio_attr, text_value

    def _parse_data(self, file_path: Path) -> Iterator[Tuple[str, str]]:
        pattern = re.compile(r'\(\s*([^\s]+)\s+"(.*?)"\s*\)')
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                audio_id, text_value = match.groups()
                text_value = text_value.replace(r"\"", '"')
                yield audio_id, text_value

    def _parse_plain_text_file(self, file_path: Path) -> Iterator[Tuple[str, str]]:
        basename = file_path.stem
        if basename.lower() in {"readme", "licence", "license"}:
            return
        text_value = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text_value:
            return
        yield basename, text_value.replace("\n", " ")

    @staticmethod
    def _looks_like_header(row: Sequence[str]) -> bool:
        lowered = [cell.lower() for cell in row if cell]
        if not lowered:
            return False
        return any(cell in TEXT_HEADER_KEYWORDS for cell in lowered)

    @staticmethod
    def _extract_by_header(row: Sequence[str], header: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        lookup = {name: idx for idx, name in enumerate(header)}
        for candidate in candidates:
            if candidate in lookup and lookup[candidate] < len(row):
                value = row[lookup[candidate]].strip()
                if value:
                    return value
        return None

    def _clean_text(self, value: str) -> str:
        text = value.strip()
        for pattern, replacement in CLEANING_PATTERNS:
            text = pattern.sub(replacement, text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _prepare_audio_target(self, rel_path: Path) -> Path:
        if not self.args.normalize_audio or not self.audio_out_dir:
            return rel_path
        source = self.root / rel_path
        try:
            relative = source.relative_to(self.audio_dir)
        except ValueError:
            relative = Path(source.name)
        target = (self.audio_out_dir / relative).with_suffix(".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(target),
            ]
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            return target.relative_to(self.root)
        except ValueError:
            return Path(target.as_posix())

    def _empty_summary(self) -> LanguageSummary:
        return LanguageSummary(
            language_dir=self.lang_dir.name,
            language_code=self.lang_code,
            total_examples=0,
            train_examples=0,
            val_examples=0,
            skipped_examples=self.skipped,
            duplicate_examples=self.duplicates,
            missing_audio=self.missing_audio,
            transcript_files=self.transcript_file_count,
            audio_files_indexed=len(self.audio_index.all_paths),
            output_train=self.output_train.relative_to(self.root).as_posix(),
            output_val=self.output_val.relative_to(self.root).as_posix(),
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Whisper JSONL splits for all language folders.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (contains <LANG>-ZA directories).")
    parser.add_argument("--languages", nargs="*", help="Optional list of <LANG>-ZA directories to process.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio (default: 0.9).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic splits.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase all transcripts per spec.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write JSONL files; just print summary.")
    parser.add_argument("--normalize-audio", action="store_true", help="Convert audio to 16 kHz mono WAV (uses ffmpeg).")
    parser.add_argument("--audio-out", type=str, default="audio_16k", help="Subdirectory for normalised audio (relative to each language).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    if not root.exists():
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        return 1
    requested = set(args.languages) if args.languages else None
    language_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.endswith("-ZA") and (requested is None or p.name in requested)
    )
    if not language_dirs:
        print(f"[WARN] No matching <LANG>-ZA directories found under {root}")
        return 0
    summaries: List[LanguageSummary] = []
    combined_dataset_path = root / "dataset.jsonl"
    combined_entries: List[Example] = []
    for lang_dir in language_dirs:
        lang_code = LANG_CODE_MAP.get(lang_dir.name, lang_dir.name.split("-")[0])
        processor = LanguageProcessor(root, lang_dir, lang_code, args)
        summary = processor.run()
        summaries.append(summary)
        combined_entries.extend(processor.examples.values())
        print(f"[INFO] {lang_dir.name}: {summary.total_examples} examples "
              f"(train={summary.train_examples}, val={summary.val_examples}, "
              f"skipped={summary.skipped_examples}, missing_audio={summary.missing_audio})")
    if combined_entries and not args.dry_run:
        with combined_dataset_path.open("w", encoding="utf-8") as f:
            for entry in combined_entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    summary_path = root / "dataset_summary.json"
    if not args.dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        print(f"[INFO] Summary written to {summary_path.relative_to(root).as_posix()}")
    else:
        print(f"[INFO] Dry-run: summary would be written to {summary_path.relative_to(root).as_posix()}")
    if combined_entries:
        if args.dry_run:
            print(f"[INFO] Dry-run: combined multilingual file would be at {combined_dataset_path.relative_to(root).as_posix()}")
        else:
            print(f"[INFO] Combined multilingual file: {combined_dataset_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

