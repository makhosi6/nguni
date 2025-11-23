#!/usr/bin/env python3
"""
Process transcripts.jsonl files from language folders:
- Split into train/val (90/10)
- Move to transcriptions/train.jsonl and transcriptions/val.jsonl
- Merge into main dataset.jsonl
- Clean up redundant files
"""

import json
import random
from pathlib import Path
from typing import Dict, List

# Language code mapping from transcripts to ISO codes
LANG_CODE_MAP: Dict[str, str] = {
    "eng": "en",
    "xho": "xh",
    "zul": "zu",
    "sot": "st",
    "tsn": "tn",
    "tso": "ts",
    "ts": "ts",
    "nr": "nr",
    "af": "af",
    "ss": "ss",
    "nso": "nso",
    "ve": "ve",
}

# Language folder to ISO code mapping
LANG_FOLDER_MAP: Dict[str, str] = {
    "en-ZA": "en",
    "xh-ZA": "xh",
    "zu-ZA": "zu",
    "st-ZA": "st",
    "tn-ZA": "tn",
    "ts-ZA": "ts",
    "nr-ZA": "nr",
    "af-ZA": "af",
    "ss-ZA": "ss",
    "nso-ZA": "nso",
    "ve-ZA": "ve",
}


def normalize_language_code(lang_code: str, folder_name: str) -> str:
    """Normalize language code to ISO format."""
    # First try direct mapping
    if lang_code in LANG_CODE_MAP:
        return LANG_CODE_MAP[lang_code]
    # Fall back to folder name mapping
    if folder_name in LANG_FOLDER_MAP:
        return LANG_FOLDER_MAP[folder_name]
    # Default to lowercase if already 2 chars
    if len(lang_code) == 2:
        return lang_code.lower()
    return lang_code.lower()


def process_transcripts_file(
    root: Path, lang_folder: str, train_ratio: float = 0.9
) -> List[Dict]:
    """Process a transcripts.jsonl file and return all entries."""
    transcripts_file = root / lang_folder / "transcripts.jsonl"
    
    if not transcripts_file.exists():
        print(f"[WARN] {transcripts_file} not found, skipping")
        return []
    
    print(f"[INFO] Processing {lang_folder}...")
    
    entries = []
    with open(transcripts_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Normalize language code
                lang_code = entry.get("language", "")
                entry["language"] = normalize_language_code(lang_code, lang_folder)
                # Ensure audio path is relative to language folder
                audio_path = entry.get("audio", "")
                if not audio_path.startswith(lang_folder):
                    # Make path relative to language folder
                    if audio_path.startswith("audio/"):
                        entry["audio"] = f"{lang_folder}/{audio_path}"
                    else:
                        entry["audio"] = f"{lang_folder}/audio/{audio_path}"
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"[INFO] {lang_folder}: Loaded {len(entries)} entries")
    return entries


def split_train_val(entries: List[Dict], train_ratio: float = 0.9) -> tuple:
    """Split entries into train and validation sets."""
    random.seed(42)  # For reproducibility
    random.shuffle(entries)
    
    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    return train_entries, val_entries


def write_jsonl(filepath: Path, entries: List[Dict]) -> None:
    """Write entries to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            # Only keep required fields for Whisper format
            whisper_entry = {
                "audio": entry["audio"],
                "text": entry["text"],
                "language": entry["language"],
            }
            f.write(json.dumps(whisper_entry, ensure_ascii=False) + "\n")


def main():
    root = Path("/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/OUTPUT")
    languages = ["en-ZA", "xh-ZA", "zu-ZA", "st-ZA", "tn-ZA"]
    train_ratio = 0.9
    
    all_entries = []
    
    # Process each language
    for lang_folder in languages:
        entries = process_transcripts_file(root, lang_folder, train_ratio)
        if not entries:
            continue
        
        # Split into train/val
        train_entries, val_entries = split_train_val(entries, train_ratio)
        
        # Write to transcriptions/train.jsonl and transcriptions/val.jsonl
        train_file = root / lang_folder / "transcriptions" / "train.jsonl"
        val_file = root / lang_folder / "transcriptions" / "val.jsonl"
        
        write_jsonl(train_file, train_entries)
        write_jsonl(val_file, val_entries)
        
        print(f"[INFO] {lang_folder}: train={len(train_entries)}, val={len(val_entries)}")
        
        # Collect for main dataset
        all_entries.extend(entries)
    
    # Write combined dataset.jsonl
    if all_entries:
        dataset_file = root / "dataset.jsonl"
        write_jsonl(dataset_file, all_entries)
        print(f"[INFO] Combined dataset: {len(all_entries)} entries written to dataset.jsonl")
    
    # Delete redundant transcripts.jsonl files
    print("\n[INFO] Cleaning up redundant files...")
    for lang_folder in languages:
        transcripts_file = root / lang_folder / "transcripts.jsonl"
        if transcripts_file.exists():
            transcripts_file.unlink()
            print(f"[INFO] Deleted {transcripts_file}")


if __name__ == "__main__":
    main()

