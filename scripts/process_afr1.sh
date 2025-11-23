#!/bin/bash

################################################################################
# Process Afrikaans AFR1 Dataset (DONE_afr-aux1)
# 
# This script standardizes the Afrikaans auxiliary dataset (DONE_afr-aux1) into
# a common structure with:
#   - audio/          : Audio files indexed by utterance ID
#   - transcriptions/ : Text transcriptions (TSV format)
#   - metadata.csv    : Dataset summary (counts, file info)
#   - README.txt      : Dataset description
#
# Usage:
#   ./process_afr1.sh [SOURCE] [TARGET] [OPTIONS]
#
# Examples:
#   # Dry-run (show what would be done without making changes)
#   ./process_afr1.sh --dry-run
#
#   # Process with defaults
#   ./process_afr1.sh
#
#   # Process custom source and target
#   ./process_afr1.sh /path/to/DONE_afr-aux1 /path/to/afr_1_output
#
#   # Use copy instead of symlink for audio
#   ./process_afr1.sh --copy-audio
#
################################################################################

set -o pipefail  # Fail on pipe errors, but allow inner errors

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_SOURCE="${REPO_ROOT}/PROCESSED/af-ZA/DONE_afr-aux1"
DEFAULT_TARGET="${REPO_ROOT}/PROCESSED/af-ZA/afr_1"

DRY_RUN=false
COPY_AUDIO=false
SOURCE=""
TARGET=""

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --copy-audio)
      COPY_AUDIO=true
      shift
      ;;
    --help)
      head -30 "$0" | tail -n +2 | sed 's/^# //'
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$SOURCE" ]]; then
        SOURCE="$1"
      elif [[ -z "$TARGET" ]]; then
        TARGET="$1"
      else
        echo "Error: Too many positional arguments" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

# Use defaults if not provided
SOURCE="${SOURCE:-$DEFAULT_SOURCE}"
TARGET="${TARGET:-$DEFAULT_TARGET}"

# ============================================================================
# Validation
# ============================================================================

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: Source directory does not exist: $SOURCE" >&2
  exit 1
fi

if [[ ! -f "$SOURCE/line_index.tsv" ]]; then
  echo "ERROR: line_index.tsv not found in source: $SOURCE/line_index.tsv" >&2
  exit 1
fi

if [[ ! -d "$SOURCE/wavs" ]]; then
  echo "ERROR: wavs/ directory not found in source: $SOURCE/wavs" >&2
  exit 1
fi

# ============================================================================
# Logging
# ============================================================================

log_info() {
  echo "[INFO] $*"
}

log_action() {
  if $DRY_RUN; then
    echo "[DRY-RUN] $*"
  else
    echo "[ACTION] $*"
  fi
}

# ============================================================================
# Main Processing Logic
# ============================================================================

log_info "Starting Afrikaans AFR1 processing..."
log_info "Source: $SOURCE"
log_info "Target: $TARGET"
log_info "Dry-run: $DRY_RUN"
log_info "Copy audio (vs symlink): $COPY_AUDIO"
log_info ""

# Count entries in line_index.tsv
total_entries=$(wc -l < "$SOURCE/line_index.tsv")
log_info "Total entries in line_index.tsv: $total_entries"

# Count audio files
total_wavs=$(find "$SOURCE/wavs" -name "*.wav" | wc -l)
log_info "Total WAV files in wavs/: $total_wavs"
log_info ""

# Create target directories
if ! $DRY_RUN; then
  mkdir -p "$TARGET/audio"
  mkdir -p "$TARGET/transcriptions"
  log_action "Created target directories"
fi

# Parse line_index.tsv and create transcriptions.tsv + process audio
log_info "Processing line_index.tsv..."
matched_wavs=0
missing_wavs=0
transcription_file="$TARGET/transcriptions/utt2text.tsv"

if ! $DRY_RUN; then
  > "$transcription_file"  # Clear file
fi

# Count matching WAV files
while IFS=$'\t' read -r utt_id transcription; do
  wav_file="$SOURCE/wavs/${utt_id}.wav"
  
  if [[ -f "$wav_file" ]]; then
    ((matched_wavs++))
    
    if ! $DRY_RUN; then
      # Write transcription line
      echo "$utt_id	$transcription" >> "$transcription_file"
      
      # Link or copy audio
      target_audio="$TARGET/audio/${utt_id}.wav"
      if [[ "$COPY_AUDIO" == "true" ]]; then
        cp "$wav_file" "$target_audio" 2>/dev/null || true
      else
        ln -s "$wav_file" "$target_audio" 2>/dev/null || true
      fi
    fi
  else
    ((missing_wavs++))
  fi
done < "$SOURCE/line_index.tsv"

log_action "Processed transcriptions: $matched_wavs entries"
if [[ $missing_wavs -gt 0 ]]; then
  log_info "Warning: $missing_wavs entries missing corresponding WAV files"
fi
log_info ""

# Create metadata.csv
metadata_file="$TARGET/metadata.csv"
if ! $DRY_RUN; then
  cat > "$metadata_file" << EOF
Field,Value
Dataset Name,Afrikaans Auxiliary (AFR1)
Source Directory,$(basename "$SOURCE")
Total Transcriptions,$matched_wavs
Total Audio Files,$total_wavs
Missing Audio Files,$missing_wavs
Language,Afrikaans (af-ZA)
Processing Date,$(date -u +%Y-%m-%dT%H:%M:%SZ)
Transcription Format,TSV (utt_id,text)
EOF
  log_action "Created metadata.csv"
fi

# Create README.txt
readme_file="$TARGET/README.txt"
if ! $DRY_RUN; then
  cat > "$readme_file" << 'EOF'
================================================================================
Afrikaans Auxiliary Dataset (AFR1)
================================================================================

Dataset Name: AFR1 (Afrikaans Auxiliary)
Language: Afrikaans (af-ZA)
Source: DONE_afr-aux1 (standardized)

================================================================================
Contents
================================================================================

audio/
  Directory containing WAV audio files indexed by utterance ID (utt_id).
  Example: afr_8963_6542844630.wav

transcriptions/utt2text.tsv
  TSV file with columns: utt_id, transcription
  Example:
    afr_8963_6542844630	Mbali het koppies meel nodig om 'n koek te bak.
    afr_8963_3239061389	Haar proeftydperk in die oorlog maak sy deur onder eerw. Vermeulen.

metadata.csv
  Dataset summary with counts and processing information.

README.txt
  This file.

================================================================================
Usage
================================================================================

To access transcriptions:
  $ cat transcriptions/utt2text.tsv | head -10

To find audio for a specific utterance:
  $ ls audio/afr_8963_6542844630.wav

To count total samples:
  $ wc -l transcriptions/utt2text.tsv

================================================================================
Notes
================================================================================

- Audio files are symlinked by default (use --copy-audio to copy instead).
- Transcription IDs follow the pattern: afr_<SPEAKER_ID>_<UTTERANCE_ID>
- Processing script: ../../../scripts/process_afr1.sh

================================================================================
EOF
  log_action "Created README.txt"
fi

log_info ""
log_info "Processing complete!"
log_info "Output directory: $TARGET"
log_info "Transcriptions: $TARGET/transcriptions/utt2text.tsv"
log_info "Audio: $TARGET/audio/ ($matched_wavs files)"
log_info "Metadata: $TARGET/metadata.csv"
