#!/bin/bash

# Process soap opera speech corpus data
#
# This script:
# 1. Validates input directories and files
# 2. Creates processed data structure
# 3. Processes XML and organizes audio files
# 4. Creates index files and metadata

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Default corpus directory is parent of script dir
CORPUS_DIR="$(dirname "$SCRIPT_DIR")"

# Process command line args
while getopts "d:" opt; do
  case $opt in
    d)
      CORPUS_DIR="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Validate corpus directory
if [ ! -d "$CORPUS_DIR" ]; then
  echo "Error: Corpus directory $CORPUS_DIR does not exist"
  exit 1
fi

if [ ! -d "$CORPUS_DIR/5lang" ]; then
  echo "Error: Required subdirectory '5lang' not found in $CORPUS_DIR"
  exit 1
fi

# Run processing script
python3 "$SCRIPT_DIR/process_soap_data.py" "$CORPUS_DIR"

echo "Processing complete. Output in $CORPUS_DIR/processed"
