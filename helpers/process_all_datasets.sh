#!/bin/bash

# Base directory
SCRIPTS_DIR="/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/scripts"

echo "Setting up dataset directory structure..."
bash "$SCRIPTS_DIR/setup_datasets_structure.sh"

echo "Normalizing and processing datasets..."
python3 "$SCRIPTS_DIR/normalize_datasets.py"

echo "Dataset processing complete!"
