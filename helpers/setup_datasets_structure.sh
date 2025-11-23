#!/bin/bash

# Base directories
PROCESSED_DIR="/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/PROCESSED"
DATASETS_DIR="/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/DATASETS"

# Language code mapping
declare -A lang_codes=(
    ["af-ZA"]="afr"
    ["en-ZA"]="eng"
    ["nr-ZA"]="nbl"
    ["nso-ZA"]="nso"
    ["ss-ZA"]="ssw"
    ["st-ZA"]="sot"
    ["tn-ZA"]="tsn"
    ["ts-ZA"]="tso"
    ["ve-ZA"]="ven"
    ["xh-ZA"]="xho"
    ["zu-ZA"]="zul"
)

# Create base DATASETS directory if it doesn't exist
mkdir -p "$DATASETS_DIR"

# Create standardized directory structure for each language
for lang_dir in "$PROCESSED_DIR"/*-ZA/; do
    if [ -d "$lang_dir" ]; then
        # Extract language code from directory name
        lang_code=$(basename "$lang_dir")
        
        # Get the ISO code for the language
        iso_code="${lang_codes[$lang_code]}"
        
        if [ ! -z "$iso_code" ]; then
            # Create the directory structure in DATASETS
            dataset_dir="$DATASETS_DIR/$lang_code/${iso_code}_1"
            mkdir -p "$dataset_dir/wavs"
            touch "$dataset_dir/line_index.tsv"
            
            echo "Created structure for $lang_code ($iso_code)"
        fi
    fi
done

echo "Dataset directory structure creation complete"
