#!/bin/bash

# Base directory for processed data
PROCESSED_DIR="/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/PROCESSED"

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

# Create standardized directory structure for each language
for lang_dir in "$PROCESSED_DIR"/*-ZA/; do
    if [ -d "$lang_dir" ]; then
        # Extract language code from directory name
        lang_code=$(basename "$lang_dir")
        
        # Get the ISO code for the language
        iso_code="${lang_codes[$lang_code]}"
        
        if [ ! -z "$iso_code" ]; then
            # Create the numbered subdirectory
            subset_dir="${lang_dir}${iso_code}_1"
            mkdir -p "$subset_dir/wavs"
            
            # Create empty line_index.tsv if it doesn't exist
            touch "$subset_dir/line_index.tsv"
            
            echo "Created structure for $lang_code ($iso_code)"
        fi
    fi
done

echo "Directory structure creation complete"
