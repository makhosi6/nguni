# Speech Data Processing Scripts

This repository contains scripts for processing and normalizing speech datasets from various sources into a standardized format.

## Directory Structure

```
SPEECH_AUDIO_DATA/
├── PROCESSED/          # Source data in various formats
│   ├── af-ZA/
│   ├── en-ZA/
│   └── ...
├── DATASETS/          # Normalized output data
│   ├── af-ZA/
│   │   └── afr_1/
│   │       ├── wavs/
│   │       ├── line_index.tsv
│   │       └── metadata.csv
│   └── ...
└── scripts/           # Processing scripts
    ├── setup_datasets_structure.sh
    ├── normalize_datasets.py
    ├── process_all_datasets.sh
    └── requirements.txt
```

## Features

1. **Audio Processing**
   - Supports multiple audio formats (WAV, MP3, FLAC, OGG, M4A)
   - Automatic conversion to WAV format
   - Duration extraction
   - Metadata preservation

2. **Metadata Collection**
   - Collects metadata from CSV, TSV, JSON, and text files
   - Combines metadata from multiple sources
   - Preserves all available information

3. **Standardized Output**
   - `line_index.tsv`: Contains utterance_id and transcription pairs
   - `metadata.csv`: Comprehensive metadata including:
     - Utterance ID
     - Transcription
     - Audio duration
     - Original audio filename
     - Additional metadata from source files

4. **Source Management**
   - Prefixes processed source directories with "DONE_"
   - Maintains original directory structure
   - Preserves original files

## Requirements

Python packages required:
- pandas: For data manipulation and CSV handling
- soundfile: For audio file processing
- pydub: For audio format conversion

Install requirements:
```bash
pip install -r scripts/requirements.txt
```

## Usage

1. **Process all datasets:**
```bash
./scripts/process_all_datasets.sh
```

2. **Process specific language:**
```bash
python3 scripts/normalize_datasets.py --processed-dir /path/to/PROCESSED --datasets-dir /path/to/DATASETS
```

## Language Codes

The scripts handle the following language codes:
- af-ZA (Afrikaans) → afr_1
- en-ZA (English) → eng_1
- nr-ZA (Southern Ndebele) → nbl_1
- nso-ZA (Northern Sotho) → nso_1
- ss-ZA (Swati) → ssw_1
- st-ZA (Southern Sotho) → sot_1
- tn-ZA (Tswana) → tsn_1
- ts-ZA (Tsonga) → tso_1
- ve-ZA (Venda) → ven_1
- xh-ZA (Xhosa) → xho_1
- zu-ZA (Zulu) → zul_1

## Output Format

### line_index.tsv
```
utterance_id   transcription
file1          text transcription
file2          another transcription
```

### metadata.csv
```
utterance_id,transcription,duration,original_audio,additional_metadata...
file1,text transcription,1.23,original.mp3,...
file2,another transcription,2.34,original.wav,...
```
