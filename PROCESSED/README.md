### Languages

# South African Speech Datasets

## Project Overview

This repository contains a comprehensive collection of South African speech datasets, encompassing all 11 official languages of South Africa. The datasets are organized to facilitate research and development in various speech technologies, including:

- Automatic Speech Recognition (ASR)
- Text-to-Speech Synthesis (TTS)
- Code-switching Analysis
- Language Learning Research
- Speech Corpus Development

### Project Structure

The datasets are organized in a standardized directory structure:
```
PROCESSED/
├── [lang-ZA]/        # Language-specific directories (e.g., zu-ZA for isiZulu)
│   ├── ASR datasets  # Speech recognition data
│   ├── TTS datasets  # Text-to-speech synthesis data
│   └── Other data    # Specialized collections
└── README.md         # This documentation file
```

Each language is identified by its ISO 639-1/2 code followed by the country code (ZA for South Africa).

### Dataset Categories

Our collections include several types of speech data:

1. **ASR Datasets**: Focused on speech recognition, containing recorded speech with corresponding transcriptions
2. **TTS Datasets**: Designed for speech synthesis, including phonetic alignments and careful pronunciations
3. **Code-switching Datasets**: Capturing multilingual speech patterns, especially English mixed with indigenous languages
4. **Language Learning**: Resources for second language acquisition research
5. **Auxiliary Data**: Supplementary collections supporting various speech technology tasks

### Data Standards

Most datasets follow established formats and include:
- High-quality audio recordings
- Accurate transcriptions
- Speaker metadata
- Recording conditions information
- Licensing information
- Documentation

## Dataset Families

### 1. Lwazi ASR Family
**Pattern**: `ASR.Lwazi.[lang].1.0`  
**Purpose**: Automatic Speech Recognition  
**Structure**:
- `audio/` - Speech recordings
- `transcriptions/` - Text transcriptions
- `Lwazi_metadata_[lang].csv` - Metadata file
- `README.txt` - Documentation
- `LICENCE.txt` - License information

**Found in**:
- `nr-ZA`: ASR.Lwazi.Nbl.1.0
- `ss-ZA`: ASR.Lwazi.Ssw.1.0
- `st-ZA`: ASR.Lwazi.Sot.1.0
- `tn-ZA`: ASR.Lwazi.Tsn.1.0
- `ve-ZA`: ASR.Lwazi.Ven.1.0
- `zu-ZA`: ASR.Lwazi.Zul.1.0

### 2. Lwazi TTS Family
**Pattern**: `tts.lwazi.[lang].1.2`  
**Purpose**: Text to Speech Synthesis  
**Structure**:
- `wavs/` - Audio files
- `textgrids/` - Phonetic alignments
- `etc/` - Configuration files
- `PHONESETMAP.txt` - Phoneme mappings
- `README.txt` - Documentation
- `LICENCE.txt` - License information

**Found in**:
- `nr-ZA`: tts.lwazi.nbl.1.2
- `ss-ZA`: tts.lwazi.ssw.1.2
- `tn-ZA`: tts.lwazi.tsn.1.2
- `ts-ZA`: tts.lwazi.tso.1.2
- `ve-ZA`: tts.lwazi.ven.1.2
- `zu-ZA`: tts.lwazi.zul.1.2

### 3. Lwazi 2 TTS Family
**Pattern**: `Lwazi 2 [Language] TTS Corpus`  
**Purpose**: Updated/expanded TTS datasets  
**Structure**:
- Audio files
- Transcription files
- Metadata

**Found in**:
- `nr-ZA`: "Lwazi 2 isiNdebele TTS Corpus"
- `ss-ZA`: "Lwazi 2 Siswati TTS Corpus"
- `st-ZA`: "Lwazi 2 Sesotho TTS Corpus" (multiple versions)
- `tn-ZA`: "Lwazi 2 Setswana TTS Corpus"
- `ts-ZA`: "Lwazi 2 Xitsonga TTS Corpus"
- `ve-ZA`: "Lwazi 2 Tshivenda TTS Corpus"
- `zu-ZA`: "Lwazi 2 isiZulu TTS Corpus"

### 4. NCHLT Family
**Pattern**: `nchlt_[lang]`  
**Purpose**: Speech Recognition  
**Structure**: ASR-focused data organization

**Found in**:
- `nr-ZA`: nchlt_nbl
- `tn-ZA`: nchlt_tsn
- `ts-ZA`: nchlt_tso
- `xh-ZA`: nchlt_xho

### 5. Balanced Code-switching Family
**Pattern**: `balanced_eng[lang]`  
**Purpose**: Code-switching Analysis  
**Structure**: Similar to 5lang corpus format

**Found in**:
- `ts-ZA`: balanced_engtsn
- `xh-ZA`: balanced_engxho

### 6. Auxiliary Datasets
**Pattern**: `[lang]-aux[1|2]`  
**Purpose**: Supplementary speech data  
**Structure**: Varies by dataset

**Found in**:
- `nr-ZA`: nbl-aux1
- `ts-ZA`: tso-aux2

### 7. Language Learner Corpora
**Pattern**: `[lang]_2nd_lang_learner_speech_corpus`  
**Purpose**: Second language acquisition research
**Structure**: Speech recordings from language learners

**Found in**:
- `zu-ZA`: zu_2nd_lang_learner_speech_corpus

### 8. Afrikaans Auxiliary Dataset (AFR1)
**Pattern**: `afr_1`  
**Purpose**: Afrikaans speech recognition and corpus development  
**Source**: Standardized from `DONE_afr-aux1`  
**Structure**:
- `audio/` - 2927 WAV audio files indexed by utterance ID
- `transcriptions/utt2text.tsv` - Tab-separated transcriptions (utt_id, text)
- `metadata.csv` - Dataset summary (counts, processing info)
- `README.txt` - Dataset documentation

**Processing Script**: `scripts/process_afr1.sh`  
**Usage Examples**:
```bash
# Dry-run to see what would be processed
./scripts/process_afr1.sh --dry-run

# Process with default paths (DONE_afr-aux1 → afr_1)
./scripts/process_afr1.sh

# Process with custom source and target
./scripts/process_afr1.sh /path/to/source /path/to/target

# Copy audio files instead of symlinking
./scripts/process_afr1.sh --copy-audio
```

**Dataset Statistics**:
- Total utterances: 2927
- Language: Afrikaans (af-ZA)
- Format: WAV audio + TSV transcriptions
- All audio files matched with transcriptions

**Found in**:
- `af-ZA/afr_1` - Standardized dataset

## Languages

```json
    [
        {
            name: "Afrikaans",
            iso639_1: "af",
            iso639_2: "afr"
        },
        {
            name: "English",
            iso639_1: "en",
            iso639_2: "eng"
        },
        {
            name: "isiNdebele",
            iso639_1: "nr",
            iso639_2: "nbl"
        },
        {
            name: "isiXhosa",
            iso639_1: "xh",
            iso639_2: "xho"
        },
        {
            name: "isiZulu",
            iso639_1: "zu",
            iso639_2: "zul"
        },
        {
            name: "Sepedi (Northern Sotho)",
            iso639_1: "nso",
            iso639_2: "nso"
        },
        {
            name: "Sesotho (Southern Sotho)",
            iso639_1: "st",
            iso639_2: "sot"
        },
        {
            name: "Setswana",
            iso639_1: "tn",
            iso639_2: "tsn"
        },
        {
            name: "siSwati",
            iso639_1: "ss",
            iso639_2: "ssw"
        },
        {
            name: "Tshivenda",
            iso639_1: "ve",
            iso639_2: "ven"
        },
        {
            name: "Xitsonga",
            iso639_1: "ts",
            iso639_2: "tso"
        }
    ]
```

## Data Processing Scripts

The repository includes several scripts to help process and organize the datasets:

### Main Scripts

1. **process_data.sh**
   - **Purpose**: Extracts speech datasets from zip archives
   - **Features**:
     - Processes zip files from a source directory
     - Maintains directory structure during extraction
     - Fixes file permissions automatically
     - Provides colored output for status
     - Logs errors for troubleshooting
   - **Usage**: `./process_data.sh`

2. **normalize_datasets.py**
   - **Purpose**: Standardizes dataset formats
   - **Features**:
     - Converts various dataset formats to a common structure
     - Processes audio and transcription files
     - Validates dataset integrity
   - **Usage**: `python3 scripts/normalize_datasets.py --processed-dir [SOURCE] --datasets-dir [DEST]`

3. **setup_datasets_structure.sh**
   - **Purpose**: Creates standardized directory structure
   - **Features**:
     - Sets up language-specific directories
     - Creates necessary subdirectories
     - Ensures consistent organization
   - **Usage**: `./scripts/setup_datasets_structure.sh`

4. **organize_processed.sh**
   - **Purpose**: Organizes processed datasets
   - **Features**:
     - Sorts files into appropriate directories
     - Maintains dataset hierarchies
     - Ensures proper file organization
   - **Usage**: `./scripts/organize_processed.sh`

### Support Files

- **requirements.txt**: Lists Python dependencies
- **scripts/README.md**: Detailed script documentation

### Script Workflow

1. Start with `setup_datasets_structure.sh` to create directory structure
2. Use `process_data.sh` to extract raw datasets
3. Run `normalize_datasets.py` to standardize formats
4. Apply `organize_processed.sh` to ensure proper organization

### Best Practices for Scripts

1. Always check script parameters before running
2. Monitor the error logs during processing
3. Backup data before running normalization scripts
4. Verify dataset integrity after processing

## Usage Guidelines

### Accessing the Data
Each dataset typically includes:
1. Audio files in standard formats (WAV)
2. Transcription files
3. Metadata and documentation
4. License information

### Common Use Cases
- **Speech Recognition**: Use ASR.Lwazi and NCHLT datasets
- **Speech Synthesis**: Use Lwazi TTS and Lwazi 2 TTS datasets
- **Code-switching Research**: Use balanced_eng[lang] datasets
- **Language Learning**: Use specialized corpora like zu_2nd_lang_learner_speech_corpus

### Best Practices
1. Always check dataset-specific README files
2. Verify licensing requirements
3. Cite the appropriate dataset in your research
4. Follow provided data format specifications

## Contributing

### Adding New Datasets
1. Follow the existing directory structure
2. Include comprehensive documentation
3. Provide clear licensing information
4. Ensure audio quality meets standards
5. Include necessary metadata

### Improving Existing Datasets
- Report issues with data quality
- Suggest improvements to documentation
- Contribute additional annotations
- Help standardize formats

## Contact

For questions, suggestions, or contributions, please contact the maintainers through the appropriate channels.

## Citation

When using these datasets, please cite both the specific dataset and the overall collection as appropriate. Refer to individual dataset documentation for specific citation requirements.
