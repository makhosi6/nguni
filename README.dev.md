# South African Speech Datasets - Developer Guide

## Overview

This project contains tools and scripts for organizing and processing South African speech datasets. The solution handles various dataset families including Lwazi ASR, Lwazi TTS, NCHLT, and specialized collections like the 5lang corpus.

## Project Structure

```
PROJECT_ROOT/
├── scripts/
│   ├── dataset_families/        # Dataset family configurations
│   │   ├── __init__.py
│   │   ├── lwazi_asr.py
│   │   ├── lwazi_tts.py
│   │   ├── lwazi2_tts.py
│   │   ├── nchlt.py
│   │   ├── balanced_cs.py
│   │   ├── auxiliary.py
│   │   └── learner.py
│   ├── config.py               # Main configuration
│   ├── organize_datasets.py    # Main organization script
│   ├── validate_datasets.py    # Dataset validation
│   ├── process_5lang.py        # 5lang corpus processor
│   └── run_organization.sh     # Main execution script
├── PROCESSED/                  # Source data directory
├── OUTPUT/                     # Processed output directory
└── README.md                  # Main documentation
```

## Setup and Installation

1. Ensure Python 3.6+ is installed
2. Clone the repository
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Basic Dataset Organization

To organize all datasets:
```bash
./scripts/run_organization.sh
```

This will:
- Process all datasets in the PROCESSED directory
- Organize them into the standardized OUTPUT structure
- Validate the organization
- Generate a validation report

### 2. Processing 5lang Corpus

To specifically process the 5lang corpus:
```bash
python scripts/process_5lang.py
```

This will:
- Separate multilingual content
- Organize by language
- Handle code-switching cases
- Generate language-specific metadata

### 3. Dataset Validation

To validate the organized datasets:
```bash
python scripts/validate_datasets.py
```

### 4. Adding New Dataset Families

1. Create a new module in `scripts/dataset_families/`
2. Define the family configuration:
   ```python
   FAMILY_NAME = "new_family"
   PATTERN = r'your_pattern_here'
   STRUCTURE = {
       'directories': {...},
       'files': {...}
   }
   ```
3. Update `__init__.py` to include the new family
4. Update main configuration if needed

## Directory Structure

### Input Structure (PROCESSED/)
```
PROCESSED/
├── [lang-ZA]/
│   ├── ASR datasets
│   ├── TTS datasets
│   └── Other collections
```

### Output Structure (OUTPUT/)
```
OUTPUT/
├── [lang-ZA]/
│   ├── audio/
│   ├── transcriptions/
│   ├── metadata/
│   ├── alignments/
│   └── config/
└── validation_report.json
```

## Dataset Family Configurations

Each dataset family has its own configuration module defining:
- Naming patterns
- Directory structure
- Required files
- Validation rules
- Supported languages

## Error Handling

- All errors are logged to `error.log`
- Validation failures appear in `validation_report.json`
- Check terminal output for real-time progress

## Best Practices

1. Always validate datasets after organization
2. Check validation reports for potential issues
3. Backup data before processing
4. Follow the defined directory structure
5. Use appropriate dataset family patterns

## Troubleshooting

Common issues and solutions:

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Permission Errors**
   ```bash
   chmod +x scripts/run_organization.sh
   ```

3. **Validation Failures**
   - Check validation_report.json
   - Verify dataset structure
   - Ensure all required files exist

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add/update dataset family configurations if needed
4. Test thoroughly
5. Submit a pull request

## License

See LICENSE file in the repository root.
