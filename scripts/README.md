# Speech Corpus Processing Scripts

This directory contains scripts for processing the soap opera speech corpus data.

## Scripts

### process_data.sh

Bash script that:
1. Validates input directory structure
2. Creates processed data structure
3. Processes XML and organizes audio files
4. Creates index files and metadata

Usage:
```
./process_data.sh [-d corpus_dir]
```

The `-d` option specifies the corpus root directory. If not provided, defaults to parent of scripts directory.

### process_soap_data.py

Python script that:
1. Parses XML metadata files
2. Organizes audio files by language/speaker 
3. Creates mapping between audio and transcriptions
4. Extracts metadata about speakers, languages, episodes

Called by process_data.sh to do the main processing work.

## Output Structure

The scripts create a processed/ directory with:

```
processed/
  ├── af-ZA/          # Afrikaans
  │   └── speakers/   # Audio organized by speaker
  ├── en-ZA/          # English 
  ├── nr-ZA/          # Southern Ndebele
  ├── ...             # Other languages
  ├── speakers.tsv    # Speaker metadata
  ├── languages.txt   # Language statistics
  └── transcripts.tsv # Audio-text mappings
```

## Requirements

- Python 3.6+
- Linux/Unix environment
- Write permissions in corpus directory
