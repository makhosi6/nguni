# South African Languages Speech Corpus

This repository contains a comprehensive collection of speech corpora for South African languages, primarily focused on:

- Setswana (tn-ZA)
- Siswati (ss-ZA)
- isiZulu (zu-ZA)
- isiXhosa (xh-ZA)
- Xitsonga (ts-ZA)
- isiNdebele (nbl-ZA)
- Tshivenda (ve-ZA)
- Sesotho (st-ZA)
- English (en-ZA)
- Afrikaans (af-ZA)

The data is organized into various corpora collections including:
- Lwazi ASR and TTS corpora
- NCHLT speech corpora
- Autshumato bilingual and monolingual corpora
- Balanced English-language pair corpora
- Cross-lingual proper name corpus
- Second language learner speech corpus

This repository serves as a resource for speech technology development, including automatic speech recognition (ASR), text-to-speech (TTS), and other speech and language processing applications for South African languages.


## **1. Project Objective**
Reorganize unstructured audio files from various project folders into a standardized `PROCESSED` directory with:
- Language-specific subdirectories (e.g., `PROCESSED/zu-ZA/` for Zulu)
- A unified metadata CSV file tracking all audio files
- most importantly, audio and transcription organized by language, linked by metadata files(also create a test to ensure all audio files are linked to a transcription file/text win a CSV file)

## **2. Current State Assessment**
- Audio files scattered across multiple root directories
- No consistent naming convention or organization
- Missing/incomplete metadata

## **3. Target Structure**
```bash
PROJECT_ROOT/
└── PROCESSED/
    ├── zu-ZA/ (ISO 639-1 language code + ISO 3166-1 country code)
    │   ├── file1.wav
    │   ├── file2.wav
    ├── en-US/
    │   ├── file1.wav
    ├── metadata.csv

    # also refer to ./tech.md
```

### **4.1 Directory Organization**
1. Create `PROCESSED` directory
2. Implement language folder naming convention:
   - ISO 639-1 language code (2-3 letters)
   - ISO 3166-1 country code (2 letters)
   - Format: `{language}-{country}` (e.g., `zu-ZA`)
3. Move audio files into appropriate language directories (PROCESSED/*) based on their metadata or file naming conventions
4. Create a `metadata.csv` file in the `PROCESSED/[language-country]` directory


### **4.3 Metadata CSV Structure**
Required columns:
- `file_path`: Relative path from PROCESSED (e.g., "zu-ZA/file1.wav")
- `language`: Language code (e.g., "zu-ZA")
- `original_source`: Original location for traceability
- `duration`: Audio duration in seconds
- `sampling_rate`: Hz (e.g., 16000)
- `speaker_id`: If available (optional)
- `text`: Transcription if available (optional)

Recommended additional columns:
- `date_created`
- `audio_quality_rating`
- `license_info`

## **5. Quality Control**

### **5.1 Validation Steps**
1. **File Integrity Check**
   - Verify all files were copied correctly
   - Check for corrupted audio files

2. **Metadata Validation**
   ```python
   def validate_metadata():
       df = pd.read_csv(METADATA_PATH)
       assert not df.empty, "Metadata is empty"
       assert df['file_path'].is_unique, "Duplicate file paths"
       assert df['language'].isin(LANGUAGE_MAPPING.values()).all()
   ```

3. **Audio Standardization**
   - Resample to consistent rate if needed
   - Normalize volume levels

## **6. Maintenance Plan**

### **6.1 Version Control**
- Initialize git repository in `PROCESSED/`
- Tag versions (v1.0, v2.0) when making significant updates

### **6.2 Update Procedure**
1. Add new files to appropriate language directory
2. Append to metadata CSV (don't overwrite)
3. Run validation checks
4. Commit changes with descriptive message

## **7. Delivery Outputs**
1. Organized `PROCESSED/` directory
2. Complete `metadata.csv`
3. Data dictionary documenting all metadata fields
4. Validation report

## **8. Timeline**
1. Initial organization: [DD/MM/YYYY]
2. Metadata completion: [DD/MM/YYYY]
3. Quality checks: [DD/MM/YYYY]
4. Final delivery: [DD/MM/YYYY]

**Project Owner:** [Your Name]  
**Last Updated:** [DD/MM/YYYY]  

```
Adjust the language mapping, file naming conventions, and metadata fields according to your specific dataset characteristics. The script provided is a starting point that will need customization based on how your current files are named and organized.
```