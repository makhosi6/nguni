# Repository Structure

## Main Organization

The repository is organized primarily by language corpus collections:

- `5lang/`: Contains audio files and metadata for 5 languages, with a focus on Siswati and Setswana
- `PROCESSED/`: Contains processed data organized by language code (e.g., af-ZA, en-ZA, etc.)
- `asr.lwazi.*`: ASR (Automatic Speech Recognition) corpora for specific languages
- `tts.lwazi.*`: TTS (Text-to-Speech) corpora for specific languages
- `nchlt.*`: NCHLT speech corpora for various languages
- `Autshumato.*`: Bilingual and monolingual corpora for translation purposes
- `*-aux*`: Auxiliary data for specific languages (e.g., ssw-aux1, tso-aux2)

## Language-Specific Folders

Language data is typically organized using ISO language codes:
- `af-ZA`: Afrikaans
- `en-ZA`: South African English
- `nbl-ZA`: isiNdebele
- `ss-ZA`: Siswati
- `st-ZA`: Sesotho
- `tn-ZA`: Setswana
- `ts-ZA`: Xitsonga
- `ve-ZA`: Tshivenda
- `xh-ZA`: isiXhosa
- `zu-ZA`: isiZulu

## Corpus Structure Pattern

Most individual corpus folders follow this pattern:
```
corpus_name/
├── LICENSE.txt       # Licensing information
├── README.txt        # Documentation
├── audio/            # WAV audio files
│   └── *.wav
├── transcriptions/   # Transcription files
├── metadata/         # Metadata files (CSV, TSV)
└── etc/              # Additional resources
```

## Special Collections

- `zu_2nd_lang_learner_speech_corpus/`: Contains speech data from second language learners of isiZulu
- `lwazi_2_cross-lingual_proper_name_corpus/`: Cross-lingual proper name pronunciation data
- `Sesotho_Function_Words/`, `Sesotho_Tone/`, `Sesotho_Vowels/`: Specialized linguistic data for Sesotho
- etc...