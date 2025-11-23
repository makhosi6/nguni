================================================================================
Afrikaans Auxiliary Dataset (AFR1)
================================================================================

Dataset Name: AFR1 (Afrikaans Auxiliary)
Language: Afrikaans (af-ZA)
Source: DONE_afr-aux1 (standardized)

================================================================================
Contents
================================================================================

audio/
  Directory containing WAV audio files indexed by utterance ID (utt_id).
  Example: afr_8963_6542844630.wav

transcriptions/utt2text.tsv
  TSV file with columns: utt_id, transcription
  Example:
    afr_8963_6542844630	Mbali het koppies meel nodig om 'n koek te bak.
    afr_8963_3239061389	Haar proeftydperk in die oorlog maak sy deur onder eerw. Vermeulen.

metadata.csv
  Dataset summary with counts and processing information.

README.txt
  This file.

================================================================================
Usage
================================================================================

To access transcriptions:
  $ cat transcriptions/utt2text.tsv | head -10

To find audio for a specific utterance:
  $ ls audio/afr_8963_6542844630.wav

To count total samples:
  $ wc -l transcriptions/utt2text.tsv

================================================================================
Notes
================================================================================

- Audio files are symlinked by default (use --copy-audio to copy instead).
- Transcription IDs follow the pattern: afr_<SPEAKER_ID>_<UTTERANCE_ID>
- Processing script: ../../../scripts/process_afr1.sh

================================================================================
