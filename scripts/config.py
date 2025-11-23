"""Configuration settings for dataset organization."""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = PROJECT_ROOT / 'OUTPUT'
PROCESSED_DIR = PROJECT_ROOT / 'PROCESSED'

# Import dataset family configurations
from dataset_families import (
    lwazi_asr,
    lwazi_tts,
    lwazi2_tts,
    nchlt,
    balanced_cs,
    auxiliary,
    learner
)

# Dataset family patterns mapping
DATASET_FAMILIES = {
    'lwazi_asr': lwazi_asr,
    'lwazi_tts': lwazi_tts,
    'lwazi2_tts': lwazi2_tts,
    'nchlt': nchlt,
    'balanced_cs': balanced_cs,
    'auxiliary': auxiliary,
    'learner': learner
}

# Compile all patterns
DATASET_PATTERNS = {
    family: module.PATTERN
    for family, module in DATASET_FAMILIES.items()
}

# Language ISO codes mapping
LANGUAGE_CODES = {
    'af': {'name': 'Afrikaans', 'iso639_2': 'afr'},
    'en': {'name': 'English', 'iso639_2': 'eng'},
    'nr': {'name': 'isiNdebele', 'iso639_2': 'nbl'},
    'xh': {'name': 'isiXhosa', 'iso639_2': 'xho'},
    'zu': {'name': 'isiZulu', 'iso639_2': 'zul'},
    'nso': {'name': 'Sepedi', 'iso639_2': 'nso'},
    'st': {'name': 'Sesotho', 'iso639_2': 'sot'},
    'tn': {'name': 'Setswana', 'iso639_2': 'tsn'},
    'ss': {'name': 'siSwati', 'iso639_2': 'ssw'},
    've': {'name': 'Tshivenda', 'iso639_2': 'ven'},
    'ts': {'name': 'Xitsonga', 'iso639_2': 'tso'}
}

# Standard directory structure for each dataset
DATASET_STRUCTURE = {
    'audio': ['wav', 'mp3', 'flac'],
    'transcriptions': ['txt', 'json', 'csv', 'tsv', 'xml', 'data', 'transcriptions'],
    'metadata': ['json', 'csv', 'yaml'],
    'alignments': ['textgrid', 'lab'],
    'config': ['json', 'yaml', 'txt']
}
