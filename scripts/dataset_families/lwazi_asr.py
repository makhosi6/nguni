"""Lwazi ASR dataset family configuration."""

FAMILY_NAME = "lwazi_asr"
PATTERN = r'ASR\.Lwazi\.[A-Za-z]+\.1\.0'

STRUCTURE = {
    'audio': {
        'directory': 'audio/',
        'extensions': ['wav']
    },
    'transcriptions': {
        'directory': 'transcriptions/',
        'extensions': ['txt', 'json', 'csv', 'tsv', 'xml', 'data', 'transcriptions']
    },
    'metadata': {
        'files': ['Lwazi_metadata_{lang}.csv'],
        'required': ['README.txt', 'LICENCE.txt']
    }
}

SUPPORTED_LANGUAGES = ['nr', 'ss', 'st', 'tn', 've', 'zu']  # ISO 639-1 codes

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions'],
    'required_files': ['README.txt', 'LICENCE.txt'],
    'metadata_pattern': r'Lwazi_metadata_[a-z]+\.csv'
}
