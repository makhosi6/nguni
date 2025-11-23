"""Balanced Code-switching dataset family configuration."""

FAMILY_NAME = "balanced_cs"
PATTERN = r'balanced_eng[a-z]+'

STRUCTURE = {
    'audio': {
        'directory': 'audio/',
        'extensions': ['wav']
    },
    'transcriptions': {
        'directory': 'transcriptions/',
        'extensions': ['txt', 'json', 'csv', 'tsv', 'xml' , 'data', 'transcriptions']
    },
    'metadata': {
        'directory': 'metadata/',
        'extensions': ['json', 'csv', 'tsv', 'xml']
    }
}

SUPPORTED_LANGUAGES = ['ts', 'xh']  # ISO 639-1 codes

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions', 'metadata'],
    'name_pattern': r'balanced_eng(tsn|xho)'
}
