"""Auxiliary dataset family configuration."""

FAMILY_NAME = "auxiliary"
PATTERN = r'[a-z]+-aux[12]'

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
        'directory': 'metadata/',
        'extensions': ['json', 'csv', 'tsv', 'xml', 'txt']
    }
}

SUPPORTED_LANGUAGES = ['nr', 'ts']  # ISO 639-1 codes

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions', 'metadata'],
    'name_pattern': r'(nbl|tso)-aux[12]'
}
