"""NCHLT dataset family configuration."""

FAMILY_NAME = "nchlt"
PATTERN = r'nchlt_[a-z]+'

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
        'extensions': ['json', 'csv', 'tsv', 'xml']
    }
}

SUPPORTED_LANGUAGES = ['nr', 'tn', 'ts', 'xh']  # ISO 639-1 codes

LANGUAGE_MAPPING = {
    'nr': 'nbl',
    'tn': 'tsn',
    'ts': 'tso',
    'xh': 'xho'
}

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions', 'metadata'],
    'name_pattern': r'nchlt_[a-z]+'
}
