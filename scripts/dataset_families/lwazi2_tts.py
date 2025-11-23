"""Lwazi 2 TTS dataset family configuration."""

FAMILY_NAME = "lwazi2_tts"
PATTERN = r'Lwazi 2 .+ TTS Corpus'

STRUCTURE = {
    'audio': {
        'directory': 'audio/',
        'extensions': ['wav']
    },
    'transcriptions': {
        'directory': 'transcriptions/',
        'extensions': ['txt', 'json', 'csv', 'tsv', 'xml', 'data' 'transcriptions']
    },
    'metadata': {
        'directory': 'metadata/',
        'extensions': ['json', 'csv', 'tsv', 'xml', 'txt']
    }
}

SUPPORTED_LANGUAGES = ['nr', 'ss', 'st', 'tn', 'ts', 've', 'zu']  # ISO 639-1 codes

LANGUAGE_NAMES = {
    'nr': 'isiNdebele',
    'ss': 'Siswati',
    'st': 'Sesotho',
    'tn': 'Setswana',
    'ts': 'Xitsonga',
    've': 'Tshivenda',
    'zu': 'isiZulu'
}

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions', 'metadata'],
    'name_pattern': r'Lwazi 2 .+ TTS Corpus'
}
