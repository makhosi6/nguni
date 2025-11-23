"""Language Learner dataset family configuration."""

FAMILY_NAME = "learner"
PATTERN = r'[a-z]+_2nd_lang_learner_speech_corpus'

STRUCTURE = {
    'audio': {
        'directory': 'audio/',
        'extensions': ['wav', 'mp3']
    },
    'transcriptions': {
        'directory': 'transcriptions/',
        'extensions': ['txt', 'json', 'csv', 'tsv', 'xml', 'data', 'transcriptions']
    },
    'metadata': {
        'directory': 'metadata/',
        'extensions': ['json', 'csv', 'tsv', 'xml']
    },
    'assessments': {
        'directory': 'assessments/',
        'extensions': ['json', 'csv']
    }
}

SUPPORTED_LANGUAGES = ['zu']  # ISO 639-1 codes

DATASET_VALIDATOR = {
    'required_dirs': ['audio', 'transcriptions', 'metadata', 'assessments'],
    'name_pattern': r'[a-z]+_2nd_lang_learner_speech_corpus'
}
