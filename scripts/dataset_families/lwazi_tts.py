"""Lwazi TTS dataset family configuration."""

FAMILY_NAME = "lwazi_tts"
PATTERN = r'tts\.lwazi\.[a-z]+\.1\.2'

STRUCTURE = {
    'audio': {
        'directory': 'wavs/',
        'extensions': ['wav']
    },
    'alignments': {
        'directory': 'textgrids/',
        'extensions': ['textgrid']
    },
    'transcriptions': {
        'directory': 'transcriptions/',
        'extensions': ['txt', 'json', 'csv', 'tsv', 'xml', 'data' 'transcriptions']
    },
    'config': {
        'directory': 'etc/',
        'files': ['PHONESETMAP.txt']
    },
    'metadata': {
        'required': ['README.txt', 'LICENCE.txt']
    }
}

SUPPORTED_LANGUAGES = ['nr', 'ss', 'tn', 'ts', 've', 'zu']  # ISO 639-1 codes

DATASET_VALIDATOR = {
    'required_dirs': ['wavs', 'textgrids', 'etc'],
    'required_files': ['README.txt', 'LICENCE.txt', 'PHONESETMAP.txt']
}
