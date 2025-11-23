#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path
import wave
import contextlib
import csv
import logging
import json
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_processing.log')
    ]
)

# Language code mapping
LANG_CODES = {
    'af-ZA': 'afr',
    'en-ZA': 'eng',
    'nr-ZA': 'nbl',
    'nso-ZA': 'nso',
    'ss-ZA': 'ssw',
    'st-ZA': 'sot',
    'tn-ZA': 'tsn',
    'ts-ZA': 'tso',
    've-ZA': 'ven',
    'xh-ZA': 'xho',
    'zu-ZA': 'zul'
}

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds."""
    try:
        if audio_path.endswith('.wav'):
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        else:
            # Use soundfile for other formats
            data, samplerate = sf.read(audio_path)
            return len(data) / float(samplerate)
    except Exception as e:
        logging.error(f"Error getting duration for {audio_path}: {str(e)}")
        return None

def convert_to_wav(src_audio, dst_wav):
    """Convert any audio format to WAV."""
    try:
        if src_audio.endswith('.wav'):
            shutil.copy2(src_audio, dst_wav)
        else:
            audio = AudioSegment.from_file(src_audio)
            audio.export(dst_wav, format='wav')
        return True
    except Exception as e:
        logging.error(f"Error converting {src_audio} to WAV: {str(e)}")
        return False

def collect_metadata(src_dir):
    """Collect all metadata from various file types."""
    metadata = {}
    speaker_metadata = {}
    
    # First look for Lwazi metadata CSV
    lwazi_csv = glob.glob(os.path.join(src_dir, '*metadata*.csv'))
    if lwazi_csv:
        try:
            df = pd.read_csv(lwazi_csv[0], skiprows=1)  # Skip the "Siswati" header
            for _, row in df.iterrows():
                speaker_id = str(row['Speaker']).zfill(3)  # Convert to 3-digit format
                speaker_metadata[speaker_id] = {
                    'gender': row['Gender'],
                    'line_type': row['Line'],
                    'age': row['Age']
                }
        except Exception as e:
            logging.warning(f"Could not process Lwazi metadata: {str(e)}")
    
    # Look for transcription files
    trans_dir = os.path.join(src_dir, 'transcriptions')
    if os.path.exists(trans_dir):
        for root, _, files in os.walk(trans_dir):
            for file in files:
                if file.endswith('.txt'):
                    try:
                        utterance_id = os.path.splitext(file)[0]
                        speaker_id = utterance_id.split('_')[1]  # Extract speaker ID
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            # Clean up the transcription
                            text = text.replace('[n]', '').replace('Siswati.', '').strip()
                            metadata[utterance_id] = {
                                'transcription': text,
                                **speaker_metadata.get(speaker_id, {})
                            }
                    except Exception as e:
                        logging.warning(f"Could not process transcription {file}: {str(e)}")
    
    # Look for other metadata files
    for ext in ['*.csv', '*.json', '*.txt', '*.tsv']:
        if ext == '*.csv' and lwazi_csv:  # Skip if we already processed Lwazi CSV
            continue
        for file_path in glob.glob(os.path.join(src_dir, '**', ext), recursive=True):
            if 'transcriptions' in file_path:  # Skip transcription directory
                continue
            try:
                if file_path.endswith('.csv') or file_path.endswith('.tsv'):
                    df = pd.read_csv(file_path, sep=None, engine='python')
                    for _, row in df.iterrows():
                        key = str(row[0])
                        metadata[key] = {**metadata.get(key, {}), **dict(row)}
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        metadata.update(json_data)
            except Exception as e:
                logging.warning(f"Could not process {file_path}: {str(e)}")
    
    return metadata

def process_audio_file(src_audio, dst_dir, utterance_id, metadata):
    """Process an audio file and copy it to the destination."""
    try:
        dst_wav = os.path.join(dst_dir, f"{utterance_id}.wav")
        os.makedirs(os.path.dirname(dst_wav), exist_ok=True)
        
        # Convert to WAV if needed
        if convert_to_wav(src_audio, dst_wav):
            duration = get_audio_duration(dst_wav)
            
            # Collect metadata for this file
            file_metadata = metadata.get(utterance_id, {})
            file_metadata['duration'] = duration
            file_metadata['original_audio'] = os.path.basename(src_audio)
            
            return file_metadata
        return None
    except Exception as e:
        logging.error(f"Error processing {src_audio}: {str(e)}")
        return None

def mark_directory_as_done(dir_path):
    """Rename a directory to add DONE_ prefix."""
    try:
        parent_dir = os.path.dirname(dir_path)
        dir_name = os.path.basename(dir_path)
        if not dir_name.startswith('DONE_'):
            new_path = os.path.join(parent_dir, f'DONE_{dir_name}')
            os.rename(dir_path, new_path)
            logging.info(f"Marked directory as done: {dir_path} -> {new_path}")
            return new_path
        return dir_path
    except Exception as e:
        logging.error(f"Error marking directory as done {dir_path}: {str(e)}")
        return dir_path

def process_language_directory(src_dir, dst_dir):
    """Process all data in a language directory."""
    try:
        # Get language code and dataset name
        lang_code = os.path.basename(os.path.dirname(src_dir))  # Get parent directory name for ss-ZA
        dataset_name = os.path.basename(src_dir)  # Get actual dataset directory name
        iso_code = LANG_CODES.get(lang_code)
        
        if not iso_code:
            logging.error(f"Unknown language code: {lang_code}")
            return
        
        # Setup paths
        dst_lang_dir = os.path.join(dst_dir, lang_code, f"{iso_code}_1")
        dst_wavs_dir = os.path.join(dst_lang_dir, "wavs")
        line_index_path = os.path.join(dst_lang_dir, "line_index.tsv")
        metadata_path = os.path.join(dst_lang_dir, "metadata.csv")
        
        # Create directories
        os.makedirs(dst_wavs_dir, exist_ok=True)
        
        # Collect all metadata first
        logging.info(f"Collecting metadata for {dataset_name}...")
        metadata = collect_metadata(src_dir)
        
        # Process audio files and create line_index.tsv
        processed_data = []
        
        # Find all audio files (specifically in audio directory for Lwazi structure)
        audio_dir = os.path.join(src_dir, 'audio')
        if os.path.exists(audio_dir):
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                audio_files.extend(glob.glob(os.path.join(audio_dir, '**', f'*{ext}'), recursive=True))
        else:
            # Fallback to searching entire directory if no audio subdirectory
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                audio_files.extend(glob.glob(os.path.join(src_dir, '**', f'*{ext}'), recursive=True))
        
        for audio_file in audio_files:
            utterance_id = os.path.splitext(os.path.basename(audio_file))[0]
            
            # Process audio and get metadata
            file_metadata = process_audio_file(audio_file, dst_wavs_dir, utterance_id, metadata)
            
            if file_metadata:
                # Add to processed data
                processed_data.append({
                    'utterance_id': utterance_id,
                    'transcription': file_metadata.get('transcription', ''),
                    'duration': file_metadata.get('duration', ''),
                    'original_audio': file_metadata.get('original_audio', ''),
                    **{k: v for k, v in file_metadata.items() if k not in ['transcription', 'duration', 'original_audio']}
                })
                
                logging.info(f"Processed: {audio_file}")
        
        # Write line_index.tsv
        with open(line_index_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            for data in processed_data:
                writer.writerow([data['utterance_id'], data['transcription']])
        
        # Write detailed metadata CSV
        pd.DataFrame(processed_data).to_csv(metadata_path, index=False)
        
        logging.info(f"Completed processing {lang_code}")
        
        # Mark directory as processed by adding DONE_ prefix
        mark_directory_as_done(src_dir)
        
    except Exception as e:
        logging.error(f"Error processing directory {src_dir}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process and normalize speech datasets.')
    parser.add_argument('--processed-dir', default='/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/PROCESSED',
                      help='Path to the PROCESSED directory')
    parser.add_argument('--datasets-dir', default='/home/makhosi/Hdrive/SPEECH_AUDIO_DATA/DATASETS',
                      help='Path to the output DATASETS directory')
    
    args = parser.parse_args()
    
    # Process each language directory
    for lang_dir in Path(args.processed_dir).glob('*-ZA'):
        if lang_dir.is_dir():
            logging.info(f"Processing {lang_dir}")
            process_language_directory(str(lang_dir), args.datasets_dir)
    
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()
