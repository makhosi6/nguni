#!/usr/bin/env python3
"""
Parse XML transcript files from 5lang/ and organize data by language.
Extracts transcripts and copies corresponding audio files to OUTPUT/<LANG>-ZA/
"""

import os
import sys
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from collections import defaultdict
import json

# Language code mapping: XML lang_id -> ISO language code
LANG_MAP = {
    'eng': 'en-ZA',
    'xho': 'xh-ZA',
    'sot': 'st-ZA',
    'tsn': 'tn-ZA',
    'zul': 'zu-ZA',
}

def find_audio_file(audio_filename, audio_dirs):
    """Find audio file in any of the audio directories."""
    for audio_dir in audio_dirs:
        audio_path = Path(audio_dir) / audio_filename
        if audio_path.exists():
            return audio_path
    return None

def parse_xml_file(xml_path, audio_dirs, output_base):
    """Parse a single XML file and extract transcripts by language."""
    print(f"Parsing {xml_path}...")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {xml_path}: {e}")
        return
    
    # Statistics
    stats = defaultdict(int)
    missing_audio = []
    
    # Process all utterances
    utterances = root.findall('.//utterance')
    print(f"Found {len(utterances)} utterances")
    
    for utterance in utterances:
        # Get audio filename
        audio_elem = utterance.find('audio')
        if audio_elem is None or audio_elem.text is None:
            continue
        
        audio_filename = audio_elem.text.strip()
        speaker_id = utterance.find('speaker_id')
        speaker = speaker_id.text if speaker_id is not None and speaker_id.text else 'UNK'
        
        # Find the audio file
        audio_path = find_audio_file(audio_filename, audio_dirs)
        if audio_path is None:
            missing_audio.append(audio_filename)
            continue
        
        # Process each utterance segment
        segments = utterance.findall('utterance_segment')
        for segment in segments:
            lang_elem = segment.find('lang_id')
            transcription_elem = segment.find('transcription')
            
            if lang_elem is None or transcription_elem is None:
                continue
            
            lang_id = lang_elem.text.strip()
            transcription = transcription_elem.text.strip() if transcription_elem.text else ""
            
            # Skip empty transcriptions
            if not transcription:
                continue
            
            # Map language code
            if lang_id not in LANG_MAP:
                print(f"Warning: Unknown language code '{lang_id}', skipping")
                continue
            
            lang_code = LANG_MAP[lang_id]
            stats[lang_code] += 1
            
            # Create output directory structure
            lang_output_dir = Path(output_base) / lang_code
            audio_output_dir = lang_output_dir / 'audio'
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy audio file (if not already copied)
            dest_audio_path = audio_output_dir / audio_filename
            if not dest_audio_path.exists():
                try:
                    shutil.copy2(audio_path, dest_audio_path)
                except Exception as e:
                    print(f"Error copying {audio_path} to {dest_audio_path}: {e}")
            
            # Create transcript entry
            transcript_entry = {
                'audio': f"audio/{audio_filename}",
                'text': transcription,
                'language': lang_id,
                'speaker': speaker,
                'source_file': str(xml_path.name)
            }
            
            # Append to language-specific transcript file (JSONL format)
            transcript_file = lang_output_dir / 'transcripts.jsonl'
            with open(transcript_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(transcript_entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nStatistics for {xml_path.name}:")
    for lang, count in sorted(stats.items()):
        print(f"  {lang}: {count} segments")
    
    if missing_audio:
        print(f"\nWarning: {len(missing_audio)} audio files not found (showing first 10):")
        for audio in missing_audio[:10]:
            print(f"  {audio}")
        if len(missing_audio) > 10:
            print(f"  ... and {len(missing_audio) - 10} more")
    
    return stats

def main():
    """Main function to process all XML files."""
    # Base directories
    base_dir = Path(__file__).parent.parent
    xml_dir = base_dir / '5lang'
    output_base = base_dir / 'OUTPUT'
    
    # Find all XML files
    xml_files = list(xml_dir.glob('*.xml'))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Find all audio directories
    audio_dirs = []
    for audio_dir_name in ['audio', 'audio 2', 'audio 3', 'audio 4']:
        audio_dir = xml_dir / audio_dir_name
        if audio_dir.exists():
            audio_dirs.append(audio_dir)
            print(f"Found audio directory: {audio_dir}")
    
    if not audio_dirs:
        print("Warning: No audio directories found!")
    
    # Process each XML file
    total_stats = defaultdict(int)
    for xml_file in xml_files:
        stats = parse_xml_file(xml_file, audio_dirs, output_base)
        if stats:
            for lang, count in stats.items():
                total_stats[lang] += count
    
    # Print final statistics
    print("\n" + "="*60)
    print("TOTAL STATISTICS:")
    print("="*60)
    for lang, count in sorted(total_stats.items()):
        print(f"  {lang}: {count} segments")
    
    print(f"\nOutput organized in: {output_base}")
    for lang in sorted(total_stats.keys()):
        lang_dir = output_base / lang
        if lang_dir.exists():
            audio_count = len(list((lang_dir / 'audio').glob('*.wav'))) if (lang_dir / 'audio').exists() else 0
            print(f"  {lang}: {audio_count} audio files, {total_stats[lang]} transcript segments")

if __name__ == '__main__':
    main()

