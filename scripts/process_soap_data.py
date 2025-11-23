#!/usr/bin/env python3

"""
Script to process multilingual soap opera speech corpus data.
This script:
1. Parses XML metadata files 
2. Organizes audio files by language/speaker
3. Creates mapping between audio and transcriptions
4. Extracts metadata about speakers, languages, episodes

Usage:
    python process_soap_data.py [corpus_dir]
"""

import os
import sys
import xml.etree.ElementTree as ET
import csv
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoapOperaCorpus:
    def __init__(self, corpus_dir):
        self.corpus_dir = Path(corpus_dir)
        self.speakers = {}
        self.utterances = {}
        self.languages = set()
        
    def parse_xml(self, xml_file):
        """Parse XML file containing metadata and transcriptions."""
        logger.info(f"Parsing {xml_file}")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Parse speaker list
            for speaker in root.findall('.//speaker'):
                speaker_id = speaker.find('char_name').text
                self.speakers[speaker_id] = {
                    'gender': speaker.find('char_gender').text,
                    'actor': speaker.find('actor_name').text,
                    'lang': speaker.find('predom_lang').text
                }
                
            # Parse utterances
            for utterance in root.findall('.//utterance'):
                speaker = utterance.find('speaker_id').text
                audio = utterance.find('audio').text
                
                segments = []
                for seg in utterance.findall('.//utterance_segment'):
                    lang = seg.find('lang_id').text
                    text = seg.find('transcription').text
                    duration = float(seg.find('duration').text)
                    segments.append({
                        'lang': lang,
                        'text': text,
                        'duration': duration
                    })
                    self.languages.add(lang)
                    
                self.utterances[audio] = {
                    'speaker': speaker,
                    'segments': segments
                }
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse {xml_file}: {e}")
            return False
            
        return True
        
    def organize_files(self, output_dir):
        """Organize audio files by language and speaker."""
        output_dir = Path(output_dir)
        
        for audio_file, utt in self.utterances.items():
            # Get predominant language
            langs = [s['lang'] for s in utt['segments']]
            main_lang = max(set(langs), key=langs.count)
            
            # Create language/speaker dirs
            lang_dir = output_dir / main_lang
            speaker_dir = lang_dir / utt['speaker']
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy audio file
            src = self.corpus_dir / 'audio' / audio_file
            dst = speaker_dir / audio_file
            if src.exists():
                shutil.copy2(src, dst)
            else:
                logger.warning(f"Audio file not found: {src}")
                
    def create_transcript_mapping(self, output_file):
        """Create TSV mapping between audio files and transcriptions."""
        with open(output_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['audio_file', 'speaker', 'language', 'text', 'duration'])
            
            for audio, utt in self.utterances.items():
                for seg in utt['segments']:
                    writer.writerow([
                        audio,
                        utt['speaker'],
                        seg['lang'], 
                        seg['text'],
                        seg['duration']
                    ])
                    
    def write_metadata(self, output_dir):
        """Write metadata files about speakers, languages etc."""
        output_dir = Path(output_dir)
        
        # Write speaker info
        with open(output_dir / 'speakers.tsv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['speaker', 'gender', 'actor', 'language'])
            for speaker, info in self.speakers.items():
                writer.writerow([
                    speaker,
                    info['gender'],
                    info['actor'],
                    info['lang']
                ])
                
        # Write language stats
        with open(output_dir / 'languages.txt', 'w', encoding='utf-8') as f:
            for lang in sorted(self.languages):
                count = sum(1 for u in self.utterances.values() 
                          for s in u['segments'] if s['lang'] == lang)
                f.write(f"{lang}\t{count}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_soap_data.py CORPUS_DIR")
        sys.exit(1)
        
    corpus_dir = sys.argv[1]
    corpus = SoapOperaCorpus(corpus_dir)
    
    # Parse XML files
    for xml_file in Path(corpus_dir).glob('*.xml'):
        corpus.parse_xml(xml_file)
        
    # Create output structure
    output_dir = Path(corpus_dir) / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    corpus.organize_files(output_dir)
    corpus.create_transcript_mapping(output_dir / 'transcripts.tsv')
    corpus.write_metadata(output_dir)
    
if __name__ == '__main__':
    main()
