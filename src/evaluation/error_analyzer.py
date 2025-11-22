"""Error analysis and categorization."""
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class LinguisticErrors:
    """Linguistic error categorization."""
    phonetic_confusions: Dict[str, int]
    morphological_errors: Dict[str, int]
    lexical_errors: int
    oov_errors: int


@dataclass
class AcousticErrors:
    """Acoustic error categorization."""
    noise_related: int
    speaker_variability: int
    quality_issues: int


@dataclass
class DemographicErrors:
    """Demographic error categorization."""
    regional_errors: Dict[str, float]  # WER by region
    urban_rural_errors: Dict[str, float]  # WER by urban/rural
    speaker_demographics: Dict[str, float]  # WER by demographic group


@dataclass
class ErrorAnalysis:
    """Complete error analysis."""
    linguistic: LinguisticErrors
    acoustic: AcousticErrors
    demographic: Optional[DemographicErrors]
    total_errors: int
    error_rate: float


class ErrorAnalyzer:
    """
    Analyzes and categorizes errors by type.
    
    Identifies linguistic, acoustic, and demographic error patterns.
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        pass
    
    def categorize_errors(
        self,
        predictions: List[str],
        references: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> ErrorAnalysis:
        """
        Categorize errors by type.
        
        Args:
            predictions: List of predictions
            references: List of references
            metadata: Optional metadata for each sample
        
        Returns:
            ErrorAnalysis with categorized errors
        """
        # Analyze linguistic errors
        linguistic = self._analyze_linguistic_errors(predictions, references)
        
        # Analyze acoustic errors (simplified)
        acoustic = self._analyze_acoustic_errors(predictions, references, metadata)
        
        # Analyze demographic errors
        demographic = None
        if metadata:
            demographic = self._analyze_demographic_errors(
                predictions,
                references,
                metadata
            )
        
        # Total errors
        total_errors = sum(
            len(pred.split()) != len(ref.split())
            for pred, ref in zip(predictions, references)
        )
        error_rate = total_errors / len(predictions) if predictions else 0.0
        
        return ErrorAnalysis(
            linguistic=linguistic,
            acoustic=acoustic,
            demographic=demographic,
            total_errors=total_errors,
            error_rate=error_rate
        )
    
    def _analyze_linguistic_errors(
        self,
        predictions: List[str],
        references: List[str]
    ) -> LinguisticErrors:
        """Analyze linguistic error patterns."""
        phonetic_confusions = defaultdict(int)
        morphological_errors = defaultdict(int)
        lexical_errors = 0
        oov_words = set()
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Simple word-level comparison
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            
            # OOV words (in reference but not in prediction vocabulary)
            oov = ref_set - pred_set
            oov_words.update(oov)
            
            # Character-level confusions (simplified)
            for ref_word, pred_word in zip(ref_words, pred_words):
                if ref_word != pred_word:
                    # Find character differences
                    if len(ref_word) == len(pred_word):
                        for r_char, p_char in zip(ref_word, pred_word):
                            if r_char != p_char:
                                confusion = f"{r_char}->{p_char}"
                                phonetic_confusions[confusion] += 1
                    
                    # Morphological errors (prefix/suffix)
                    if ref_word.startswith(('ma', 'ba', 'ka')) or \
                       pred_word.startswith(('ma', 'ba', 'ka')):
                        morphological_errors['prefix'] += 1
        
        return LinguisticErrors(
            phonetic_confusions=dict(phonetic_confusions),
            morphological_errors=dict(morphological_errors),
            lexical_errors=lexical_errors,
            oov_errors=len(oov_words)
        )
    
    def _analyze_acoustic_errors(
        self,
        predictions: List[str],
        references: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> AcousticErrors:
        """Analyze acoustic error patterns."""
        noise_related = 0
        speaker_variability = 0
        quality_issues = 0
        
        if metadata:
            for pred, ref, meta in zip(predictions, references, metadata):
                # Check for quality indicators in metadata
                if meta.get('snr', 0) < 10:
                    noise_related += 1
                if meta.get('quality', 'good') != 'good':
                    quality_issues += 1
        
        return AcousticErrors(
            noise_related=noise_related,
            speaker_variability=speaker_variability,
            quality_issues=quality_issues
        )
    
    def _analyze_demographic_errors(
        self,
        predictions: List[str],
        references: List[str],
        metadata: List[Dict]
    ) -> DemographicErrors:
        """Analyze demographic disparities."""
        regional_errors = defaultdict(list)
        urban_rural_errors = defaultdict(list)
        speaker_demographics = defaultdict(list)
        
        for pred, ref, meta in zip(predictions, references, metadata):
            # Compute error for this sample
            pred_words = pred.split()
            ref_words = ref.split()
            errors = abs(len(pred_words) - len(ref_words))
            error_rate = errors / len(ref_words) if ref_words else 0.0
            
            # Categorize by demographic
            if 'region' in meta:
                regional_errors[meta['region']].append(error_rate)
            
            if 'urban_rural' in meta:
                urban_rural_errors[meta['urban_rural']].append(error_rate)
            
            if 'speaker_id' in meta:
                speaker_demographics[meta['speaker_id']].append(error_rate)
        
        # Average error rates
        regional_avg = {
            region: sum(errors) / len(errors) if errors else 0.0
            for region, errors in regional_errors.items()
        }
        
        urban_rural_avg = {
            category: sum(errors) / len(errors) if errors else 0.0
            for category, errors in urban_rural_errors.items()
        }
        
        speaker_avg = {
            speaker: sum(errors) / len(errors) if errors else 0.0
            for speaker, errors in speaker_demographics.items()
        }
        
        return DemographicErrors(
            regional_errors=regional_avg,
            urban_rural_errors=urban_rural_avg,
            speaker_demographics=speaker_avg
        )
    
    def analyze_phonetic_confusions(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, int]:
        """
        Generate phonetic confusion matrix.
        
        Uses rule-based G2P for better phonetic analysis.
        """
        confusions = defaultdict(int)
        
        # Simple rule-based G2P (same as in metrics.py)
        def text_to_phonemes(text: str) -> str:
            text = text.lower()
            replacements = [
                ('ng', 'N'), ('ny', 'Y'), ('th', 'T'), ('ph', 'P'),
                ('kh', 'K'), ('bh', 'B'), ('dl', 'L'), ('hl', 'H'),
                ('sh', 'S'), ('ts', 'Z'), ('tsh', 'C')
            ]
            for char, phone in replacements:
                text = text.replace(char, phone)
            return text
        
        for pred, ref in zip(predictions, references):
            # Phonetic alignment
            pred_phon = text_to_phonemes(pred)
            ref_phon = text_to_phonemes(ref)
            
            # Simple alignment (can be improved with Levenshtein backtrace)
            min_len = min(len(pred_phon), len(ref_phon))
            for i in range(min_len):
                if pred_phon[i] != ref_phon[i]:
                    confusion = f"{ref_phon[i]}->{pred_phon[i]}"
                    confusions[confusion] += 1
        
        return dict(confusions)
    
    def analyze_morphological_errors(
        self,
        predictions: List[str],
        references: List[str],
        language: str
    ) -> Dict[str, int]:
        """
        Analyze morphological errors for Bantu languages.
        
        Checks for specific prefix/suffix patterns common in Bantu languages.
        """
        prefix_errors = defaultdict(int)
        suffix_errors = defaultdict(int)
        
        # Expanded Bantu prefixes and suffixes
        bantu_prefixes = [
            'ma', 'ba', 'ka', 'mu', 'li', 'si', 'zi', 'u', 'ku', 'lu', 
            'ama', 'aba', 'izi', 'imi', 'uku'
        ]
        bantu_suffixes = ['ni', 'eni', 'ini', 'ile', 'ela', 'isa', 'ana']
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Align words (simple)
            min_len = min(len(pred_words), len(ref_words))
            for i in range(min_len):
                p_word = pred_words[i]
                r_word = ref_words[i]
                
                if p_word == r_word:
                    continue
                
                # Check prefix errors
                for prefix in bantu_prefixes:
                    # Missing prefix
                    if r_word.startswith(prefix) and not p_word.startswith(prefix):
                        # Check if the rest matches (stem match)
                        if r_word[len(prefix):] == p_word:
                            prefix_errors[f'missing_{prefix}'] += 1
                    
                    # Extra prefix
                    elif not r_word.startswith(prefix) and p_word.startswith(prefix):
                        if p_word[len(prefix):] == r_word:
                            prefix_errors[f'extra_{prefix}'] += 1
                            
                    # Wrong prefix
                    elif r_word.startswith(prefix) and p_word.startswith(prefix):
                        # If stems match but prefixes don't (handled by other logic usually, but good to catch)
                        pass

                # Check suffix errors
                for suffix in bantu_suffixes:
                    # Missing suffix
                    if r_word.endswith(suffix) and not p_word.endswith(suffix):
                        if r_word[:-len(suffix)] == p_word:
                            suffix_errors[f'missing_{suffix}'] += 1
                    
                    # Extra suffix
                    elif not r_word.endswith(suffix) and p_word.endswith(suffix):
                        if p_word[:-len(suffix)] == r_word:
                            suffix_errors[f'extra_{suffix}'] += 1
        
        return {
            'prefix_errors': dict(prefix_errors),
            'suffix_errors': dict(suffix_errors)
        }


