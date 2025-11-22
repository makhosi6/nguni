"""Core metric computation with statistical rigor."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import jiwer
import unicodedata
from scipy import stats


@dataclass
class MetricResult:
    """Result of metric computation with confidence intervals."""
    metric_name: str
    value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_samples: int = 0
    breakdown: Optional[Dict] = None


@dataclass
class WERResult(MetricResult):
    """WER result with error breakdown."""
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    hits: int = 0


class MetricCalculator:
    """
    Computes evaluation metrics with bootstrap confidence intervals.
    
    Provides statistically rigorous metric computation with proper
    uncertainty quantification.
    """
    
    def __init__(
        self,
        num_bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42
    ):
        """
        Initialize metric calculator.
        
        Args:
            num_bootstrap_samples: Number of bootstrap iterations
            confidence_level: Confidence level for intervals (0.95 for 95%)
            seed: Random seed for reproducibility
        """
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.seed = seed
        np.random.seed(seed)
    
    def _bootstrap_resample(
        self,
        predictions: List[str],
        references: List[str],
        metric_fn
    ) -> np.ndarray:
        """
        Perform bootstrap resampling.
        
        Args:
            predictions: List of predictions
            references: List of references
            metric_fn: Function to compute metric on a sample
        
        Returns:
            Array of bootstrap metric values
        """
        n = len(predictions)
        bootstrap_values = []
        
        for _ in range(self.num_bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            pred_sample = [predictions[i] for i in indices]
            ref_sample = [references[i] for i in indices]
            
            # Compute metric on bootstrap sample
            value = metric_fn(pred_sample, ref_sample)
            bootstrap_values.append(value)
        
        return np.array(bootstrap_values)
    
    def _compute_ci(self, bootstrap_values: np.ndarray) -> Tuple[float, float]:
        """
        Compute confidence interval from bootstrap distribution.
        
        Args:
            bootstrap_values: Array of bootstrap metric values
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_values, lower_percentile)
        ci_upper = np.percentile(bootstrap_values, upper_percentile)
        
        return ci_lower, ci_upper
    
    def compute_wer_with_ci(
        self,
        predictions: List[str],
        references: List[str]
    ) -> WERResult:
        """
        Compute WER with 95% confidence intervals.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
        
        Returns:
            WERResult with point estimate, CI, and error breakdown
        """
        # Normalize texts
        preds_norm = [self._normalize_text(p) for p in predictions]
        refs_norm = [self._normalize_text(r) for r in references]
        
        # Compute point estimate
        wer_value = jiwer.wer(refs_norm, preds_norm)
        
        # Get error breakdown
        measures = jiwer.compute_measures(refs_norm, preds_norm)
        substitutions = measures.get('substitutions', 0)
        insertions = measures.get('insertions', 0)
        deletions = measures.get('deletions', 0)
        hits = measures.get('hits', 0)
        
        # Bootstrap resampling
        def compute_wer(pred, ref):
            return jiwer.wer(ref, pred)
        
        bootstrap_values = self._bootstrap_resample(
            preds_norm,
            refs_norm,
            compute_wer
        )
        
        ci_lower, ci_upper = self._compute_ci(bootstrap_values)
        
        return WERResult(
            metric_name="wer",
            value=wer_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=len(predictions),
            substitutions=substitutions,
            insertions=insertions,
            deletions=deletions,
            hits=hits,
            breakdown={
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions,
                'hits': hits
            }
        )
    
    def compute_cer_with_ci(
        self,
        predictions: List[str],
        references: List[str],
        diacritic_sensitive: bool = True
    ) -> MetricResult:
        """
        Compute CER with Unicode normalization and confidence intervals.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            diacritic_sensitive: Whether to preserve diacritics
        
        Returns:
            MetricResult with CER and CI
        """
        # Apply NFC Unicode normalization
        if diacritic_sensitive:
            preds_norm = [unicodedata.normalize('NFC', p) for p in predictions]
            refs_norm = [unicodedata.normalize('NFC', r) for r in references]
        else:
            # Remove diacritics for insensitive evaluation
            preds_norm = [
                unicodedata.normalize('NFD', p)
                .encode('ascii', 'ignore')
                .decode('ascii')
                for p in predictions
            ]
            refs_norm = [
                unicodedata.normalize('NFD', r)
                .encode('ascii', 'ignore')
                .decode('ascii')
                for r in references
            ]
        
        # Compute point estimate
        cer_value = jiwer.cer(refs_norm, preds_norm)
        
        # Bootstrap resampling
        def compute_cer(pred, ref):
            return jiwer.cer(ref, pred)
        
        bootstrap_values = self._bootstrap_resample(
            preds_norm,
            refs_norm,
            compute_cer
        )
        
        ci_lower, ci_upper = self._compute_ci(bootstrap_values)
        
        # Character substitution patterns
        char_substitutions = self._analyze_character_substitutions(
            preds_norm,
            refs_norm
        )
        
        return MetricResult(
            metric_name="cer",
            value=cer_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=len(predictions),
            breakdown={
                'character_substitutions': char_substitutions
            }
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for metric computation."""
        # Lowercase and strip whitespace
        text = text.lower().strip()
        # Remove extra whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _analyze_character_substitutions(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, int]:
        """Analyze common character substitution patterns."""
        substitutions = {}
        
        for pred, ref in zip(predictions, references):
            # Simple character-level alignment
            pred_chars = list(pred)
            ref_chars = list(ref)
            
            # Find substitutions (simplified)
            min_len = min(len(pred_chars), len(ref_chars))
            for i in range(min_len):
                if pred_chars[i] != ref_chars[i]:
                    sub_key = f"{ref_chars[i]}->{pred_chars[i]}"
                    substitutions[sub_key] = substitutions.get(sub_key, 0) + 1
        
        # Return top 10 most common
        sorted_subs = sorted(
            substitutions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return dict(sorted_subs)
    
    def compute_per_with_ci(
        self,
        predictions: List[str],
        references: List[str],
        language: str,
        phoneme_inventory: Optional[Dict[str, List[str]]] = None
    ) -> MetricResult:
        """
        Compute PER (Phoneme Error Rate) with confidence intervals.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            language: Language code for phoneme inventory
            phoneme_inventory: Optional phoneme inventory mapping
        
        Returns:
            MetricResult with PER and CI
        """
        # For now, use character-level as approximation
        # Full PER would require phoneme alignment
        # This is a placeholder that can be extended with proper phoneme alignment
        
        # Normalize texts
        preds_norm = [self._normalize_text(p) for p in predictions]
        refs_norm = [self._normalize_text(r) for r in references]
        
        # Compute character-level error as approximation
        per_value = jiwer.cer(refs_norm, preds_norm)
        
        # Bootstrap resampling
        def compute_per(pred, ref):
            return jiwer.cer(ref, pred)
        
        bootstrap_values = self._bootstrap_resample(
            preds_norm,
            refs_norm,
            compute_per
        )
        
        ci_lower, ci_upper = self._compute_ci(bootstrap_values)
        
        # TODO: Implement proper phoneme alignment and confusion matrices
        # For Bantu languages with click consonants
        
        return MetricResult(
            metric_name="per",
            value=per_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=len(predictions),
            breakdown={
                'note': 'PER computed as character-level approximation. '
                       'Full phoneme alignment not yet implemented.'
            }
        )
    
    def compute_ner_f1_with_ci(
        self,
        predictions: List[str],
        references: List[str],
        entity_tagger=None
    ) -> MetricResult:
        """
        Compute NER F1 score with confidence intervals.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            entity_tagger: Optional NER tagger (spaCy model)
        
        Returns:
            MetricResult with NER F1 and CI
        """
        # If no tagger provided, use simple proper noun detection
        if entity_tagger is None:
            # Simple heuristic: capitalized words
            import re
            def extract_entities(text):
                # Find capitalized words (simple heuristic)
                entities = re.findall(r'\b[A-Z][a-z]+\b', text)
                return set(entities)
        else:
            def extract_entities(text):
                doc = entity_tagger(text)
                entities = {ent.text for ent in doc.ents}
                return entities
        
        # Extract entities from predictions and references
        pred_entities = [extract_entities(p) for p in predictions]
        ref_entities = [extract_entities(r) for r in references]
        
        # Compute F1 scores per example
        f1_scores = []
        for pred_ent, ref_ent in zip(pred_entities, ref_entities):
            if len(ref_ent) == 0:
                # No entities in reference
                f1_scores.append(1.0 if len(pred_ent) == 0 else 0.0)
                continue
            
            # Precision
            if len(pred_ent) == 0:
                precision = 0.0
            else:
                precision = len(pred_ent & ref_ent) / len(pred_ent)
            
            # Recall
            recall = len(pred_ent & ref_ent) / len(ref_ent)
            
            # F1
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            f1_scores.append(f1)
        
        # Average F1
        f1_value = np.mean(f1_scores)
        
        # Bootstrap resampling
        bootstrap_values = []
        for _ in range(self.num_bootstrap_samples):
            indices = np.random.choice(
                len(f1_scores),
                size=len(f1_scores),
                replace=True
            )
            bootstrap_f1 = np.mean([f1_scores[i] for i in indices])
            bootstrap_values.append(bootstrap_f1)
        
        bootstrap_values = np.array(bootstrap_values)
        ci_lower, ci_upper = self._compute_ci(bootstrap_values)
        
        return MetricResult(
            metric_name="ner_f1",
            value=f1_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_samples=len(predictions)
        )

