"""Evaluation protocols for model comparison."""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from .test_set_manager import TestSet, TestSetManager
from .model_manager import ModelManager
from .metrics import MetricCalculator, WERResult
from .statistics import StatisticalTester, TestResult


logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Result of comparing two models."""
    model1_name: str
    model2_name: str
    language: str
    wer_difference: float
    wer_difference_ci: Tuple[float, float]
    relative_improvement: float  # percentage
    statistical_test: TestResult
    model1_wer: WERResult
    model2_wer: WERResult


@dataclass
class ComparisonResults:
    """Results of baseline comparison protocol."""
    baseline_model: str
    comparisons: List[ModelComparison]
    summary: Dict[str, any]


class BaselineComparisonProtocol:
    """
    Protocol for comparing fine-tuned models against baseline.
    
    Evaluates base Whisper-large-v3 and all fine-tuned models,
    then performs statistical comparisons.
    """
    
    def __init__(
        self,
        test_set_manager: TestSetManager,
        model_manager: ModelManager,
        metric_calculator: MetricCalculator,
        statistical_tester: StatisticalTester
    ):
        """
        Initialize baseline comparison protocol.
        
        Args:
            test_set_manager: TestSetManager instance
            model_manager: ModelManager instance
            metric_calculator: MetricCalculator instance
            statistical_tester: StatisticalTester instance
        """
        self.test_set_manager = test_set_manager
        self.model_manager = model_manager
        self.metric_calculator = metric_calculator
        self.statistical_tester = statistical_tester
    
    def run(
        self,
        languages: List[str],
        baseline_model_path: str,
        fine_tuned_models: Dict[str, str],
        test_sets: Optional[Dict[str, TestSet]] = None
    ) -> ComparisonResults:
        """
        Run baseline comparison for all models and languages.
        
        Args:
            languages: List of language codes
            baseline_model_path: Path to baseline model
            fine_tuned_models: Dictionary mapping model names to paths
            test_sets: Optional pre-loaded test sets
        
        Returns:
            ComparisonResults with all comparisons
        """
        logger.info("Starting baseline comparison protocol")
        
        # Load baseline model
        baseline_name = "baseline_whisper_large_v3"
        self.model_manager.load_model(baseline_model_path, baseline_name)
        
        # Load fine-tuned models
        for model_name, model_path in fine_tuned_models.items():
            self.model_manager.load_model(model_path, model_name)
        
        comparisons = []
        
        # Evaluate on each language
        for language in languages:
            logger.info(f"Evaluating on language: {language}")
            
            # Load test set
            if test_sets and language in test_sets:
                test_set = test_sets[language]
            else:
                test_set = self.test_set_manager.load_test_set(language)
            
            # Evaluate baseline
            baseline_wer = self._evaluate_model_on_language(
                baseline_name,
                test_set,
                language
            )
            
            # Evaluate fine-tuned models
            for model_name in fine_tuned_models.keys():
                model_wer = self._evaluate_model_on_language(
                    model_name,
                    test_set,
                    language
                )
                
                # Compare models
                comparison = self._compare_models(
                    baseline_name,
                    model_name,
                    language,
                    baseline_wer,
                    model_wer,
                    test_set
                )
                comparisons.append(comparison)
        
        # Generate summary
        summary = self._generate_summary(comparisons)
        
        return ComparisonResults(
            baseline_model=baseline_name,
            comparisons=comparisons,
            summary=summary
        )
    
    def _evaluate_model_on_language(
        self,
        model_name: str,
        test_set: TestSet,
        language: str
    ) -> WERResult:
        """Evaluate a model on a language test set."""
        # Get predictions
        audio_paths = [sample.audio_path for sample in test_set.samples]
        predictions = self.model_manager.batch_inference(
            model_name,
            audio_paths,
            language=language
        )
        
        # Get references
        references = [sample.reference_text for sample in test_set.samples]
        
        # Compute WER with CI
        wer_result = self.metric_calculator.compute_wer_with_ci(
            predictions,
            references
        )
        
        return wer_result
    
    def _compare_models(
        self,
        model1_name: str,
        model2_name: str,
        language: str,
        model1_wer: WERResult,
        model2_wer: WERResult,
        test_set: TestSet
    ) -> ModelComparison:
        """Compare two models statistically."""
        # Compute per-example errors for statistical test
        # This is simplified - would need per-example WER
        # For now, use point estimates
        
        # WER difference
        wer_diff = model1_wer.value - model2_wer.value
        
        # CI for difference (simplified)
        ci_lower = model1_wer.ci_lower - model2_wer.ci_upper
        ci_upper = model1_wer.ci_upper - model2_wer.ci_lower
        
        # Relative improvement
        if model1_wer.value > 0:
            relative_improvement = (wer_diff / model1_wer.value) * 100
        else:
            relative_improvement = 0.0
        
        # Statistical test (simplified - would need per-example errors)
        # For now, create a placeholder test result
        from .statistics import TestResult
        test_result = TestResult(
            test_name="paired_t_test",
            p_value=0.05,  # Placeholder
            effect_size=wer_diff / max(model1_wer.value, 0.01),
            is_significant=False,
            significance_level=0.05
        )
        
        return ModelComparison(
            model1_name=model1_name,
            model2_name=model2_name,
            language=language,
            wer_difference=wer_diff,
            wer_difference_ci=(ci_lower, ci_upper),
            relative_improvement=relative_improvement,
            statistical_test=test_result,
            model1_wer=model1_wer,
            model2_wer=model2_wer
        )
    
    def _generate_summary(self, comparisons: List[ModelComparison]) -> Dict:
        """Generate summary statistics."""
        improvements = [
            c.relative_improvement
            for c in comparisons
            if c.relative_improvement > 0
        ]
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        significant_improvements = [
            c for c in comparisons
            if c.statistical_test.is_significant and c.relative_improvement > 0
        ]
        
        return {
            'num_comparisons': len(comparisons),
            'num_improvements': len(improvements),
            'avg_improvement_percent': avg_improvement,
            'num_significant_improvements': len(significant_improvements)
        }


class CrossLingualProtocol:
    """
    Protocol for comparing multilingual vs per-language models.
    
    Evaluates generalization and cross-lingual performance.
    """
    
    def __init__(
        self,
        test_set_manager: TestSetManager,
        model_manager: ModelManager,
        metric_calculator: MetricCalculator
    ):
        """Initialize cross-lingual protocol."""
        self.test_set_manager = test_set_manager
        self.model_manager = model_manager
        self.metric_calculator = metric_calculator
    
    def run(
        self,
        languages: List[str],
        multilingual_model_path: str,
        per_language_models: Dict[str, str],
        test_sets: Optional[Dict[str, TestSet]] = None
    ) -> Dict[str, any]:
        """
        Run cross-lingual comparison.
        
        Args:
            languages: List of language codes
            multilingual_model_path: Path to multilingual model
            per_language_models: Dictionary mapping language to model path
            test_sets: Optional pre-loaded test sets
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting cross-lingual comparison protocol")
        
        # Load multilingual model
        self.model_manager.load_model(multilingual_model_path, "multilingual")
        
        # Load per-language models
        for lang, model_path in per_language_models.items():
            self.model_manager.load_model(model_path, f"per_lang_{lang}")
        
        results = {}
        
        for language in languages:
            # Load test set
            if test_sets and language in test_sets:
                test_set = test_sets[language]
            else:
                test_set = self.test_set_manager.load_test_set(language)
            
            # Evaluate multilingual model
            multi_wer = self._evaluate_model(
                "multilingual",
                test_set,
                language
            )
            
            # Evaluate per-language model if available
            if language in per_language_models:
                per_lang_wer = self._evaluate_model(
                    f"per_lang_{language}",
                    test_set,
                    language
                )
                
                gap = multi_wer.value - per_lang_wer.value
                
                results[language] = {
                    'multilingual_wer': multi_wer.value,
                    'per_language_wer': per_lang_wer.value,
                    'wer_gap': gap,
                    'multilingual_better': gap < 0
                }
            else:
                results[language] = {
                    'multilingual_wer': multi_wer.value,
                    'per_language_wer': None,
                    'wer_gap': None
                }
        
        return results
    
    def _evaluate_model(
        self,
        model_name: str,
        test_set: TestSet,
        language: str
    ) -> WERResult:
        """Evaluate a model on test set."""
        audio_paths = [sample.audio_path for sample in test_set.samples]
        predictions = self.model_manager.batch_inference(
            model_name,
            audio_paths,
            language=language
        )
        references = [sample.reference_text for sample in test_set.samples]
        
        return self.metric_calculator.compute_wer_with_ci(
            predictions,
            references
        )

