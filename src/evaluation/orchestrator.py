"""Evaluation orchestrator for coordinating evaluation protocols."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .test_set_manager import TestSetManager, TestSet
from .model_manager import ModelManager
from .metrics import MetricCalculator
from .statistics import StatisticalTester
from .protocols import BaselineComparisonProtocol, CrossLingualProtocol
from .error_analyzer import ErrorAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    languages: List[str]
    baseline_model_path: str
    fine_tuned_models: Dict[str, str]
    multilingual_model_path: Optional[str] = None
    per_language_models: Optional[Dict[str, str]] = None
    test_set_dir: str = "./test_sets"
    output_dir: str = "./evaluation_results"
    device: str = "cuda"
    batch_size: int = 16
    num_bootstrap_samples: int = 1000
    seed: int = 42


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    config: EvaluationConfig
    baseline_comparison: Optional[Any] = None
    cross_lingual: Optional[Dict] = None
    timestamp: str = ""
    results_path: str = ""


class EvaluationOrchestrator:
    """
    Orchestrates complete evaluation workflow.
    
    Coordinates test set management, model evaluation, protocol execution,
    and result aggregation.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation orchestrator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        
        # Initialize components
        self.test_set_manager = TestSetManager(
            test_set_dir=config.test_set_dir,
            seed=config.seed
        )
        
        self.model_manager = ModelManager(
            device=config.device
        )
        
        self.metric_calculator = MetricCalculator(
            num_bootstrap_samples=config.num_bootstrap_samples,
            seed=config.seed
        )
        
        self.statistical_tester = StatisticalTester()
        
        self.error_analyzer = ErrorAnalyzer()
        
        # Initialize protocols
        self.baseline_protocol = BaselineComparisonProtocol(
            self.test_set_manager,
            self.model_manager,
            self.metric_calculator,
            self.statistical_tester
        )
        
        self.cross_lingual_protocol = CrossLingualProtocol(
            self.test_set_manager,
            self.model_manager,
            self.metric_calculator
        )
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_evaluation(
        self,
        create_test_sets: bool = False,
        val_data_paths: Optional[Dict[str, str]] = None,
        audio_base_path: Optional[str] = None
    ) -> EvaluationResults:
        """
        Run complete evaluation suite.
        
        Args:
            create_test_sets: Whether to create test sets from validation data
            val_data_paths: Dictionary mapping language to validation JSONL path
            audio_base_path: Base directory for audio files
        
        Returns:
            EvaluationResults with all protocol results
        """
        logger.info("Starting full evaluation suite")
        
        # Create test sets if needed
        test_sets = {}
        if create_test_sets and val_data_paths and audio_base_path:
            logger.info("Creating test sets")
            for language in self.config.languages:
                if language in val_data_paths:
                    test_set = self.test_set_manager.create_test_set(
                        language=language,
                        val_data_path=val_data_paths[language],
                        audio_base_path=audio_base_path
                    )
                    test_sets[language] = test_set
        else:
            # Load existing test sets
            for language in self.config.languages:
                try:
                    test_set = self.test_set_manager.load_test_set(language)
                    test_sets[language] = test_set
                except FileNotFoundError:
                    logger.warning(f"Test set not found for {language}, skipping")
        
        # Run baseline comparison
        logger.info("Running baseline comparison protocol")
        baseline_comparison = self.baseline_protocol.run(
            languages=list(test_sets.keys()),
            baseline_model_path=self.config.baseline_model_path,
            fine_tuned_models=self.config.fine_tuned_models,
            test_sets=test_sets
        )
        
        # Run cross-lingual comparison if multilingual model provided
        cross_lingual = None
        if self.config.multilingual_model_path:
            logger.info("Running cross-lingual comparison protocol")
            cross_lingual = self.cross_lingual_protocol.run(
                languages=list(test_sets.keys()),
                multilingual_model_path=self.config.multilingual_model_path,
                per_language_models=self.config.per_language_models or {},
                test_sets=test_sets
            )
        
        # Create results
        results = EvaluationResults(
            config=self.config,
            baseline_comparison=baseline_comparison,
            cross_lingual=cross_lingual,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        results_path = self._save_results(results)
        results.results_path = str(results_path)
        
        logger.info(f"Evaluation completed. Results saved to: {results_path}")
        
        return results
    
    def _save_results(self, results: EvaluationResults) -> Path:
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        # Convert to dict for JSON serialization
        results_dict = {
            'config': asdict(results.config),
            'baseline_comparison': self._serialize_baseline_comparison(
                results.baseline_comparison
            ),
            'cross_lingual': results.cross_lingual,
            'timestamp': results.timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        return results_file
    
    def _serialize_baseline_comparison(self, comparison) -> Dict:
        """Serialize baseline comparison for JSON."""
        if comparison is None:
            return None
        
        return {
            'baseline_model': comparison.baseline_model,
            'summary': comparison.summary,
            'comparisons': [
                {
                    'model1_name': c.model1_name,
                    'model2_name': c.model2_name,
                    'language': c.language,
                    'wer_difference': c.wer_difference,
                    'wer_difference_ci': list(c.wer_difference_ci),
                    'relative_improvement': c.relative_improvement,
                    'model1_wer': {
                        'value': c.model1_wer.value,
                        'ci_lower': c.model1_wer.ci_lower,
                        'ci_upper': c.model1_wer.ci_upper
                    },
                    'model2_wer': {
                        'value': c.model2_wer.value,
                        'ci_lower': c.model2_wer.ci_lower,
                        'ci_upper': c.model2_wer.ci_upper
                    }
                }
                for c in comparison.comparisons
            ]
        }

