# Design Document: Model Evaluation Framework

## Overview

This document describes the architecture and design of a comprehensive evaluation framework for benchmarking fine-tuned Whisper-large-v3 models against baseline models across South African languages. The framework provides statistically rigorous performance comparisons, detailed error analysis, robustness testing, and actionable deployment recommendations.

### Design Principles

1. **Statistical Rigor**: All metrics include confidence intervals and significance testing
2. **Reproducibility**: Complete artifact versioning and deterministic evaluation
3. **Modularity**: Independent components for metrics, testing, visualization, and reporting
4. **Performance**: Optimized for completion within 24 hours for full test suite
5. **Actionability**: Generate clear recommendations for model deployment and improvement

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Evaluation Orchestrator                        │
│         (Coordinates evaluation protocols and workflows)         │
└────────────────┬────────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Baseline  │ │Cross-    │ │Resource  │
│Compare   │ │Lingual   │ │Strategy  │
│Protocol  │ │Protocol  │ │Protocol  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────────┐      ┌─────────────────┐
│ Metric Engine   │      │ Model Manager   │
│                 │      │                 │
│ • WER/CER/PER   │      │ • Loader        │
│ • Bootstrap CI  │      │ • Inference     │
│ • Statistical   │      │ • Batch Proc    │
│   Tests         │      │ • Cache         │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └────────┬───────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────────┐      ┌─────────────────┐
│ Test Set        │      │ Error Analyzer  │
│ Manager         │      │                 │
│                 │      │ • Categorizer   │
│ • Loader        │      │ • Confusion     │
│ • Validator     │      │ • Patterns      │
│ • Balancer      │      │ • Fairness      │
└─────────────────┘      └─────────────────┘
         │                        │
         └────────┬───────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────────────┐      ┌─────────────────┐
│ Report          │      │ Dashboard       │
│ Generator       │      │ Server          │
│                 │      │                 │
│ • PDF/HTML      │      │ • Interactive   │
│ • Visualizations│      │ • Filtering     │
│ • Recommendations│     │ • Drill-down    │
└─────────────────┘      └─────────────────┘
```

### Component Responsibilities

**Evaluation Orchestrator**: Coordinates evaluation protocols, manages model-language combinations, handles distributed execution, and aggregates results.

**Protocol Modules**: Implement specific evaluation strategies (baseline comparison, cross-lingual, resource strategy analysis) with appropriate statistical tests.

**Metric Engine**: Computes all evaluation metrics (WER, CER, PER, NER F1) with bootstrap confidence intervals and statistical significance tests.

**Model Manager**: Handles model loading, batch inference, feature caching, and performance profiling across multiple models.

**Test Set Manager**: Manages held-out test sets, ensures balanced sampling, validates data integrity, and handles cross-validation splits.

**Error Analyzer**: Categorizes errors by type, generates confusion matrices, identifies patterns, and performs fairness analysis.

**Report Generator**: Creates comprehensive PDF/HTML reports with visualizations, statistical summaries, and deployment recommendations.

**Dashboard Server**: Provides interactive web interface for exploring results, comparing models, and drilling down into error analysis.

## Components and Interfaces

### 1. Test Set Management Module

#### TestSetManager (`test_set_manager.py`)

**Purpose**: Manage held-out test sets with balanced sampling and integrity verification.

**Key Classes**:

```python
class TestSetManager:
    """
    Manages test set creation, loading, and validation.
    
    Attributes:
        test_set_dir: Directory for storing test sets
        min_samples: Minimum samples per language (50)
        max_samples: Maximum samples per language (1000)
        seed: Random seed for reproducibility
    """
    
    def create_test_set(self, language: str, val_data_path: str) -> TestSet:
        """
        Create balanced test set from validation data.
        
        Args:
            language: Language code
            val_data_path: Path to validation JSONL
            
        Returns:
            TestSet with samples and metadata
        """
    
    def load_test_set(self, language: str) -> TestSet:
        """Load test set with checksum verification"""
    
    def create_cross_lingual_test_set(self, languages: List[str]) -> TestSet:
        """Create balanced multilingual test set"""
    
    def validate_test_set(self, test_set: TestSet) -> bool:
        """Verify test set integrity and checksums"""
```

**Design Decisions**:
- Store test sets as JSONL with separate metadata JSON containing checksums
- Use stratified sampling when demographic metadata is available
- Cap at 1000 samples per language for computational efficiency while maintaining statistical power
- Compute SHA-256 checksums for audio files and transcriptions
- Support both per-language and cross-lingual test sets

#### TestSet Data Model

```python
@dataclass
class TestSet:
    """Test set with metadata"""
    language: str
    samples: List[TestSample]
    metadata: TestSetMetadata
    checksum: str
    
@dataclass
class TestSample:
    """Individual test sample"""
    audio_path: str
    reference_text: str
    audio_duration: float
    audio_checksum: str
    metadata: Optional[Dict] = None  # Demographics, quality, etc.
    
@dataclass
class TestSetMetadata:
    """Test set metadata"""
    creation_date: str
    source_dataset: str
    num_samples: int
    total_duration: float
    demographic_distribution: Optional[Dict] = None
```

### 2. Metric Computation Module

#### MetricCalculator (`metric_calculator.py`)

**Purpose**: Compute evaluation metrics with statistical rigor including confidence intervals.

**Key Classes**:

```python
class MetricCalculator:
    """
    Computes evaluation metrics with bootstrap confidence intervals.
    
    Attributes:
        num_bootstrap_samples: Number of bootstrap iterations (1000)
        confidence_level: Confidence level for intervals (0.95)
        seed: Random seed for reproducibility
    """
    
    def compute_wer_with_ci(self, predictions: List[str], 
                           references: List[str]) -> MetricResult:
        """
        Compute WER with 95% confidence intervals.
        
        Returns:
            MetricResult with point estimate, CI, and error breakdown
        """
    
    def compute_cer_with_ci(self, predictions: List[str], 
                           references: List[str]) -> MetricResult:
        """Compute CER with Unicode normalization and CI"""
    
    def compute_per_with_ci(self, predictions: List[str], 
                           references: List[str],
                           language: str) -> MetricResult:
        """Compute phoneme error rate with language-specific inventory"""
    
    def compute_ner_f1_with_ci(self, predictions: List[str], 
                              references: List[str]) -> MetricResult:
        """Compute NER F1 score with CI"""
    
    def _bootstrap_resample(self, predictions: List[str], 
                           references: List[str],
                           metric_fn: Callable) -> List[float]:
        """Perform bootstrap resampling for confidence intervals"""
```

**Design Decisions**:
- Use jiwer library for WER/CER computation (industry standard)
- Implement BCa (bias-corrected and accelerated) bootstrap for accurate CIs
- Apply NFC Unicode normalization before CER computation
- Use language-specific phoneme inventories from linguistic databases
- Cache bootstrap samples for faster repeated computations

#### MetricResult Data Model

```python
@dataclass
class MetricResult:
    """Metric with confidence interval"""
    metric_name: str
    point_estimate: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    num_samples: int
    error_breakdown: Optional[Dict] = None  # Sub/Ins/Del for WER
    bootstrap_distribution: Optional[np.ndarray] = None
```

#### WERCalculator (`wer_calculator.py`)

**Purpose**: Detailed WER computation with alignment and error categorization.

**Key Functions**:

```python
def compute_wer_detailed(predictions: List[str], 
                        references: List[str]) -> WERResult:
    """
    Compute WER with detailed error breakdown.
    
    Returns:
        WERResult with substitutions, insertions, deletions, and alignment
    """

def compute_alignment(prediction: str, reference: str) -> Alignment:
    """Compute optimal alignment using dynamic programming"""

def categorize_errors(alignment: Alignment) -> ErrorCategories:
    """Categorize errors by linguistic type"""
```

### 3. Statistical Testing Module

#### StatisticalTester (`statistical_tests.py`)

**Purpose**: Perform statistical significance tests with multiple comparison correction.

**Key Classes**:

```python
class StatisticalTester:
    """
    Performs statistical significance tests.
    
    Attributes:
        alpha: Significance level (0.05)
        correction_method: Multiple comparison correction ("bonferroni")
    """
    
    def paired_t_test(self, errors_a: np.ndarray, 
                     errors_b: np.ndarray) -> TestResult:
        """
        Perform paired t-test on per-example errors.
        
        Args:
            errors_a: Per-example error rates for model A
            errors_b: Per-example error rates for model B
            
        Returns:
            TestResult with p-value and effect size
        """
    
    def permutation_test(self, errors_a: np.ndarray, 
                        errors_b: np.ndarray,
                        num_permutations: int = 10000) -> TestResult:
        """Non-parametric permutation test"""
    
    def compute_effect_size(self, errors_a: np.ndarray, 
                           errors_b: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
    
    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
    
    def anova_test(self, error_groups: List[np.ndarray]) -> TestResult:
        """One-way ANOVA for comparing multiple models"""
```

**Design Decisions**:
- Use scipy.stats for statistical tests (well-tested implementations)
- Default to paired tests since same test set is used for all models
- Apply Bonferroni correction conservatively to control Type I error
- Report both p-values and effect sizes for practical significance
- Use permutation tests when normality assumptions are violated

#### TestResult Data Model

```python
@dataclass
class TestResult:
    """Statistical test result"""
    test_name: str
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool  # p < alpha
    confidence_level: float
    test_statistic: float
    degrees_of_freedom: Optional[int] = None
```

### 4. Model Management Module

#### ModelManager (`model_manager.py`)

**Purpose**: Handle model loading, inference, and performance profiling.

**Key Classes**:

```python
class ModelManager:
    """
    Manages model loading and inference.
    
    Attributes:
        device: Inference device (cuda/cpu)
        batch_size: Batch size for inference
        cache_dir: Directory for caching embeddings
        use_fp16: Whether to use FP16 inference
    """
    
    def load_model(self, model_path: str, language: Optional[str] = None) -> Model:
        """Load model with optimizations"""
    
    def batch_inference(self, model: Model, 
                       test_set: TestSet) -> List[str]:
        """Perform batch inference with progress tracking"""
    
    def profile_inference(self, model: Model, 
                         test_set: TestSet) -> PerformanceMetrics:
        """Profile inference performance (latency, throughput, memory)"""
    
    def cache_embeddings(self, model: Model, 
                        test_set: TestSet) -> str:
        """Cache encoder embeddings for reuse across models"""
    
    def load_cached_embeddings(self, cache_key: str) -> torch.Tensor:
        """Load cached embeddings"""
```

**Design Decisions**:
- Use FP16 inference by default for 2x speedup
- Cache encoder embeddings since they're identical across fine-tuned models
- Implement dynamic batching to maximize GPU utilization
- Profile memory usage using torch.cuda.max_memory_allocated()
- Support distributed inference across multiple GPUs

#### PerformanceMetrics Data Model

```python
@dataclass
class PerformanceMetrics:
    """Inference performance metrics"""
    throughput_samples_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    peak_memory_mb: float
    model_size_mb: float
    num_parameters: int
```

### 5. Error Analysis Module

#### ErrorAnalyzer (`error_analyzer.py`)

**Purpose**: Categorize and analyze errors by linguistic, acoustic, and demographic factors.

**Key Classes**:

```python
class ErrorAnalyzer:
    """
    Analyzes and categorizes prediction errors.
    
    Attributes:
        language: Language code for linguistic analysis
        phoneme_inventory: Language-specific phoneme set
    """
    
    def categorize_errors(self, predictions: List[str], 
                         references: List[str],
                         test_samples: List[TestSample]) -> ErrorAnalysis:
        """
        Categorize errors by type.
        
        Returns:
            ErrorAnalysis with linguistic, acoustic, demographic breakdowns
        """
    
    def analyze_phonetic_confusions(self, predictions: List[str], 
                                   references: List[str]) -> ConfusionMatrix:
        """Generate phoneme confusion matrix"""
    
    def analyze_morphological_errors(self, predictions: List[str], 
                                    references: List[str]) -> MorphologyAnalysis:
        """Analyze prefix/suffix errors for Bantu languages"""
    
    def analyze_oov_errors(self, predictions: List[str], 
                          references: List[str],
                          vocabulary: Set[str]) -> OOVAnalysis:
        """Analyze out-of-vocabulary word errors"""
    
    def analyze_demographic_disparities(self, predictions: List[str], 
                                       references: List[str],
                                       test_samples: List[TestSample]) -> FairnessMetrics:
        """Compute performance disparities across demographic groups"""
```

**Design Decisions**:
- Use language-specific phoneme inventories from Phoible database
- Implement Levenshtein alignment for error attribution
- Categorize errors hierarchically (linguistic → phonetic → specific phoneme)
- Compute fairness metrics only when demographic metadata is available
- Generate interactive confusion matrices using plotly

#### ErrorAnalysis Data Model

```python
@dataclass
class ErrorAnalysis:
    """Comprehensive error analysis"""
    linguistic_errors: LinguisticErrors
    acoustic_errors: Optional[AcousticErrors]
    demographic_errors: Optional[DemographicErrors]
    confusion_matrices: Dict[str, ConfusionMatrix]
    
@dataclass
class LinguisticErrors:
    """Linguistic error breakdown"""
    phonetic_confusions: List[PhoneticConfusion]
    morphological_errors: List[MorphologyError]
    lexical_errors: List[LexicalError]
    oov_rate: float
    
@dataclass
class PhoneticConfusion:
    """Phoneme confusion pattern"""
    predicted_phoneme: str
    reference_phoneme: str
    count: int
    examples: List[str]
```

### 6. Evaluation Protocol Module

#### BaselineComparisonProtocol (`protocols/baseline_comparison.py`)

**Purpose**: Compare fine-tuned models against base Whisper-large-v3.

**Key Classes**:

```python
class BaselineComparisonProtocol:
    """
    Protocol for comparing models against baseline.
    
    Attributes:
        baseline_model_name: Base model identifier
        test_set_manager: Test set manager
        model_manager: Model manager
        metric_calculator: Metric calculator
        statistical_tester: Statistical tester
    """
    
    def run(self, fine_tuned_models: List[str], 
           languages: List[str]) -> ComparisonResults:
        """
        Run baseline comparison protocol.
        
        For each language and model:
        1. Evaluate baseline model
        2. Evaluate fine-tuned model
        3. Compute WER difference with CI
        4. Test statistical significance
        5. Calculate relative improvement
        
        Returns:
            ComparisonResults with all comparisons
        """
    
    def evaluate_model_on_language(self, model_path: str, 
                                  language: str) -> EvaluationResult:
        """Evaluate single model on single language"""
    
    def compare_models(self, baseline_result: EvaluationResult, 
                      finetuned_result: EvaluationResult) -> ModelComparison:
        """Compare two models with statistical tests"""
```

**Design Decisions**:
- Evaluate baseline model once per language and cache results
- Use paired t-tests since same test set is used
- Report both absolute WER difference and relative improvement percentage
- Generate comparison visualizations automatically
- Support parallel evaluation across languages

#### CrossLingualProtocol (`protocols/cross_lingual.py`)

**Purpose**: Evaluate multilingual model generalization across languages.

**Key Classes**:

```python
class CrossLingualProtocol:
    """
    Protocol for cross-lingual evaluation.
    
    Tests:
    - Multilingual model on all languages
    - Per-language models on same languages
    - Zero-shot performance on held-out languages
    """
    
    def run(self, multilingual_model: str, 
           per_language_models: Dict[str, str],
           languages: List[str]) -> CrossLingualResults:
        """
        Run cross-lingual evaluation protocol.
        
        Returns:
            CrossLingualResults with generalization analysis
        """
    
    def evaluate_zero_shot(self, model: str, 
                          target_language: str) -> EvaluationResult:
        """Evaluate model on language not seen during training"""
    
    def compute_generalization_gap(self, multilingual_result: EvaluationResult,
                                  specialist_result: EvaluationResult) -> float:
        """Compute performance gap between multilingual and specialist"""
```

#### ResourceStrategyProtocol (`protocols/resource_strategy.py`)

**Purpose**: Analyze effectiveness of training strategies by resource tier.

**Key Classes**:

```python
class ResourceStrategyProtocol:
    """
    Protocol for resource strategy analysis.
    
    Compares:
    - High-resource: full fine-tuning
    - Medium-resource: extended training
    - Low-resource: transfer learning
    - Multilingual: balanced sampling
    """
    
    def run(self, models_by_strategy: Dict[str, List[str]]) -> StrategyResults:
        """
        Run resource strategy analysis.
        
        Returns:
            StrategyResults with cost-benefit analysis
        """
    
    def compute_efficiency_score(self, performance: float, 
                                training_cost: float) -> float:
        """Calculate performance per training hour"""
    
    def analyze_transfer_learning_benefit(self, 
                                         with_transfer: List[EvaluationResult],
                                         without_transfer: List[EvaluationResult]) -> TransferAnalysis:
        """Measure benefit of transfer learning for low-resource languages"""
```

### 7. Robustness Testing Module

#### RobustnessTester (`robustness_tester.py`)

**Purpose**: Test model robustness under various audio perturbations.

**Key Classes**:

```python
class RobustnessTester:
    """
    Tests model robustness to audio degradations.
    
    Perturbations:
    - Noise injection (white, babble, street)
    - Speed perturbation (0.9x, 1.0x, 1.1x)
    - Codec degradation (MP3, Opus, AAC)
    """
    
    def test_noise_robustness(self, model: str, 
                             test_set: TestSet,
                             noise_types: List[str],
                             snr_levels: List[float]) -> RobustnessResults:
        """Test robustness to noise at various SNR levels"""
    
    def test_speed_robustness(self, model: str, 
                             test_set: TestSet,
                             speed_factors: List[float]) -> RobustnessResults:
        """Test robustness to speed perturbations"""
    
    def test_codec_robustness(self, model: str, 
                             test_set: TestSet,
                             codecs: List[str]) -> RobustnessResults:
        """Test robustness to codec compression"""
    
    def apply_noise(self, audio: np.ndarray, 
                   noise_type: str, 
                   snr_db: float) -> np.ndarray:
        """Apply noise to audio at specified SNR"""
    
    def apply_speed_perturbation(self, audio: np.ndarray, 
                                speed_factor: float) -> np.ndarray:
        """Apply speed perturbation using librosa"""
    
    def apply_codec_compression(self, audio: np.ndarray, 
                               codec: str) -> np.ndarray:
        """Apply codec compression and decompression"""
```

**Design Decisions**:
- Use pyroomacoustics for realistic noise simulation
- Use librosa for speed perturbation (time-stretching)
- Use pydub for codec compression/decompression
- Test at multiple SNR levels: -5dB, 0dB, 5dB, 10dB, 15dB, 20dB
- Report WER degradation relative to clean audio

### 8. Visualization Module

#### Visualizer (`visualizer.py`)

**Purpose**: Generate visualizations for evaluation results.

**Key Classes**:

```python
class Visualizer:
    """
    Generates visualizations for evaluation results.
    
    Visualizations:
    - Radar charts for multi-metric comparison
    - Bar charts with error bars for WER comparison
    - Heatmaps for confusion matrices
    - Scatter plots for effect size vs p-value
    - Line plots for robustness degradation
    """
    
    def create_radar_chart(self, models: List[str], 
                          metrics: Dict[str, List[float]]) -> Figure:
        """Create radar chart for multi-metric comparison"""
    
    def create_wer_comparison_chart(self, results: ComparisonResults) -> Figure:
        """Create bar chart with confidence intervals"""
    
    def create_confusion_matrix_heatmap(self, confusion_matrix: ConfusionMatrix) -> Figure:
        """Create interactive confusion matrix heatmap"""
    
    def create_significance_plot(self, comparisons: List[ModelComparison]) -> Figure:
        """Create scatter plot of effect size vs p-value"""
    
    def create_robustness_plot(self, robustness_results: RobustnessResults) -> Figure:
        """Create line plot showing WER degradation"""
```

**Design Decisions**:
- Use plotly for interactive visualizations
- Use matplotlib for static PDF report figures
- Include error bars (confidence intervals) on all charts
- Use colorblind-friendly color palettes
- Export figures in both PNG and SVG formats

### 9. Report Generation Module

#### ReportGenerator (`report_generator.py`)

**Purpose**: Generate comprehensive evaluation reports in PDF and HTML formats.

**Key Classes**:

```python
class ReportGenerator:
    """
    Generates comprehensive evaluation reports.
    
    Report sections:
    - Executive summary
    - Methodology
    - Results overview
    - Detailed analysis
    - Recommendations
    """
    
    def generate_report(self, evaluation_results: EvaluationResults,
                       output_format: str = "html") -> str:
        """
        Generate comprehensive report.
        
        Args:
            evaluation_results: All evaluation results
            output_format: "html" or "pdf"
            
        Returns:
            Path to generated report
        """
    
    def generate_executive_summary(self, results: EvaluationResults) -> str:
        """Generate executive summary with key findings"""
    
    def generate_methodology_section(self) -> str:
        """Document evaluation methodology"""
    
    def generate_results_section(self, results: EvaluationResults) -> str:
        """Generate detailed results section with visualizations"""
    
    def generate_recommendations(self, results: EvaluationResults) -> List[Recommendation]:
        """Generate actionable recommendations"""
```

**Design Decisions**:
- Use Jinja2 templates for HTML reports
- Use WeasyPrint for PDF generation from HTML
- Include interactive visualizations in HTML reports
- Generate static images for PDF reports
- Structure reports hierarchically with table of contents
- Include methodology section for reproducibility

#### Recommendation Data Model

```python
@dataclass
class Recommendation:
    """Deployment or improvement recommendation"""
    category: str  # "deployment", "improvement", "data_collection"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    rationale: str
    expected_impact: Optional[str] = None
```

### 10. Dashboard Module

#### DashboardServer (`dashboard_server.py`)

**Purpose**: Provide interactive web interface for exploring evaluation results.

**Key Classes**:

```python
class DashboardServer:
    """
    Interactive dashboard for evaluation results.
    
    Features:
    - Model selection and comparison
    - Interactive filtering by language, metric, model type
    - Drill-down into error categories
    - Temporal comparison across evaluation runs
    """
    
    def __init__(self, results_dir: str, port: int = 8050):
        """Initialize Dash app"""
    
    def create_layout(self) -> html.Div:
        """Create dashboard layout"""
    
    def register_callbacks(self):
        """Register interactive callbacks"""
    
    def run(self, debug: bool = False):
        """Start dashboard server"""
```

**Design Decisions**:
- Use Plotly Dash for interactive dashboard
- Store results in SQLite database for efficient querying
- Support filtering by multiple dimensions simultaneously
- Implement drill-down from summary to detailed error analysis
- Cache expensive computations for responsive UI

### 11. Evaluation Orchestrator

#### EvaluationOrchestrator (`evaluation_orchestrator.py`)

**Purpose**: Coordinate complete evaluation workflow across all protocols.

**Key Classes**:

```python
class EvaluationOrchestrator:
    """
    Orchestrates complete evaluation workflow.
    
    Workflow:
    1. Load test sets
    2. Run baseline comparison protocol
    3. Run cross-lingual protocol
    4. Run resource strategy protocol
    5. Run robustness testing
    6. Generate reports
    7. Launch dashboard
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize with configuration"""
    
    def run_full_evaluation(self, models: Dict[str, str],
                           languages: List[str]) -> EvaluationResults:
        """
        Run complete evaluation suite.
        
        Args:
            models: Dict mapping model names to paths
            languages: List of language codes to evaluate
            
        Returns:
            EvaluationResults with all protocol results
        """
    
    def run_protocol(self, protocol_name: str, **kwargs) -> ProtocolResults:
        """Run specific evaluation protocol"""
    
    def aggregate_results(self, protocol_results: List[ProtocolResults]) -> EvaluationResults:
        """Aggregate results from all protocols"""
    
    def generate_artifacts(self, results: EvaluationResults):
        """Generate reports, visualizations, and dashboard"""
```

**Design Decisions**:
- Support running individual protocols or full suite
- Implement checkpointing to resume interrupted evaluations
- Distribute evaluation across multiple GPUs when available
- Log progress and estimated time remaining
- Generate intermediate results for early insights

## Data Models

### Core Data Models

```python
@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    models: Dict[str, str]  # model_name -> path
    languages: List[str]
    test_set_dir: str
    output_dir: str
    protocols: List[str]  # Which protocols to run
    num_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    device: str = "cuda"
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42

@dataclass
class EvaluationResult:
    """Result for single model on single language"""
    model_name: str
    language: str
    wer: MetricResult
    cer: MetricResult
    per: Optional[MetricResult]
    ner_f1: Optional[MetricResult]
    predictions: List[str]
    references: List[str]
    inference_time: float
    performance_metrics: PerformanceMetrics

@dataclass
class ModelComparison:
    """Comparison between two models"""
    model_a: str
    model_b: str
    language: str
    wer_difference: float
    wer_difference_ci: Tuple[float, float]
    relative_improvement: float
    statistical_test: TestResult
    effect_size: float

@dataclass
class ComparisonResults:
    """Results from baseline comparison protocol"""
    baseline_results: Dict[str, EvaluationResult]  # language -> result
    finetuned_results: Dict[str, Dict[str, EvaluationResult]]  # model -> language -> result
    comparisons: List[ModelComparison]
    summary_statistics: Dict[str, float]

@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    config: EvaluationConfig
    baseline_comparison: ComparisonResults
    cross_lingual: Optional[CrossLingualResults]
    resource_strategy: Optional[StrategyResults]
    robustness: Optional[RobustnessResults]
    error_analysis: Dict[str, ErrorAnalysis]
    recommendations: List[Recommendation]
    timestamp: str
    evaluation_duration: float
```

## Error Handling

### Error Categories

1. **Test Set Errors**:
   - Missing test set → Create from validation data
   - Checksum mismatch → Warn and regenerate
   - Insufficient samples → Use all available samples

2. **Model Errors**:
   - Model loading failure → Log error, skip model
   - Inference failure → Log error, skip example
   - OOM during inference → Reduce batch size, retry

3. **Statistical Errors**:
   - Insufficient samples for bootstrap → Use analytical CI
   - Non-normal distribution → Use permutation test
   - Zero variance → Report exact values

4. **Visualization Errors**:
   - Missing data for plot → Skip plot, log warning
   - Plot generation failure → Use fallback visualization

### Error Recovery Strategy

```python
class EvaluationError(Exception):
    """Base exception for evaluation errors"""
    pass

def evaluate_with_recovery(orchestrator, max_retries=3):
    """
    Run evaluation with automatic error recovery.
    
    Strategies:
    - Model loading error: Skip model, continue with others
    - Inference OOM: Reduce batch size, retry
    - Statistical test failure: Use alternative test
    """
    results = {}
    for model_name, model_path in orchestrator.models.items():
        for attempt in range(max_retries):
            try:
                results[model_name] = orchestrator.evaluate_model(model_path)
                break
            except torch.cuda.OutOfMemoryError:
                orchestrator.batch_size //= 2
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                if attempt == max_retries - 1:
                    results[model_name] = None
    return results
```

## Testing Strategy

### Unit Tests

**Metric Computation Tests** (`tests/test_metrics.py`):
- Test WER computation with known examples
- Test bootstrap CI coverage (should contain true value 95% of time)
- Test CER with Unicode edge cases
- Test PER with language-specific phonemes

**Statistical Tests** (`tests/test_statistical.py`):
- Test paired t-test with synthetic data
- Test Bonferroni correction
- Test effect size computation
- Test permutation test convergence

**Error Analysis Tests** (`tests/test_error_analysis.py`):
- Test error categorization logic
- Test confusion matrix generation
- Test fairness metric computation

### Integration Tests

**End-to-End Evaluation** (`tests/test_e2e_evaluation.py`):
- Test complete evaluation on 2 models, 2 languages
- Test baseline comparison protocol
- Test report generation
- Test dashboard creation

**Protocol Tests** (`tests/test_protocols.py`):
- Test each protocol independently
- Test protocol with missing data
- Test protocol with single model

### Performance Tests

**Scalability Tests** (`tests/test_performance.py`):
- Benchmark evaluation time vs number of models
- Benchmark memory usage
- Test distributed evaluation
- Profile bottlenecks

## Deployment Considerations

### Computational Requirements

**Hardware**:
- GPU: NVIDIA GPU with 16GB+ VRAM for inference
- CPU: 16+ cores for parallel processing
- RAM: 64GB+ for large test sets
- Storage: 100GB+ for models, test sets, and results

**Software**:
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+
- Dependencies: transformers, jiwer, scipy, plotly, dash

### Performance Optimization

1. **Parallel Evaluation**: Distribute model-language combinations across GPUs
2. **Embedding Caching**: Cache encoder outputs for reuse
3. **Batch Inference**: Maximize GPU utilization with large batches
4. **Progressive Sampling**: Stop early if CI is sufficiently narrow
5. **Result Caching**: Cache intermediate results for faster re-runs

### Monitoring

**Evaluation Metrics**:
- Evaluation progress (models completed / total)
- Estimated time remaining
- GPU utilization
- Memory usage
- Error rate (failed evaluations / total)

## Future Enhancements

1. **Active Learning**: Identify examples where models disagree for labeling
2. **Adversarial Testing**: Generate adversarial examples to test robustness
3. **Explainability**: Add attention visualization for error analysis
4. **Continuous Evaluation**: Automatically evaluate new model checkpoints
5. **A/B Testing**: Support for online A/B testing in production
6. **Multi-modal Evaluation**: Extend to video or multi-speaker scenarios
7. **Real-time Dashboard**: Live updates as evaluation progresses
8. **Automated Recommendations**: ML-based recommendation generation
