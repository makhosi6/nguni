# Model Evaluation Framework

Comprehensive evaluation framework for benchmarking fine-tuned Whisper models against baseline models across South African languages.

## Features

- **Test Set Management**: Balanced sampling, checksum verification, cross-lingual test sets
- **Statistical Rigor**: Bootstrap confidence intervals, significance testing, effect sizes
- **Multiple Metrics**: WER, CER, PER, NER F1 with confidence intervals
- **Model Comparison**: Baseline comparison, cross-lingual analysis
- **Error Analysis**: Linguistic, acoustic, and demographic error categorization
- **Performance Profiling**: Latency, throughput, memory usage measurement

## Components

### Test Set Manager (`test_set_manager.py`)
- Creates balanced test sets (50-1000 samples per language)
- Validates data integrity with checksums
- Supports cross-lingual test sets
- Demographic balancing when metadata available

### Metric Calculator (`metrics.py`)
- WER with bootstrap confidence intervals
- CER with Unicode normalization
- PER (phoneme error rate) for linguistic analysis
- NER F1 for entity recognition
- All metrics include 95% confidence intervals via bootstrap resampling

### Statistical Tester (`statistics.py`)
- Paired t-tests for model comparisons
- Permutation tests (non-parametric)
- Cohen's d effect size calculation
- Bonferroni correction for multiple comparisons
- ANOVA for comparing multiple models

### Model Manager (`model_manager.py`)
- Efficient model loading with FP16 optimization
- Batch inference with progress tracking
- Performance profiling (latency, throughput, memory)
- Embedding caching support

### Error Analyzer (`error_analyzer.py`)
- Categorizes errors by type (linguistic, acoustic, demographic)
- Phonetic confusion analysis
- Morphological error detection for Bantu languages
- Demographic disparity analysis

### Evaluation Protocols (`protocols.py`)
- **BaselineComparisonProtocol**: Compare all models against baseline
- **CrossLingualProtocol**: Compare multilingual vs per-language models

### Evaluation Orchestrator (`orchestrator.py`)
- Coordinates complete evaluation workflow
- Manages test sets, models, and protocols
- Aggregates results and saves to disk

## Usage

### Basic Evaluation

```python
from src.evaluation.orchestrator import EvaluationOrchestrator, EvaluationConfig

config = EvaluationConfig(
    languages=['nr', 'tn', 'ts', 'xh', 'zu', 'af'],
    baseline_model_path='openai/whisper-large-v3',
    fine_tuned_models={
        'per_lang_nr': './checkpoints/nr/best',
        'per_lang_tn': './checkpoints/tn/best',
        'multilingual': './checkpoints/multilingual/best'
    },
    test_set_dir='./test_sets',
    output_dir='./evaluation_results'
)

orchestrator = EvaluationOrchestrator(config)
results = orchestrator.run_full_evaluation()
```

### Command Line

```bash
python scripts/evaluate_comprehensive.py \
    --languages nr tn ts xh zu af \
    --baseline-model openai/whisper-large-v3 \
    --fine-tuned-models models.json \
    --multilingual-model ./checkpoints/multilingual/best \
    --test-set-dir ./test_sets \
    --output-dir ./evaluation_results
```

## Output

Evaluation results are saved as JSON files containing:
- Baseline comparison results with statistical tests
- Cross-lingual comparison results
- Per-language metrics with confidence intervals
- Error analysis breakdowns
- Performance metrics

## Requirements

- Statistical rigor: All metrics include 95% confidence intervals
- Reproducibility: Deterministic seeding and checksum verification
- Performance: Optimized for batch inference and caching
- Extensibility: Modular design for adding new metrics/protocols

## Future Enhancements

- Cross-validation support
- Robustness testing (noise, speed, codec perturbations)
- Interactive dashboard
- PDF/HTML report generation
- Advanced visualization

