
import sys
import os
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import MetricCalculator
from src.evaluation.robustness import RobustnessTester
from src.data.audio_preprocessor import add_noise, change_speed
from src.evaluation.error_analyzer import ErrorAnalyzer

logging.basicConfig(level=logging.INFO)

def test_metrics():
    print("Testing Metrics...")
    calculator = MetricCalculator(num_bootstrap_samples=10)
    
    # Test PER
    preds = ["ngiyabonga", "kakhulu"]
    refs = ["ngiyabonga", "kakhulu"]
    per_result = calculator.compute_per_with_ci(preds, refs, "zu")
    print(f"PER (perfect match): {per_result.value}")
    assert per_result.value == 0.0
    
    preds_err = ["niyabonga", "kakulu"] # Missing 'g' and 'h'
    refs_err = ["ngiyabonga", "kakhulu"]
    per_result_err = calculator.compute_per_with_ci(preds_err, refs_err, "zu")
    print(f"PER (with errors): {per_result_err.value}")
    assert per_result_err.value > 0.0

    # Test NER
    preds_ner = ["I live in eGoli"]
    refs_ner = ["I live in eGoli"]
    ner_result = calculator.compute_ner_f1_with_ci(preds_ner, refs_ner)
    print(f"NER F1 (perfect match): {ner_result.value}")
    assert ner_result.value == 1.0
    
    preds_ner_err = ["I live in egoli"] # Lowercase
    refs_ner_err = ["I live in eGoli"]
    ner_result_err = calculator.compute_ner_f1_with_ci(preds_ner_err, refs_ner_err)
    print(f"NER F1 (case error): {ner_result_err.value}")
    # Should be lower because 'egoli' might not be caught or won't match 'eGoli'
    
def test_robustness():
    print("\nTesting Robustness...")
    # Create dummy audio
    audio = np.random.uniform(-1, 1, 16000) # 1 sec
    
    # Test noise
    noisy = add_noise(audio, 'white', 20)
    assert len(noisy) == len(audio)
    assert not np.array_equal(audio, noisy)
    print("Noise injection successful")
    
    # Test speed
    fast = change_speed(audio, 1.1)
    # Length should change roughly by 1/1.1
    expected_len = int(len(audio) / 1.1)
    # Allow some tolerance
    assert abs(len(fast) - expected_len) < 1000
    print("Speed perturbation successful")

def test_error_analysis():
    print("\nTesting Error Analysis...")
    analyzer = ErrorAnalyzer()
    
    preds = ["ngihamba", "abantu"]
    refs = ["ngiyahamba", "abantu"]
    
    # 'ngihamba' vs 'ngiyahamba' -> missing 'ya' (infix/morphology)
    # But our simple analyzer checks prefixes/suffixes
    
    preds_morph = ["hamba"]
    refs_morph = ["ukuhamba"] # 'uku' prefix
    
    morph_errors = analyzer.analyze_morphological_errors(preds_morph, refs_morph, "zu")
    print(f"Morphological Errors: {morph_errors}")
    assert morph_errors['prefix_errors'].get('missing_uku', 0) > 0

if __name__ == "__main__":
    test_metrics()
    test_robustness()
    test_error_analysis()
    print("\nAll tests passed!")
