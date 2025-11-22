"""Statistical significance testing for model comparisons."""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from itertools import combinations


@dataclass
class TestResult:
    """Result of statistical test."""
    test_name: str
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool
    significance_level: float = 0.05
    statistic: Optional[float] = None
    degrees_of_freedom: Optional[int] = None


class StatisticalTester:
    """
    Performs statistical significance testing for model comparisons.
    
    Supports paired t-tests, permutation tests, and multiple comparison correction.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            significance_level: Significance level (alpha) for tests
        """
        self.significance_level = significance_level
    
    def paired_t_test(
        self,
        model1_errors: List[float],
        model2_errors: List[float]
    ) -> TestResult:
        """
        Perform paired t-test on per-example error rates.
        
        Args:
            model1_errors: Error rates for model 1 (per example)
            model2_errors: Error rates for model 2 (per example)
        
        Returns:
            TestResult with p-value, effect size, and significance
        """
        if len(model1_errors) != len(model2_errors):
            raise ValueError("Error lists must have same length for paired test")
        
        # Compute differences
        differences = np.array(model1_errors) - np.array(model2_errors)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_errors, model2_errors)
        
        # Compute Cohen's d (effect size)
        effect_size = self.compute_effect_size(model1_errors, model2_errors)
        
        is_significant = p_value < self.significance_level
        
        return TestResult(
            test_name="paired_t_test",
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            significance_level=self.significance_level,
            statistic=t_stat,
            degrees_of_freedom=len(differences) - 1
        )
    
    def permutation_test(
        self,
        model1_errors: List[float],
        model2_errors: List[float],
        num_permutations: int = 10000
    ) -> TestResult:
        """
        Perform permutation test (non-parametric alternative).
        
        Args:
            model1_errors: Error rates for model 1
            model2_errors: Error rates for model 2
            num_permutations: Number of permutations
        
        Returns:
            TestResult with p-value from permutation test
        """
        if len(model1_errors) != len(model2_errors):
            raise ValueError("Error lists must have same length")
        
        # Observed difference
        observed_diff = np.mean(model1_errors) - np.mean(model2_errors)
        
        # Combine errors
        all_errors = np.array(list(model1_errors) + list(model2_errors))
        n1 = len(model1_errors)
        
        # Permutation distribution
        permuted_diffs = []
        for _ in range(num_permutations):
            # Randomly shuffle
            shuffled = np.random.permutation(all_errors)
            perm_diff = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
            permuted_diffs.append(perm_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        # Effect size
        effect_size = self.compute_effect_size(model1_errors, model2_errors)
        
        is_significant = p_value < self.significance_level
        
        return TestResult(
            test_name="permutation_test",
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            significance_level=self.significance_level
        )
    
    def compute_effect_size(
        self,
        group1: List[float],
        group2: List[float]
    ) -> float:
        """
        Compute Cohen's d (standardized effect size).
        
        Args:
            group1: First group of values
            group2: Second group of values
        
        Returns:
            Cohen's d value
        """
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        return d
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        num_comparisons: Optional[int] = None
    ) -> List[float]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: List of p-values
            num_comparisons: Number of comparisons (defaults to len(p_values))
        
        Returns:
            List of corrected p-values
        """
        if num_comparisons is None:
            num_comparisons = len(p_values)
        
        corrected = [p * num_comparisons for p in p_values]
        # Cap at 1.0
        corrected = [min(p, 1.0) for p in corrected]
        
        return corrected
    
    def anova_test(
        self,
        *groups: List[List[float]]
    ) -> TestResult:
        """
        Perform ANOVA test for comparing multiple models.
        
        Args:
            *groups: Variable number of groups (each group is list of error rates)
        
        Returns:
            TestResult with F-statistic and p-value
        """
        # Flatten groups for scipy
        all_groups = [np.array(group) for group in groups]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*all_groups)
        
        # Effect size (eta-squared approximation)
        # Simplified calculation
        all_values = np.concatenate(all_groups)
        grand_mean = np.mean(all_values)
        
        between_group_var = sum(
            len(group) * (np.mean(group) - grand_mean)**2
            for group in all_groups
        )
        total_var = np.sum((all_values - grand_mean)**2)
        
        if total_var == 0:
            effect_size = 0.0
        else:
            effect_size = between_group_var / total_var
        
        is_significant = p_value < self.significance_level
        
        return TestResult(
            test_name="anova",
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            significance_level=self.significance_level,
            statistic=f_stat
        )
    
    def compare_multiple_models(
        self,
        model_errors: Dict[str, List[float]],
        correction: str = "bonferroni"
    ) -> Dict[Tuple[str, str], TestResult]:
        """
        Compare multiple models with multiple comparison correction.
        
        Args:
            model_errors: Dictionary mapping model names to error lists
            correction: Correction method ('bonferroni' or None)
        
        Returns:
            Dictionary mapping (model1, model2) pairs to TestResult
        """
        model_names = list(model_errors.keys())
        results = {}
        p_values = []
        
        # Perform all pairwise comparisons
        for model1, model2 in combinations(model_names, 2):
            result = self.paired_t_test(
                model_errors[model1],
                model_errors[model2]
            )
            results[(model1, model2)] = result
            p_values.append(result.p_value)
        
        # Apply correction if requested
        if correction == "bonferroni":
            num_comparisons = len(p_values)
            corrected_p_values = self.bonferroni_correction(
                p_values,
                num_comparisons
            )
            
            # Update results with corrected p-values
            for i, (pair, result) in enumerate(results.items()):
                corrected_p = corrected_p_values[i]
                result.p_value = corrected_p
                result.is_significant = corrected_p < self.significance_level
        
        return results

