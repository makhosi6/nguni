"""Configuration loader with inheritance and validation."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .models import ExperimentConfig


class ConfigLoader:
    """Load and merge YAML configuration files with inheritance."""
    
    def __init__(self, base_config_path: str = "configs/base_config.yaml"):
        """
        Initialize config loader.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = Path(base_config_path)
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file with environment variable substitution."""
        with open(path, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        content = os.path.expandvars(content)
        
        return yaml.safe_load(content) or {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_config(
        self,
        base_config: Optional[str] = None,
        tier_config: Optional[str] = None,
        language_config: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Load configuration with inheritance.
        
        Args:
            base_config: Path to base config (defaults to self.base_config_path)
            tier_config: Path to resource tier config (high/medium/low_resource.yaml)
            language_config: Path to language-specific config
            overrides: Dictionary of additional overrides
        
        Returns:
            Validated ExperimentConfig
        """
        # Start with base config
        if base_config is None:
            base_config = str(self.base_config_path)
        
        config = self._load_yaml(Path(base_config))
        
        # Apply tier config if provided
        if tier_config:
            tier_path = Path(tier_config)
            if not tier_path.is_absolute():
                tier_path = self.base_config_path.parent / tier_path
            if tier_path.exists():
                tier_config_dict = self._load_yaml(tier_path)
                config = self._merge_configs(config, tier_config_dict)
        
        # Apply language config if provided
        if language_config:
            lang_path = Path(language_config)
            if not lang_path.is_absolute():
                lang_path = self.base_config_path.parent / "languages" / lang_path
            if lang_path.exists():
                lang_config_dict = self._load_yaml(lang_path)
                config = self._merge_configs(config, lang_config_dict)
        
        # Apply direct overrides
        if overrides:
            config = self._merge_configs(config, overrides)
        
        # Validate and return
        try:
            return ExperimentConfig(**config)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def load_language_config(
        self,
        language_code: str,
        resource_tier: str = "high"
    ) -> ExperimentConfig:
        """
        Load configuration for a specific language.
        
        Args:
            language_code: Language code (e.g., 'nr', 'af', 'en')
            resource_tier: Resource tier ('high', 'medium', 'low')
        
        Returns:
            Validated ExperimentConfig
        """
        tier_map = {
            "high": "high_resource.yaml",
            "medium": "medium_resource.yaml",
            "low": "low_resource.yaml"
        }
        
        tier_config = tier_map.get(resource_tier, "high_resource.yaml")
        language_config = f"{language_code}-ZA.yaml"
        
        return self.load_config(
            tier_config=f"configs/{tier_config}",
            language_config=language_config
        )
    
    def load_multilingual_config(self) -> ExperimentConfig:
        """Load configuration for multilingual training."""
        return self.load_config(base_config="configs/multilingual.yaml")

