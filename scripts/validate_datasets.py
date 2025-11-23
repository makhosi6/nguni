#!/usr/bin/env python3
"""
Utility script for validating dataset structure and integrity.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List
from config import OUTPUT_DIR, DATASET_STRUCTURE, LANGUAGE_CODES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self):
        self.output_dir = OUTPUT_DIR

    def validate_directory_structure(self) -> bool:
        """Validate the basic directory structure for all languages."""
        valid = True
        for lang_code in LANGUAGE_CODES:
            lang_dir = self.output_dir / f"{lang_code}-ZA"
            if not lang_dir.exists():
                logger.error(f"Missing language directory: {lang_dir}")
                valid = False
                continue

            for subdir in DATASET_STRUCTURE:
                subdir_path = lang_dir / subdir
                if not subdir_path.exists():
                    logger.error(f"Missing subdirectory: {subdir_path}")
                    valid = False

        return valid

    def validate_file_extensions(self) -> bool:
        """Validate that files have correct extensions for their directories."""
        valid = True
        for lang_dir in self.output_dir.glob('*-ZA'):
            for category, valid_extensions in DATASET_STRUCTURE.items():
                category_dir = lang_dir / category
                if not category_dir.exists():
                    continue

                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        ext = file_path.suffix.lower().lstrip('.')
                        if ext not in valid_extensions:
                            logger.warning(
                                f"File {file_path.name} has invalid extension for "
                                f"category {category}"
                            )
                            valid = False

        return valid

    def generate_report(self) -> Dict:
        """Generate a report of the dataset organization."""
        report = {}
        for lang_dir in self.output_dir.glob('*-ZA'):
            lang_code = lang_dir.name.split('-')[0]
            report[lang_code] = {
                'name': LANGUAGE_CODES[lang_code]['name'],
                'categories': {}
            }

            for category in DATASET_STRUCTURE:
                category_dir = lang_dir / category
                if category_dir.exists():
                    files = list(category_dir.glob('*'))
                    report[lang_code]['categories'][category] = {
                        'file_count': len(files),
                        'size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    }

        return report

    def run_validation(self):
        """Run all validation checks and generate report."""
        logger.info("Starting dataset validation...")
        
        structure_valid = self.validate_directory_structure()
        extensions_valid = self.validate_file_extensions()
        
        if structure_valid and extensions_valid:
            logger.info("All validation checks passed!")
        else:
            logger.warning("Some validation checks failed. See logs above.")

        # Generate and save report
        report = self.generate_report()
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        return structure_valid and extensions_valid

def main():
    """Main execution function."""
    try:
        validator = DatasetValidator()
        validator.run_validation()
    except Exception as e:
        logger.error(f"An error occurred during validation: {str(e)}")

if __name__ == "__main__":
    main()
