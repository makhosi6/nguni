#!/usr/bin/env python3
"""
Main script for organizing South African speech datasets into a uniform format.
"""

import os
import shutil
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    DATASET_PATTERNS,
    LANGUAGE_CODES,
    DATASET_STRUCTURE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetOrganizer:
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.processed_dir = PROCESSED_DIR
        self._setup_output_directory()

    def _setup_output_directory(self):
        """Create the output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for lang_code in LANGUAGE_CODES:
            lang_dir = self.output_dir / f"{lang_code}-ZA"
            lang_dir.mkdir(exist_ok=True)
            # Create standard subdirectories
            for subdir in DATASET_STRUCTURE:
                (lang_dir / subdir).mkdir(exist_ok=True)

    def identify_dataset_family(self, dataset_name: str) -> Optional[str]:
        """Identify which family a dataset belongs to."""
        for family, pattern in DATASET_PATTERNS.items():
            if re.match(pattern, dataset_name, re.IGNORECASE):
                return family
        return None

    def get_language_code(self, dataset_name: str) -> Optional[str]:
        """Extract language code from dataset name."""
        for code in LANGUAGE_CODES:
            if code in dataset_name.lower() or LANGUAGE_CODES[code]['iso639_2'] in dataset_name.lower():
                return code
        return None
    # 
    def organize_dataset(self, dataset_path: Path):
        """Organize a single dataset into the uniform format."""
        dataset_name = dataset_path.name
        family = self.identify_dataset_family(dataset_name)
        lang_code = self.get_language_code(dataset_name)

        if not family or not lang_code:
            logger.warning(f"Could not identify family or language for {dataset_name}")
            return

        output_lang_dir = self.output_dir / f"{lang_code}-ZA"
        logger.info(f"Processing {dataset_name} for language {lang_code}")

        try:
            # Process files based on their extensions
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower().lstrip('.')
                    for category, extensions in DATASET_STRUCTURE.items():
                        if ext in extensions:
                            dest_dir = output_lang_dir / category
                            dest_file = dest_dir / f"{dataset_name}_{file_path.name}"
                            shutil.copy2(file_path, dest_file)
                            logger.info(f"Copied {file_path.name} to {dest_file}")
                            break

        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")

    def process_all_datasets(self):
        """Process all datasets in the PROCESSED directory."""
        if not self.processed_dir.exists():
            logger.error(f"Processed directory {self.processed_dir} does not exist!")
            return

        # Initialize completion tracker
        from dataset_completion import DatasetCompletionTracker
        tracker = DatasetCompletionTracker(self.processed_dir.parent)

        for lang_dir in self.processed_dir.glob('*-ZA'):
            if lang_dir.is_dir():
                logger.info(f"Processing language directory: {lang_dir}")
                for dataset_dir in lang_dir.iterdir():
                    if dataset_dir.is_dir() and not dataset_dir.name.startswith('DONE_'):
                        try:
                            # Process the dataset
                            self.organize_dataset(dataset_dir)
                            
                            # Mark as completed if processing was successful
                            if tracker.mark_completed(dataset_dir):
                                logger.info(f"Successfully marked as completed: {dataset_dir}")
                            else:
                                logger.warning(f"Failed to mark as completed: {dataset_dir}")
                        except Exception as e:
                            logger.error(f"Error processing dataset {dataset_dir}: {str(e)}")
                            continue

def main():
    """Main execution function."""
    try:
        organizer = DatasetOrganizer()
        organizer.process_all_datasets()
        logger.info("Dataset organization completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")

if __name__ == "__main__":
    main()
