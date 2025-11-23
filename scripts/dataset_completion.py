#!/usr/bin/env python3
"""
Utility script to mark completed datasets and track their processing status.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetCompletionTracker:
    """Tracks and marks completed datasets."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.completion_log = self.project_root / 'complete_datasets.txt'
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the completion log file exists."""
        if not self.completion_log.exists():
            self.completion_log.touch()
            logger.info(f"Created completion log file: {self.completion_log}")

    def is_completed(self, dataset_path: Path) -> bool:
        """Check if a dataset has been marked as completed."""
        with open(self.completion_log, 'r') as f:
            completed_paths = f.read().splitlines()
            return str(dataset_path) in completed_paths

    def mark_completed(self, dataset_path: Path) -> bool:
        """
        Mark a dataset as completed by:
        1. Logging the path in complete_datasets.txt
        2. Renaming the folder with DONE_ prefix
        """
        try:
            # Get relative path for logging
            rel_path = dataset_path.relative_to(self.project_root)
            
            # Check if already completed
            if self.is_completed(rel_path):
                logger.info(f"Dataset already marked as completed: {rel_path}")
                return True

            # Add to completion log
            with open(self.completion_log, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{rel_path}\t{timestamp}\n")

            # Rename the folder
            parent_dir = dataset_path.parent
            new_name = f"DONE_{dataset_path.name}"
            new_path = parent_dir / new_name

            # Check if DONE_ folder already exists
            if new_path.exists():
                logger.warning(f"DONE_ folder already exists for: {rel_path}")
                return False

            # Rename the folder
            dataset_path.rename(new_path)
            logger.info(f"Marked as completed: {rel_path} -> {new_name}")
            return True

        except Exception as e:
            logger.error(f"Error marking dataset as completed: {str(e)}")
            return False

    def list_completed_datasets(self) -> list:
        """Return a list of all completed datasets with their completion times."""
        completed = []
        with open(self.completion_log, 'r') as f:
            for line in f:
                if line.strip():
                    path, timestamp = line.strip().split('\t')
                    completed.append({
                        'path': path,
                        'completed_at': timestamp
                    })
        return completed

    def verify_completion_status(self) -> bool:
        """Verify that all marked datasets are properly renamed and logged."""
        success = True
        with open(self.completion_log, 'r') as f:
            for line in f:
                if line.strip():
                    path, _ = line.strip().split('\t')
                    full_path = self.project_root / path
                    parent_dir = full_path.parent
                    done_name = f"DONE_{full_path.name}"
                    done_path = parent_dir / done_name

                    if not done_path.exists():
                        logger.error(f"Inconsistency found: {done_path} does not exist")
                        success = False

        return success

def main():
    """Main execution function."""
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Create tracker instance
        tracker = DatasetCompletionTracker(project_root)
        
        # Verify completion status
        if tracker.verify_completion_status():
            logger.info("All completion statuses are consistent")
        else:
            logger.warning("Some inconsistencies found in completion status")

        # List all completed datasets
        completed = tracker.list_completed_datasets()
        logger.info(f"Total completed datasets: {len(completed)}")
        for dataset in completed:
            logger.info(f"- {dataset['path']} (completed: {dataset['completed_at']})")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
