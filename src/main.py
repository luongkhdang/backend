#!/usr/bin/env python3
"""
main.py - Orchestrator for Data Refinery Pipeline

This script orchestrates the entire data refinery pipeline by executing each step in sequence:
- Step 1: Data Collection, Processing and Storage
  - Step 1.1: Data Collection from news-db
  - Step 1.2: Article Content Processing
  - Step 1.3: Article Content Validation
  - Step 1.4: Incremental Database Storage
  - Step 1.5: Error Reporting
  - Step 1.6: Embedding Generation
- Step 2: Embedding Generation (future)
- Step 3: Clustering (future)
- Step 4: Summary Generation (future)

Each step is implemented in its own module in the src/steps/ directory.

Usage:
  python src/main.py [--workers N]
  
Options:
  --workers N    Number of parallel workers to use for processing (default: auto)
"""
import logging
import sys
import os
import time
import argparse
from dotenv import load_dotenv

# Import step modules
from src.steps.step1 import run as run_step1

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Refinery Pipeline")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers to use (default: auto)")
    return parser.parse_args()


def main():
    """
    Main function orchestrating all steps of the Data Refinery Pipeline
    """
    # Parse command line arguments
    args = parse_arguments()

    logger.info("========= STARTING DATA REFINERY PIPELINE =========")

    # Execute Step 1: Data Collection, Processing and Storage
    step1_result = run_step1(max_workers=args.workers)

    if step1_result == 0:
        logger.warning(
            "Step 1 did not process any articles or encountered an error.")
        logger.info(
            "Pipeline will continue with subsequent steps if implemented.")
    else:
        logger.info(f"Step 1 successfully processed {step1_result} articles.")

    # Future steps would be executed here
    # step2_result = run_step2()
    # step3_result = run_step3()
    # step4_result = run_step4()

    logger.info("========= DATA REFINERY PIPELINE COMPLETE =========")
    # Make sure to explicitly exit with status code 0 to avoid container restart
    sys.exit(0)


if __name__ == "__main__":
    main()
