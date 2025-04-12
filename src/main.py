#!/usr/bin/env python3
"""
main.py - Orchestrator for Data Refinery Pipeline

This script orchestrates the entire data refinery pipeline by executing each step in sequence:
- Step 0: Network Connectivity Check
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

# Import network checker
from src.utils.network_checker import check_network_connectivity, print_results

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
    parser.add_argument("--skip-network-check", action="store_true",
                        help="Skip the network connectivity check")
    return parser.parse_args()


def check_network(skip_check=False):
    """
    Step 0: Check network connectivity to required services.

    Args:
        skip_check: Whether to skip the network check

    Returns:
        bool: True if network check passed or was skipped, False if check failed
    """
    if skip_check:
        logger.info("Network connectivity check skipped.")
        return True

    logger.info(
        "========= STARTING STEP 0: NETWORK CONNECTIVITY CHECK =========")
    wait_seconds = int(os.getenv("WAIT_SECONDS", "5"))
    logger.info(f"Waiting {wait_seconds} seconds for services to be ready...")
    time.sleep(wait_seconds)

    # Run network connectivity check
    logger.info("Running network connectivity check...")
    results = check_network_connectivity()
    print_results(results)

    if not results["success"]:
        logger.error(
            "Network connectivity check failed. Please check your Docker network configuration.")
        logger.error(
            "You may need to run the connect_networks.ps1 script manually or restart the application.")

        # Check if we should proceed anyway based on environment variable
        force_continue = os.getenv(
            "FORCE_CONTINUE_ON_NETWORK_ERROR", "false").lower() == "true"
        if force_continue:
            logger.warning(
                "FORCE_CONTINUE_ON_NETWORK_ERROR is set to true, continuing despite network errors...")
            return True
        else:
            return False

    logger.info(
        "Network connectivity check passed. All required services are reachable.")
    return True


def main():
    """
    Main function orchestrating all steps of the Data Refinery Pipeline
    """
    # Parse command line arguments
    args = parse_arguments()

    logger.info("========= STARTING DATA REFINERY PIPELINE =========")

    # Step 0: Check network connectivity
    if not check_network(skip_check=args.skip_network_check):
        logger.error("Aborting pipeline due to network connectivity issues.")
        return 1

    # Execute Step 1: Data Collection, Processing and Storage
    step1_result = run_step1(max_workers=args.workers)

    if step1_result == 0:
        logger.warning("Step 1 did not process any articles.")
    else:
        logger.info(f"Step 1 successfully processed {step1_result} articles.")

    # Future steps would be executed here
    # step2_result = run_step2()
    # step3_result = run_step3()
    # step4_result = run_step4()

    logger.info("========= DATA REFINERY PIPELINE COMPLETE =========")
    # Make sure to explicitly exit with status code 0 to avoid container restart
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
