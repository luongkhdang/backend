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
  python src/main.py [--workers N] [--skip-network-check] [--network-retries N] [--retry-delay N]
  
Options:
  --workers N          Number of parallel workers to use for processing (default: auto)
  --skip-network-check Skip the network connectivity check
  --network-retries N  Number of retries for network connectivity check (default: 3)
  --retry-delay N      Delay in seconds between retries (default: 5)
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
    parser.add_argument("--network-retries", type=int, default=3,
                        help="Number of retries for network connectivity check (default: 3)")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay in seconds between retries (default: 5)")
    return parser.parse_args()


def check_network(skip_check=False, retry_count=3, retry_delay=5):
    """
    Step 0: Check network connectivity to required services.

    Args:
        skip_check: Whether to skip the network check
        retry_count: Number of times to retry connectivity checks
        retry_delay: Delay between retries in seconds

    Returns:
        bool: True if network check passed or was skipped, False if check failed
    """
    if skip_check:
        logger.info("Network connectivity check skipped.")
        return True

    logger.info(
        "========= STARTING STEP 0: NETWORK CONNECTIVITY CHECK =========")

    # Get initial wait time from environment or use default
    initial_wait_seconds = int(os.getenv("INITIAL_WAIT_SECONDS", "5"))
    if initial_wait_seconds > 0:
        time.sleep(initial_wait_seconds)

    # Run network connectivity check with retries
    results = check_network_connectivity(
        retry_count=retry_count, retry_delay=retry_delay)
    print_results(results)

    if not results["success"]:
        logger.error(
            "Network connectivity check failed. Some essential services are unreachable.")

        # Extract unreachable services from results
        unreachable = []
        for service_name, service_info in results["services"].items():
            if service_info["status"] == "disconnected":
                unreachable.append(service_name)

        if unreachable:
            logger.error(f"Unreachable services: {', '.join(unreachable)}")

        # Log diagnostics from the check
        for diag in results["diagnostics"]:
            logger.info(f"Diagnostic: {diag}")

        # Check if we should proceed anyway based on environment variable
        force_continue = os.getenv(
            "FORCE_CONTINUE_ON_NETWORK_ERROR", "false").lower() == "true"
        if force_continue:
            logger.warning(
                "FORCE_CONTINUE_ON_NETWORK_ERROR is set to true, continuing despite network errors...")
            return True
        else:
            logger.error(
                "To ignore network issues, use --skip-network-check or set FORCE_CONTINUE_ON_NETWORK_ERROR=true")
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
    if not check_network(
        skip_check=args.skip_network_check,
        retry_count=args.network_retries,
        retry_delay=args.retry_delay
    ):
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
