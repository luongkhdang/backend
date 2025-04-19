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
- Step 2: Clustering (Article clustering based on embeddings)
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
import json  # Import json for pretty printing
import asyncio  # Add asyncio import

# Import step modules
from src.steps.step1 import run as run_step1
from src.steps.step2 import run as run_step2
from src.steps.step3 import run as run_step3  # run_step3 is now async
from src.steps.step4 import run as run_step4  # run_step4 is now async
from src.steps.domain_goodness import calculate_domain_goodness_scores

# Import network checker
from src.utils.network_checker import check_network_connectivity, print_results

# Import GPU checker
from src.utils.gpu_client import check_gpu_availability

# Configure logging
# Get log level from environment variable, default to INFO
log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(level=log_level,
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
        return True

    logger.info("STEP 0: NETWORK CHECK")

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
            "Network check failed: Some essential services are unreachable")

        # Extract unreachable services from results
        unreachable = []
        for service_name, service_info in results["services"].items():
            if service_info["status"] == "disconnected":
                unreachable.append(service_name)

        if unreachable:
            logger.error(f"Unreachable: {', '.join(unreachable)}")

        # Log diagnostics from the check - only if there are any
        if results["diagnostics"]:
            logger.debug(f"Diagnostics: {'; '.join(results['diagnostics'])}")

        # Check if we should proceed anyway based on environment variable
        force_continue = os.getenv(
            "FORCE_CONTINUE_ON_NETWORK_ERROR", "false").lower() == "true"
        if force_continue:
            logger.warning(
                "Continuing despite network errors (FORCE_CONTINUE_ON_NETWORK_ERROR=true)")
            return True
        else:
            logger.error(
                "To ignore network issues, use --skip-network-check or set FORCE_CONTINUE_ON_NETWORK_ERROR=true")
            return False

    return True


async def main():  # Make main async
    """
    Main function orchestrating all steps of the Data Refinery Pipeline
    """
    # Parse command line arguments
    args = parse_arguments()

    logger.info("STARTING DATA REFINERY PIPELINE")

    # GPU Check (Log availability)
    gpu_available = check_gpu_availability()
    logger.info(f"GPU Check: CUDA Available for PyTorch = {gpu_available}")

    # Step 0: Check network connectivity
    if not check_network(
        skip_check=args.skip_network_check,
        retry_count=args.network_retries,
        retry_delay=args.retry_delay
    ):
        logger.error("Aborting pipeline due to network issues")
        return 1  # Return non-zero exit code

    # Execute Step 1: Data Collection, Processing and Storage
    step1_result = run_step1(max_workers=args.workers)

    # Handle both possible return types from step1 (dict or int)
    if isinstance(step1_result, dict):
        # New version returns status dictionary
        logger.info("Step 1 Summary:")
        logger.debug(json.dumps(step1_result, indent=2))

        inserted_count = step1_result.get("step1.4_inserted", 0)
        if inserted_count == 0:
            logger.warning("No new articles inserted")
        else:
            logger.info(f"Inserted {inserted_count} articles")
    else:
        # Old version returns integer count of processed articles
        if step1_result == 0:
            logger.warning("Step 1 did not process any articles")
        else:
            logger.info(
                f"Step 1 successfully processed {step1_result} articles")

    # Execute Step 2: Clustering (only if enabled via environment variable)
    if os.getenv("RUN_CLUSTERING_STEP", "false").lower() == "true":
        logger.info("========= STARTING STEP 2: CLUSTERING =========")
        try:
            step2_status = run_step2()
            logger.info("Step 2 Summary:")
            logger.debug(json.dumps(step2_status, indent=2))

            if step2_status.get("success", False):
                logger.info(
                    f"Clustering successful: {step2_status.get('clusters_found', 0)} clusters created")
            else:
                logger.warning(
                    f"Clustering completed with issues: {step2_status.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Step 2 failed with error: {e}", exc_info=True)
        finally:
            logger.info("========= STEP 2 COMPLETE =========")
    else:
        logger.info(
            "Skipping Step 2: Clustering (RUN_CLUSTERING_STEP not true)")

    # Execute Step 3: Entity Extraction (only if enabled via environment variable)
    if os.getenv("RUN_STEP3", "false").lower() == "true":
        logger.info("========= STARTING STEP 3: ENTITY EXTRACTION =========")
        try:
            step3_status = await run_step3()  # Use await for the async function
            logger.info("Step 3 Summary:")
            logger.debug(json.dumps(step3_status, indent=2))

            if step3_status.get("success", False):
                logger.info(
                    f"Entity extraction successful: {step3_status.get('processed', 0)} articles processed")
                logger.info(
                    f"Created {step3_status.get('entity_links_created', 0)} entity links")
                logger.info(
                    f"Stored {step3_status.get('snippets_stored', 0)} supporting snippets")

                # Calculate influence scores for entities extracted in this run
                if os.getenv("CALCULATE_INFLUENCE_SCORES", "true").lower() == "true":
                    logger.info(
                        "Calculating influence scores for extracted entities")
                    from src.database.reader_db_client import ReaderDBClient
                    db_client = ReaderDBClient()
                    try:
                        # Note: calculate_entities_influence_scores is currently synchronous
                        # If it becomes async later, it would need await here.
                        influence_status = db_client.calculate_entities_influence_scores(
                            force_recalculate=False  # Skip recalculation if done within 24 hours
                        )
                        logger.info(
                            f"Influence score calculation: {influence_status}")
                    finally:
                        db_client.close()
            else:
                logger.warning(
                    f"Entity extraction completed with issues: {step3_status.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Step 3 failed with error: {e}", exc_info=True)
        finally:
            logger.info("========= STEP 3 COMPLETE =========")
    else:
        logger.info("Skipping Step 3: Entity Extraction (RUN_STEP3 not true)")

    # Execute Step 4: Data Export (only if enabled via environment variable)
    if os.getenv("RUN_STEP4", "false").lower() == "true":
        logger.info("========= STARTING STEP 4: DATA EXPORT =========")
        try:
            step4_status = await run_step4()  # Now properly awaits the async function
            logger.info("Step 4 Summary:")
            logger.debug(json.dumps(step4_status, indent=2))

            if step4_status.get("success", False):
                logger.info(
                    f"Data export successful: Exported {step4_status.get('articles_processed', 0)} articles to {step4_status.get('output_file')}")
            else:
                logger.warning(
                    f"Data export completed with issues: {step4_status.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Step 4 failed with error: {e}", exc_info=True)
        finally:
            logger.info("========= STEP 4 COMPLETE =========")
    else:
        logger.info("Skipping Step 4: Data Export (RUN_STEP4 not true)")

    # Weekly domain goodness calculation (can be triggered separately or scheduled)
    if os.getenv("CALCULATE_DOMAIN_GOODNESS", "false").lower() == "true":
        logger.info("========= CALCULATING DOMAIN GOODNESS SCORES =========")
        try:
            from src.database.reader_db_client import ReaderDBClient
            db_client = ReaderDBClient()
            try:
                # Note: calculate_domain_goodness_scores is currently synchronous
                domain_status = calculate_domain_goodness_scores(db_client)
                logger.info(f"Domain goodness calculation: {domain_status}")

                if domain_status.get("top_domains"):
                    logger.info("Top domains by goodness score:")
                    for domain in domain_status["top_domains"]:
                        logger.info(
                            f"  {domain['domain']}: {domain['score']:.4f}")
            finally:
                db_client.close()
        except Exception as e:
            logger.error(
                f"Domain goodness calculation failed: {e}", exc_info=True)
        finally:
            logger.info(
                "========= DOMAIN GOODNESS CALCULATION COMPLETE =========")

    # Future steps would be executed here
    # step4_result = run_step4()

    logger.info("DATA REFINERY PIPELINE COMPLETE")
    # Make sure to explicitly exit with status code 0 to avoid container restart
    return 0


if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        # Catch any top-level errors during async execution
        logger.critical(
            f"Pipeline failed with unhandled exception: {e}", exc_info=True)
        sys.exit(1)  # Exit with non-zero code on critical failure
