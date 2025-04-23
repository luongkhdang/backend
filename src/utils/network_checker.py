#!/usr/bin/env python3
"""
network_checker.py - Utility for verifying network connectivity

This script verifies network connectivity to essential services and provides diagnostics
in case of connection issues. It can be run as part of container startup.

Exported functions:
- check_network_connectivity(): Checks connectivity to all required services
- ping_host(host, port): Attempts to connect to a specific host and port
- print_results(results): Prints connectivity check results in a readable format

Related files:
- src/main.py: Can call this utility during startup
- src/database/news_api_client.py: Used for API connectivity tests
- src/database/reader_db_client.py: Used for database connectivity tests
"""
import socket
import logging
import os
import sys
import time
import requests
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ping_host(host: str, port: int, timeout: int = 5) -> bool:
    """
    Attempt to connect to a host and port to check connectivity.

    Args:
        host: Hostname or IP address to check
        port: Port number to check
        timeout: Connection timeout in seconds

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.debug(f"Error pinging {host}:{port} - {e}")
        return False


def check_network_connectivity(retry_count: int = 1, retry_delay: int = 5) -> Dict[str, Any]:
    """
    Check connectivity to all required services with retry support.

    Args:
        retry_count: Number of times to retry connectivity checks
        retry_delay: Delay between retries in seconds

    Returns:
        Dict[str, Any]: Results of connectivity checks
    """
    attempt = 0
    last_results = None

    while attempt <= retry_count:
        if attempt > 0:
            logger.info(
                f"Retry attempt {attempt}/{retry_count} after {retry_delay}s delay...")
            time.sleep(retry_delay)

        results = {
            "success": True,
            "services": {},
            "diagnostics": []
        }

        # Define essential services to check
        services = [
            {"name": "reader-db", "host": "postgres", "port": 5432},
            {"name": "news-api", "host": "news-api", "port": 8000},
            {"name": "reader-pgadmin", "host": "pgadmin", "port": 80}
        ]

        # Check each service
        for service in services:
            service_result = {
                "status": "unknown",
                "diagnostics": []
            }

            # Try primary host
            if ping_host(service["host"], service["port"]):
                service_result["status"] = "connected"
                service_result["diagnostics"].append(
                    f"Successfully connected to {service['host']}:{service['port']}"
                )
            else:
                # If failed, mark as disconnected and add diagnostic
                service_result["status"] = "disconnected"
                service_result["diagnostics"].append(
                    f"Failed to connect to {service['host']}:{service['port']}"
                )

            # Add to results
            results["services"][service["name"]] = service_result

            # Update overall success status - failure for any service means overall failure
            if service_result["status"] in ["disconnected", "unknown"]:
                results["success"] = False

        # Add overall diagnostics
        if not results["success"]:
            results["diagnostics"].append(
                "Network connectivity issues detected. Some services are unreachable."
            )

            # Try to determine if we're in Docker
            in_docker = os.path.exists("/.dockerenv")
            results["diagnostics"].append(
                f"Running inside Docker: {in_docker}"
            )

            # Suggest fixes based on the specific issues detected
            unreachable_services = [name for name, svc in results["services"].items()
                                    if svc["status"] in ["disconnected", "unknown"]]

            if unreachable_services:
                results["diagnostics"].append(
                    f"Unreachable services: {', '.join(unreachable_services)}"
                )

                # Add more specific troubleshooting suggestions
                results["diagnostics"].append("Suggestions:")
                results["diagnostics"].append(
                    "1. Check if these services are running"
                )
                results["diagnostics"].append(
                    "2. Verify network connectivity between containers"
                )
                results["diagnostics"].append(
                    "3. Ensure the Docker containers are on the same network (reader_network)"
                )
        else:
            results["diagnostics"].append(
                "All network connectivity checks passed."
            )

        last_results = results

        # If successful, no need to retry
        if results["success"]:
            break

        attempt += 1

    # Add retry information if we retried
    if attempt > 0:
        last_results["diagnostics"].append(
            f"Performed {attempt} retry attempts for connectivity checks."
        )

    return last_results


def print_results(results: Dict[str, Any]) -> None:
    """
    Print the results of connectivity checks in a readable format.

    Args:
        results: Results dictionary from check_network_connectivity()
    """
    print("\n" + "="*50)
    print(" NETWORK CONNECTIVITY CHECK RESULTS ")
    print("="*50)

    if results["success"]:
        print("\n✅ OVERALL STATUS: All connectivity checks PASSED\n")
    else:
        print("\n❌ OVERALL STATUS: Some connectivity checks FAILED\n")

    print("-"*50)
    print(" SERVICE STATUS ")
    print("-"*50)

    for service_name, service_result in results["services"].items():
        status = service_result["status"]

        if status == "connected":
            print(f"✅ {service_name}: Connected successfully")
        else:
            print(f"❌ {service_name}: Connection FAILED")

    print("\n" + "-"*50)
    print(" DIAGNOSTICS ")
    print("-"*50)

    for diag in results["diagnostics"]:
        print(f"- {diag}")

    print("\n" + "="*50 + "\n")


def main():
    """
    Main function to run the network connectivity check.
    """
    start_time = time.time()
    logger.info("Starting network connectivity check...")

    # Parse retry arguments if provided
    retry_count = 1
    retry_delay = 5

    if len(sys.argv) > 1:
        try:
            retry_count = int(sys.argv[1])
        except ValueError:
            logger.warning(
                f"Invalid retry count argument: {sys.argv[1]}, using default: {retry_count}")

    if len(sys.argv) > 2:
        try:
            retry_delay = int(sys.argv[2])
        except ValueError:
            logger.warning(
                f"Invalid retry delay argument: {sys.argv[2]}, using default: {retry_delay}")

    # Run connectivity check with retries
    results = check_network_connectivity(
        retry_count=retry_count, retry_delay=retry_delay)
    print_results(results)

    elapsed = time.time() - start_time
    logger.info(f"Network check completed in {elapsed:.2f} seconds")

    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
