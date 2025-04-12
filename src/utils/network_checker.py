#!/usr/bin/env python3
"""
network_checker.py - Utility for verifying network connectivity

This script verifies network connectivity to essential services and provides diagnostics
in case of connection issues. It can be run as part of container startup.

Exported functions:
- check_network_connectivity(): Checks connectivity to all required services
- ping_host(host, port): Attempts to connect to a specific host and port

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
import subprocess
import requests
from typing import Dict, List, Tuple, Any

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


def get_container_ips(network_name: str = "reader_network") -> Dict[str, str]:
    """
    Try to get container IPs if running inside Docker.

    Args:
        network_name: Name of the Docker network to check

    Returns:
        Dict[str, str]: Dictionary of container names and their IPs
    """
    ips = {}
    try:
        # This will only work if we're running in Docker with Docker CLI available
        result = subprocess.run(
            ["docker", "network", "inspect", network_name],
            capture_output=True, text=True, check=True
        )

        # Simple parsing - this could be improved with proper JSON parsing
        for line in result.stdout.splitlines():
            if "Name" in line and "IPv4Address" in result.stdout:
                # Very basic parsing - would need more robust implementation
                parts = line.split(":")
                if len(parts) > 1:
                    name = parts[1].strip().strip('",')
                    ip_line = next((l for l in result.stdout.splitlines()
                                   if "IPv4Address" in l), "")
                    if ip_line:
                        ip_parts = ip_line.split(":")
                        if len(ip_parts) > 1:
                            ip = ip_parts[1].strip().strip('",').split("/")[0]
                            ips[name] = ip

        logger.debug(f"Found container IPs: {ips}")
    except Exception as e:
        logger.debug(f"Error getting container IPs: {e}")

    return ips


def check_network_connectivity() -> Dict[str, Any]:
    """
    Check connectivity to all required services.

    Returns:
        Dict[str, Any]: Results of connectivity checks
    """
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

    # Add fallback hosts for news-api
    news_api_fallbacks = [
        "host.docker.internal",
        "172.17.0.1",
        "localhost"
    ]

    # Check if fallback URLs are specified in environment
    fallback_env = os.getenv("NEWS_API_FALLBACK_URLS")
    if fallback_env:
        for url in fallback_env.split(","):
            url = url.strip()
            if "://" in url:
                parts = url.split("://")[1].split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 8000
                if host not in [s.get("host") for s in services] + news_api_fallbacks:
                    news_api_fallbacks.append(host)

    # Try to get container IPs for better diagnostics
    container_ips = get_container_ips()

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
            # If failed, try additional diagnostics
            service_result["status"] = "disconnected"
            service_result["diagnostics"].append(
                f"Failed to connect to {service['host']}:{service['port']}"
            )

            # For news-api, try fallbacks
            if service["name"] == "news-api":
                for fallback in news_api_fallbacks:
                    if ping_host(fallback, service["port"]):
                        service_result["status"] = "connected-fallback"
                        service_result["diagnostics"].append(
                            f"Connected to fallback {fallback}:{service['port']}"
                        )
                        service_result["fallback_host"] = fallback
                        break

            # Check IP from Docker network
            if service["name"] in container_ips:
                ip = container_ips[service["name"]]
                if ping_host(ip, service["port"]):
                    service_result["status"] = "connected-ip"
                    service_result["diagnostics"].append(
                        f"Connected using IP {ip}:{service['port']}"
                    )
                    service_result["ip"] = ip

        # Add to results
        results["services"][service["name"]] = service_result

        # Update overall success status
        if service_result["status"] in ["disconnected", "unknown"]:
            if service["name"] == "news-api":
                # For news-api, only fail if all fallbacks fail
                if "fallback_host" not in service_result and "ip" not in service_result:
                    results["success"] = False
            else:
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

        # Check for Docker networks
        try:
            networks = subprocess.run(
                ["docker", "network", "ls"],
                capture_output=True, text=True, check=True
            )
            if "reader_network" in networks.stdout:
                results["diagnostics"].append(
                    "reader_network exists in Docker networks"
                )
            else:
                results["diagnostics"].append(
                    "reader_network NOT FOUND in Docker networks"
                )
        except Exception:
            results["diagnostics"].append(
                "Unable to check Docker networks. Docker CLI may not be available."
            )

        # Suggest fixes
        results["diagnostics"].append(
            "Suggestions:"
        )
        results["diagnostics"].append(
            "1. Ensure all required containers are running: docker ps"
        )
        results["diagnostics"].append(
            "2. Verify network configuration: docker network inspect reader_network"
        )
        results["diagnostics"].append(
            "3. Manually connect containers: docker network connect reader_network <container_name>"
        )
    else:
        results["diagnostics"].append(
            "All network connectivity checks passed."
        )

    return results


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
        elif status == "connected-fallback":
            print(
                f"⚠️ {service_name}: Connected via fallback {service_result.get('fallback_host', 'unknown')}")
        elif status == "connected-ip":
            print(
                f"⚠️ {service_name}: Connected via IP {service_result.get('ip', 'unknown')}")
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

    results = check_network_connectivity()
    print_results(results)

    elapsed = time.time() - start_time
    logger.info(f"Network check completed in {elapsed:.2f} seconds")

    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
