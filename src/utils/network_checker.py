#!/usr/bin/env python3
"""
network_checker.py - Utility for verifying network connectivity

This script verifies network connectivity to essential services and provides diagnostics
in case of connection issues. It can be run as part of container startup.

Exported functions:
- check_network_connectivity(): Checks connectivity to all required services
- ping_host(host, port): Attempts to connect to a specific host and port
- get_container_ips(network_name): Gets container IPs in a Docker network
- print_results(results): Prints connectivity check results in a readable format

Related files:
- src/main.py: Can call this utility during startup
- src/database/news_api_client.py: Used for API connectivity tests
- src/database/reader_db_client.py: Used for database connectivity tests
- docker-compose.yml: Works in tandem with the network-connector service
"""
import socket
import logging
import os
import sys
import time
import subprocess
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
            capture_output=True, text=True, check=False
        )

        # Check for successful command execution
        if result.returncode != 0:
            logger.debug(
                f"Failed to inspect network {network_name}: {result.stderr.strip()}")
            return ips

        # Use safer JSON parsing for network inspection output
        try:
            import json
            network_data = json.loads(result.stdout)

            # Extract container IPs from network data
            if network_data and isinstance(network_data, list) and len(network_data) > 0:
                containers = network_data[0].get('Containers', {})
                for container_id, container_data in containers.items():
                    name = container_data.get('Name')
                    if name:
                        ip_address = container_data.get('IPv4Address', '')
                        if ip_address:
                            ip = ip_address.split('/')[0]  # Remove subnet mask
                            ips[name] = ip

            logger.debug(f"Found container IPs: {ips}")
        except json.JSONDecodeError:
            # Fall back to string parsing if JSON parsing fails
            for line in result.stdout.splitlines():
                if "Name" in line and "IPv4Address" in result.stdout:
                    parts = line.split(":")
                    if len(parts) > 1:
                        name = parts[1].strip().strip('",')
                        ip_line = next((l for l in result.stdout.splitlines()
                                        if "IPv4Address" in l), "")
                        if ip_line:
                            ip_parts = ip_line.split(":")
                            if len(ip_parts) > 1:
                                ip = ip_parts[1].strip().strip(
                                    '",').split("/")[0]
                                ips[name] = ip

            logger.debug(f"Found container IPs (using string parsing): {ips}")
    except Exception as e:
        logger.debug(f"Error getting container IPs: {e}")

    return ips


def check_docker_network(network_name: str = "reader_network") -> Dict[str, Any]:
    """
    Check if a Docker network exists and which containers are connected to it.

    Args:
        network_name: Name of the Docker network to check

    Returns:
        Dict[str, Any]: Results of the network check including existence and connected containers
    """
    results = {
        "exists": False,
        "connected_containers": [],
        "error": None
    }

    try:
        # Check if network exists
        network_check = subprocess.run(
            ["docker", "network", "inspect", network_name],
            capture_output=True, text=True, check=False
        )

        if network_check.returncode != 0:
            if "No such network" in network_check.stderr:
                logger.warning(f"Network {network_name} does not exist")
                results["error"] = f"Network {network_name} does not exist"
                return results
            else:
                logger.error(
                    f"Error inspecting network: {network_check.stderr.strip()}")
                results["error"] = f"Error inspecting network: {network_check.stderr.strip()}"
                return results

        # Network exists
        results["exists"] = True

        # Parse connected containers
        try:
            import json
            network_data = json.loads(network_check.stdout)

            if network_data and isinstance(network_data, list) and len(network_data) > 0:
                containers = network_data[0].get('Containers', {})
                for container_id, container_data in containers.items():
                    name = container_data.get('Name')
                    if name:
                        results["connected_containers"].append(name)

            logger.debug(
                f"Connected containers: {results['connected_containers']}")
        except json.JSONDecodeError:
            results["error"] = "Failed to parse network inspection output"
            logger.error("Failed to parse network inspection output")

    except Exception as e:
        results["error"] = f"Error checking Docker network: {str(e)}"
        logger.error(f"Error checking Docker network: {str(e)}")

    return results


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
            "diagnostics": [],
            "docker_network": None
        }

        # Check Docker network first
        network_results = check_docker_network()
        results["docker_network"] = network_results

        if not network_results["exists"]:
            results["diagnostics"].append(
                f"Docker network 'reader_network' does not exist. Network connector service may have failed."
            )
        else:
            required_containers = ["reader-db", "news-api", "reader-pgadmin"]
            missing_containers = [
                container for container in required_containers
                if not any(connected.startswith(container) for connected in network_results["connected_containers"])
            ]

            if missing_containers:
                results["diagnostics"].append(
                    f"Essential containers missing from reader_network: {', '.join(missing_containers)}"
                )

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
                container_name = service.get("name")
                if container_name in container_ips:
                    ip = container_ips[container_name]
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

            # Suggest fixes based on the specific issues detected
            results["diagnostics"].append("Suggestions:")

            if not network_results["exists"]:
                results["diagnostics"].append(
                    "1. Ensure reader_network exists: docker network create --driver bridge reader_network"
                )

            if "connected_containers" in network_results and network_results["connected_containers"]:
                missing = []
                for service in services:
                    service_name = service["name"]
                    if not any(container.startswith(service_name) for container in network_results["connected_containers"]):
                        missing.append(service_name)

                if missing:
                    results["diagnostics"].append(
                        f"2. Connect missing containers to network: {', '.join(missing)}"
                    )
                    for container in missing:
                        results["diagnostics"].append(
                            f"   docker network connect reader_network {container}"
                        )

            results["diagnostics"].append(
                "3. Ensure all required containers are running: docker ps"
            )
            results["diagnostics"].append(
                "4. Try restarting the network-connector service: docker-compose up -d --force-recreate network-connector"
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

    # Print Docker network status if available
    if "docker_network" in results and results["docker_network"]:
        print("-"*50)
        print(" DOCKER NETWORK STATUS ")
        print("-"*50)

        network_results = results["docker_network"]
        if network_results["exists"]:
            print(f"✅ Docker network 'reader_network' exists")
            if "connected_containers" in network_results and network_results["connected_containers"]:
                print(
                    f"Connected containers ({len(network_results['connected_containers'])}):")
                for container in sorted(network_results["connected_containers"]):
                    print(f"  - {container}")
            else:
                print("⚠️ No containers connected to the network")
        else:
            print(f"❌ Docker network 'reader_network' does NOT exist")

        if "error" in network_results and network_results["error"]:
            print(f"⚠️ Network check error: {network_results['error']}")

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
