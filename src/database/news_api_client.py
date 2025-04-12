import requests
import logging
import os
import time
import random
import socket
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsAPIClient:
    """
    Client for interactions with the news-db API.

    This client uses the REST API instead of direct database access.
    It handles Docker networking connectivity issues automatically.

    Exported functions:
    - test_connection(): Tests connection to News API, returns connection details
    - get_articles_ready_for_review(): Returns list of articles with 'ReadyForReview' status
    - fetch_article_ids(): Returns list of article IDs with 'ReadyForReview' status 
    - close(): Closes the client connection (no-op for API)

    Related files:
    - src/main.py: Uses this client to fetch articles for processing
    - src/database/reader_db_client.py: Stores processed articles
    """

    def __init__(self,
                 api_base_url=os.getenv(
                     "NEWS_API_BASE_URL", "http://host.docker.internal:8000"),
                 max_retries=5,
                 retry_delay=10):
        """Initialize the News API client.

        Args:
            api_base_url: Base URL for the News API
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.api_base_url = api_base_url.rstrip(
            '/')  # Remove trailing slash if present
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.status_path = "/status"  # Default status path
        self.api_path_found = False   # Flag to track if we found a working API path

        # Define fallback URLs for different network scenarios
        # First check for environment variable with comma-separated URLs
        fallback_env = os.getenv("NEWS_API_FALLBACK_URLS")
        if fallback_env:
            # Split by comma and clean up each URL
            custom_fallbacks = [url.strip() for url in fallback_env.split(",")]
            logger.info(
                f"Using fallback URLs from environment: {custom_fallbacks}")
            self.fallback_urls = custom_fallbacks
        else:
            # Default fallbacks if not provided in environment
            self.fallback_urls = [
                # Container name (try first in Docker environment)
                "http://news-api:8000",
                "http://172.23.0.5:8000",            # Docker network IP
                "http://host.docker.internal:8000",  # Windows/Mac Docker
                "http://localhost:8000",             # Direct local
                "http://172.17.0.1:8000",            # Default Docker bridge
            ]

        # Add the base URL to the fallbacks if it's not already there
        if self.api_base_url not in self.fallback_urls:
            self.fallback_urls.append(self.api_base_url)

        logger.info(f"Fallback URLs (in order): {self.fallback_urls}")

        # Different possible API path structures to try
        self.api_path_variants = [
            "/status",                 # Standard path
            "/api/status",             # Common REST pattern
            "/v1/status",              # Versioned API
            "/health",                 # Health check endpoint
            "/",                       # Root path
            "/api",                    # Alternative API root
        ]

        logger.info(f"Initializing connection to News API at {api_base_url}")

        # Test connection on init and try fallbacks if needed
        self._test_and_set_working_url()

    def _test_and_set_working_url(self):
        """Test connections to potential URLs and set working one as primary"""
        # First try the configured URL with different API paths
        for path in self.api_path_variants:
            test_url = f"{self.api_base_url}{path}"
            logger.debug(f"Testing primary URL with path: {test_url}")
            try:
                response = requests.get(test_url, timeout=5)
                if response.status_code < 400:  # Accept any non-error status code
                    logger.info(f"Successfully connected to: {test_url}")
                    self.status_path = path
                    self.api_path_found = True
                    return
            except requests.exceptions.RequestException as e:
                logger.debug(f"Path {path} on primary URL failed: {e}")

        # If we get here, the primary URL failed with all path variants
        logger.warning(
            f"Primary URL {self.api_base_url} failed with all API path variants")

        # Try fallback URLs with all path variants
        for url in self.fallback_urls:
            if url == self.api_base_url:
                continue  # Skip if same as primary

            for path in self.api_path_variants:
                test_url = f"{url}{path}"
                logger.debug(f"Testing fallback URL: {test_url}")
                try:
                    response = requests.get(test_url, timeout=5)
                    if response.status_code < 400:  # Accept any non-error status code
                        logger.info(f"Found working API at: {test_url}")
                        self.api_base_url = url
                        self.status_path = path
                        self.api_path_found = True
                        return
                except requests.exceptions.RequestException:
                    pass

        # If we reach here, we couldn't find any working URL + path combination
        logger.warning(
            "No working URL and API path combination found. Will try base endpoints on API calls.")

    def _make_request(self, endpoint, params=None):
        """Helper method to make requests with automatic fallback for network errors

        Args:
            endpoint: API endpoint (should start with /)
            params: Query parameters for the request

        Returns:
            Response object from successful request
        """
        # Ensure endpoint starts with a slash
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint

        # For status endpoint, use the discovered path if we found one
        if endpoint == '/status' and self.api_path_found:
            endpoint = self.status_path

        # Try the requested URL first
        url = f"{self.api_base_url}{endpoint}"
        try:
            logger.debug(f"Making request to: {url} with params: {params}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:

            logger.warning(f"Request to {url} failed: {e}")

            # If connection error with primary URL, try fallbacks
            for fallback_url in self.fallback_urls:
                if fallback_url == self.api_base_url:
                    continue

                # Use the fallback URL with the endpoint
                fallback_full_url = f"{fallback_url}{endpoint}"
                logger.info(f"Trying fallback URL: {fallback_full_url}")

                try:
                    response = requests.get(
                        fallback_full_url, params=params, timeout=10)
                    response.raise_for_status()
                    # If successful, update primary URL for future requests
                    self.api_base_url = fallback_url
                    logger.info(f"Switched to working URL: {fallback_url}")
                    return response
                except requests.exceptions.RequestException as nested_e:
                    logger.debug(
                        f"Fallback URL {fallback_full_url} failed: {nested_e}")

            # Re-raise the original exception if all fallbacks fail
            raise

    def test_connection(self):
        """Test the API connection and return connection details."""
        try:
            # Test the API connection - use the status path we discovered
            status_endpoint = self.status_path if self.api_path_found else "/status"
            try:
                response = self._make_request(status_endpoint)
                api_status = response.json()
            except Exception as e:
                logger.warning(
                    f"Status endpoint failed: {e}, trying root path")
                # If status endpoint fails, try the root path
                response = self._make_request("/")
                api_status = {"status": "unknown",
                              "message": "Connected to API root"}

            # Try to get database info through various possible status endpoints
            status_count_endpoints = [
                "/status/counts",
                "/api/status/counts",
                "/v1/status/counts",
                "/counts",
                "/stats"
            ]

            status_data = {"counts": {}, "total": 0}
            for endpoint in status_count_endpoints:
                try:
                    count_response = self._make_request(endpoint)
                    status_data = count_response.json()
                    break  # Stop if we found a working endpoint
                except Exception:
                    continue  # Try the next endpoint

            return {
                "version": "API",
                "database": "news-db",
                "user": "api-user",
                "tables": ["articles"],
                "connection_string": self.api_base_url,
                "api_status": api_status.get("status", "unknown"),
                "api_message": api_status.get("message", ""),
                "api_path_found": self.api_path_found,
                "status_path_used": self.status_path,
                "article_counts": status_data.get("counts", {}),
                "total_articles": status_data.get("total", 0)
            }

        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return {
                "error": str(e),
                "api_base_url": self.api_base_url,
                "status_path_tried": self.status_path,
                "fallback_urls_tried": self.fallback_urls
            }

    def get_articles_ready_for_review(self) -> List[Dict[str, Any]]:
        """Get all articles with proceeding_status = 'ReadyForReview'"""
        articles = []
        retries = 0
        page = 1
        limit = 100  # Fetch articles in batches of 100

        # Different possible endpoints to try for articles
        article_endpoints = [
            "/articles",
            "/api/articles",
            "/v1/articles",
            "/news/articles"
        ]

        # Find a working articles endpoint first
        working_endpoint = None
        for endpoint in article_endpoints:
            try:
                logger.info(f"Testing articles endpoint: {endpoint}")
                self._make_request(endpoint, params={"limit": 1})
                working_endpoint = endpoint
                logger.info(f"Found working articles endpoint: {endpoint}")
                break
            except Exception as e:
                logger.debug(f"Articles endpoint {endpoint} failed: {e}")

        if not working_endpoint:
            logger.error("Could not find a working articles endpoint")
            return []

        while retries < self.max_retries:
            try:
                logger.info(
                    f"Fetching articles with status 'ReadyForReview' (page {page}, limit {limit})")

                response = self._make_request(
                    working_endpoint,
                    params={"status": "ReadyForReview",
                            "page": page, "limit": limit}
                )
                data = response.json()

                # Append articles from this page
                articles.extend(data.get("articles", []))

                # Check if we need to fetch more pages
                total_articles = data.get("total", 0)
                retrieved_articles = len(articles)

                logger.info(
                    f"Retrieved {retrieved_articles} of {total_articles} total articles")

                # If we've fetched all articles, break out of the loop
                if retrieved_articles >= total_articles:
                    break

                # Otherwise, get the next page
                page += 1

            except requests.exceptions.RequestException as e:
                retries += 1
                wait_time = self.retry_delay * (1 + random.random())
                logger.warning(
                    f"API request failed (attempt {retries}/{self.max_retries}): {e}")

                if retries >= self.max_retries:
                    logger.error("Maximum retry attempts reached")
                    break

                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error retrieving articles: {e}")
                break

        logger.info(
            f"Retrieved {len(articles)} articles with status 'ReadyForReview'")
        return articles

    def fetch_article_ids(self) -> List[int]:
        """
        Fetch only article IDs from news-db where proceeding_status = 'ReadyForReview'

        Returns:
            List[int]: List of article IDs
        """
        try:
            # Get articles with 'ReadyForReview' status
            articles = self.get_articles_ready_for_review()

            # Extract only the IDs
            article_ids = [article['id'] for article in articles]

            logger.info(
                f"Retrieved {len(article_ids)} article IDs with 'ReadyForReview' status")
            return article_ids

        except Exception as e:
            logger.error(f"Error fetching article IDs: {e}")
            return []

    def close(self):
        """Close the client - no-op for API client."""
        logger.info("Closed News API client (no-op)")
