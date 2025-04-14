"""
news_api_client.py - Client for interacting with the News API service

This module provides a client class for interacting with the News API service to fetch articles with their
content and metadata for further processing in the data refinery pipeline.

Exports:
- NewsAPIClient: Client class for fetching articles via News API with status filtering
  - test_connection(): Dict[str, Any] - Tests connection to News API, returns connection details
  - get_articles_ready_for_review(): List[Dict[str, Any]] - Returns list of articles with 'ReadyForReview' status
  - fetch_article_ids(): List[int] - Returns list of article IDs with 'ReadyForReview' status
  - close(): None - Closes the client connection (no-op for API client)

Related files:
- src/main.py: Uses this client to fetch articles for processing
- src/database/reader_db_client.py: Stores processed articles
- src/steps/step1.py: Uses this client in Step 1.1 (Data Collection)
"""
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

    This client uses the REST API to access the news-api service on the shared Docker network.

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
                     "NEWS_API_BASE_URL", "http://news-api:8000"),
                 max_retries=5,
                 retry_delay=10):
        """Initialize the News API client.

        Args:
            api_base_url: Base URL for the News API, defaults to Docker service name
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.api_base_url = api_base_url.rstrip(
            '/')  # Remove trailing slash if present
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Common API endpoints
        self.api_endpoints = {
            "status": "/status",
            "articles": "/articles"
        }

        logger.info(f"Initializing connection to News API at {api_base_url}")

        # Test the connection at initialization
        test_result = self.test_connection()
        if "error" in test_result:
            pass
        else:
            pass

    def _make_request(self, endpoint, params=None):
        """Helper method to make requests with retry support for temporary failures

        Args:
            endpoint: API endpoint (should start with /)
            params: Query parameters for the request

        Returns:
            Response object from successful request
        """
        # Ensure endpoint starts with a slash
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint

        url = f"{self.api_base_url}{endpoint}"
        retries = 0
        last_error = None

        # Try with retries only for connection/timeout errors
        while retries < self.max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                # Raise HTTPError for bad responses (4xx or 5xx)
                response.raise_for_status()
                return response
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                # Only retry on potentially transient network errors
                retries += 1
                last_error = e

                if retries >= self.max_retries:
                    break

                # Add jitter to retry delay
                wait_time = self.retry_delay * (1 + random.random() * 0.5)
                time.sleep(wait_time)
            except requests.exceptions.HTTPError as e:
                # Do not retry on HTTP errors (like 404 Not Found), raise immediately
                raise e

        # If we exit the loop due to max retries
        if last_error:
            raise last_error
        # Should not be reached if HTTPError is raised correctly
        raise Exception(
            f"Failed to make request to {url} after {self.max_retries} attempts")

    def test_connection(self):
        """Test the API connection and return connection details."""
        api_status = {}
        status_data = {"counts": {}, "total": 0}
        root_path_checked = False

        try:
            # 1. Try the primary status endpoint
            try:
                response = self._make_request(self.api_endpoints["status"])
                api_status = response.json()
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 404:
                    logger.info(
                        f"Status endpoint {self.api_endpoints['status']} not found (404). Will check root path.")
                    # Fallback handled below if status is still empty
                else:
                    # Re-raise other HTTP errors to be caught by the outer handler
                    raise http_err
            except Exception as e:
                # Log other specific errors from _make_request if needed, but often it handles retries
                logger.warning(
                    f"Could not connect to {self.api_endpoints['status']}: {e}. Will check root path.")

            # 2. If status endpoint failed (e.g., 404 or other connection issue), try root path
            if not api_status:  # Check if api_status dict is still empty
                try:
                    response = self._make_request("/")
                    # We don't expect JSON from root, just success (2xx)
                    api_status = {"status": "ok",
                                  "message": "Connected to API root"}
                    logger.info("Successfully connected to API root path.")
                    root_path_checked = True
                except Exception as root_e:
                    # If both status and root fail, log error and return
                    logger.error(
                        f"API connection test failed on both /status and root path: {root_e}")
                    return {
                        "error": f"Connection failed: {root_e}",
                        "api_base_url": self.api_base_url
                    }

            # 3. Try to get article counts (optional)
            try:
                count_response = self._make_request("/status/counts")
                status_data = count_response.json()
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 404:
                    logger.info(
                        "Optional endpoint /status/counts not found (404). Proceeding without counts.")
                    # Keep default status_data
                else:
                    logger.warning(
                        f"HTTP error retrieving article counts from /status/counts: {http_err}")
                    # Keep default status_data, but log warning for non-404 HTTP errors
            except Exception as count_e:
                logger.warning(
                    f"Could not retrieve article counts from /status/counts: {count_e}")
                # Keep default status_data

            # 4. Construct final result
            return {
                "version": "API",
                "database": "news-db",
                "connection_string": self.api_base_url,
                "api_status": api_status.get("status", "unknown"),
                "api_message": api_status.get("message", ""),
                "article_counts": status_data.get("counts", {}),
                "total_articles": status_data.get("total", 0)
            }

        except Exception as e:
            # Catch unexpected errors during the process
            logger.error(
                f"Unexpected error during API connection test: {e}", exc_info=True)
            return {
                "error": str(e),
                "api_base_url": self.api_base_url
            }

    def get_articles_ready_for_review(self) -> List[Dict[str, Any]]:
        """Get all articles with proceeding_status = 'ReadyForReview'"""
        articles = []
        retries = 0
        page = 1
        limit = 100  # Fetch articles in batches of 100

        while retries < self.max_retries:
            try:
                response = self._make_request(
                    self.api_endpoints["articles"],
                    params={"status": "ReadyForReview",
                            "page": page, "limit": limit}
                )
                data = response.json()

                # Append articles from this page
                articles.extend(data.get("articles", []))

                # Check if we need to fetch more pages
                total_articles = data.get("total", 0)
                retrieved_articles = len(articles)

                # If we've fetched all articles, break out of the loop
                if retrieved_articles >= total_articles:
                    break

                # Otherwise, get the next page
                page += 1

            except requests.exceptions.RequestException as e:
                retries += 1
                wait_time = self.retry_delay * (1 + random.random())

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

            return article_ids

        except Exception as e:
            logger.error(f"Error fetching article IDs: {e}")
            return []

    def close(self):
        """Close the client - no-op for API client."""
        pass
