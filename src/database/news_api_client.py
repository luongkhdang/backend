import requests
import logging
import os
import time
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewsAPIClient:
    """Client for interactions with the news-db API.

    This client uses the REST API instead of direct database access.
    """

    def __init__(self,
                 api_base_url=os.getenv(
                     "NEWS_API_BASE_URL", "http://172.23.0.5:8000"),
                 max_retries=5,
                 retry_delay=10):
        """Initialize the News API client.

        Args:
            api_base_url: Base URL for the News API
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.api_base_url = api_base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Add fallback IP address
        self.fallback_url = "http://172.23.0.5:8000"

        logger.info(f"Initializing connection to News API at {api_base_url}")

    def _make_request(self, url, params=None):
        """Helper method to make requests with fallback to IP address"""
        try:
            return requests.get(url, params=params)
        except requests.exceptions.ConnectionError as e:
            if "news-api" in url and "Failed to resolve 'news-api'" in str(e):
                # Try with fallback IP if hostname resolution failed
                fallback_url = url.replace("news-api", "172.23.0.5")
                logger.info(
                    f"Hostname resolution failed, trying IP address directly: {fallback_url}")
                return requests.get(fallback_url, params=params)
            else:
                # Re-raise the original exception for other errors
                raise

    def test_connection(self):
        """Test the API connection and return connection details."""
        try:
            # Test the API connection
            response = self._make_request(f"{self.api_base_url}")
            response.raise_for_status()
            api_status = response.json()

            # Get database info through the API status endpoint
            response = self._make_request(f"{self.api_base_url}/status/counts")
            response.raise_for_status()
            status_data = response.json()

            return {
                "version": "API",
                "database": "news-db",
                "user": "api-user",
                "tables": ["articles"],
                "connection_string": self.api_base_url,
                "api_status": api_status.get("status", "unknown"),
                "api_message": api_status.get("message", ""),
                "article_counts": status_data.get("counts", {}),
                "total_articles": status_data.get("total", 0)
            }

        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return {"error": str(e)}

    def get_articles_ready_for_review(self) -> List[Dict[str, Any]]:
        """Get all articles with proceeding_status = 'ReadyForReview'"""
        articles = []
        retries = 0
        page = 1
        limit = 100  # Fetch articles in batches of 100

        while retries < self.max_retries:
            try:
                logger.info(
                    f"Fetching articles with status 'ReadyForReview' (page {page}, limit {limit})")

                response = self._make_request(
                    f"{self.api_base_url}/articles",
                    params={"status": "ReadyForReview",
                            "page": page, "limit": limit}
                )
                response.raise_for_status()
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
