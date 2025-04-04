"""
Example client for the Articles API.

This script demonstrates how to consume the Articles API from another project.
"""
import requests
import json
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')


def get_articles(status: str = "ReadyForReview", page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """
    Get articles with specified status

    Args:
        status: Article status to filter by
        page: Page number for pagination
        limit: Number of articles per page

    Returns:
        Dictionary with articles data
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/articles",
            params={"status": status, "page": page, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return {"articles": [], "count": 0, "page": page, "limit": limit, "total": 0}


def get_article_by_id(article_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific article by ID

    Args:
        article_id: ID of the article to retrieve

    Returns:
        Dictionary with article data or None if not found
    """
    try:
        response = requests.get(f"{API_BASE_URL}/articles/{article_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Article with ID {article_id} not found")
        else:
            print(f"HTTP error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article: {e}")
        return None


def check_url_exists(url: str) -> bool:
    """
    Check if a URL exists in the database

    Args:
        url: URL to check

    Returns:
        True if the URL exists, False otherwise
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/check-url",
            params={"url": url}
        )
        response.raise_for_status()
        return response.json().get("exists", False)
    except requests.exceptions.RequestException as e:
        print(f"Error checking URL: {e}")
        return False


def get_article_status_counts() -> Dict[str, Any]:
    """
    Get counts of articles by status

    Returns:
        Dictionary with status counts
    """
    try:
        response = requests.get(f"{API_BASE_URL}/status/counts")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching status counts: {e}")
        return {"counts": {}, "total": 0}


# Example usage
if __name__ == "__main__":
    print("Articles API Client Example")
    print("==========================")

    # Check API connection
    try:
        response = requests.get(f"{API_BASE_URL}")
        response.raise_for_status()
        print(f"API Status: {response.json()['status']}")
        print(f"Message: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        exit(1)

    # Get article status counts
    status_counts = get_article_status_counts()
    print("\nArticle Status Counts:")
    for status, count in status_counts.get("counts", {}).items():
        print(f"{status}: {count}")
    print(f"Total: {status_counts.get('total', 0)}")

    # Get articles
    articles_data = get_articles(status="ReadyForReview", page=1, limit=5)
    print(
        f"\nRetrieved {articles_data['count']} articles (page {articles_data['page']} of {articles_data['total'] // articles_data['limit'] + 1})")

    # Display articles
    for article in articles_data.get("articles", []):
        print(
            f"ID: {article['id']}, Title: {article['title']}, Domain: {article['domain']}")

        # Example of getting article details
        if article['id']:
            article_detail = get_article_by_id(article['id'])
            if article_detail:
                content_preview = article_detail.get(
                    'content', '')[:100] + '...' if article_detail.get('content') else 'No content'
                print(f"  Content preview: {content_preview}")

            # Check if a URL exists
            url_exists = check_url_exists(article['url'])
            print(f"  URL in database: {url_exists}")

        print("")
