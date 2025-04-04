"""
Simple test client for the Articles API.

Run this script to test the API connection and retrieve some data.
"""
import requests
import json
import sys

# API configuration - change this to match your setup if needed
API_BASE_URL = "http://localhost:8000"


def test_api_connection():
    """Test the API connection by calling the root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}")
        response.raise_for_status()
        print(f"✅ API Connection Successful")
        print(f"   Status: {response.json()['status']}")
        print(f"   Message: {response.json()['message']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ API Connection Failed: {e}")
        return False


def get_status_counts():
    """Get counts of articles by status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/counts")
        response.raise_for_status()
        data = response.json()
        print("\n📊 Article Status Counts:")
        for status, count in data.get("counts", {}).items():
            print(f"   {status}: {count}")
        print(f"   Total: {data.get('total', 0)}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting status counts: {e}")
        return {}


def get_articles(status="ReadyForReview", page=1, limit=5):
    """Get articles with specified status"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/articles",
            params={"status": status, "page": page, "limit": limit}
        )
        response.raise_for_status()
        data = response.json()

        print(
            f"\n📄 Retrieved {data['count']} articles (page {data['page']} of {data['total'] // data['limit'] + 1 if data['total'] else 0})")
        print(f"   Status: {status}")

        for i, article in enumerate(data.get("articles", []), 1):
            print(f"\n   Article {i} of {len(data.get('articles', []))}")
            print(f"   ID: {article['id']}")
            print(f"   Title: {article['title']}")
            print(f"   URL: {article['url']}")
            print(f"   Domain: {article['domain']}")

        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting articles: {e}")
        return {}


def get_article_by_id(article_id):
    """Get a specific article by ID"""
    try:
        response = requests.get(f"{API_BASE_URL}/articles/{article_id}")
        response.raise_for_status()
        article = response.json()

        print(f"\n📝 Article Details (ID: {article['id']})")
        print(f"   Title: {article['title']}")
        print(f"   URL: {article['url']}")
        print(f"   Domain: {article['domain']}")
        print(f"   Status: {article['proceeding_status']}")
        print(f"   Published: {article['pub_date']}")
        print(f"   Content Preview: {article.get('content', '')[:100]}...")

        return article
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"❌ Article with ID {article_id} not found")
        else:
            print(f"❌ HTTP error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting article: {e}")
        return None


if __name__ == "__main__":
    print("=== Articles API Test Client ===")

    # Test API connection first
    if not test_api_connection():
        print("\nCannot connect to API. Please make sure the API service is running.")
        print(f"API URL: {API_BASE_URL}")
        sys.exit(1)

    # Get status counts
    status_counts = get_status_counts()

    # Get some articles
    articles_data = get_articles(status="ReadyForReview", page=1, limit=5)

    # Get details for the first article if we have any
    if articles_data and articles_data.get("articles") and len(articles_data["articles"]) > 0:
        first_article_id = articles_data["articles"][0]["id"]
        get_article_by_id(first_article_id)

    print("\n✅ API Test Completed Successfully!")
    print(f"The API is accessible at {API_BASE_URL}")
    print("For more detailed documentation, visit:")
    print(f"- {API_BASE_URL}/docs (Swagger UI)")
    print(f"- {API_BASE_URL}/redoc (ReDoc UI)")
