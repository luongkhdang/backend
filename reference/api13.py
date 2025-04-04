"""
Articles API Client: Provides a REST API for safe access to the articles database.

Exported Components:
- app: FastAPI application instance
- start_api_server(): Starts the API server
- ArticleResponse: Pydantic model for API responses
- StatusResponse: Pydantic model for status responses

Related Files:
- postgreSQL/postgreSQL_client.py: Database client used by this API
"""
from src.postgreSQL.postgreSQL_client import PostgreSQLClient
import os
import sys
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import time

# Add project root to path to ensure imports work correctly
# This handles both Docker and local development scenarios
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import PostgreSQL client

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Article Database API",
    description="API for safely accessing the article database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database client
db_client = PostgreSQLClient()

# Pydantic models for API responses


class ArticleResponse(BaseModel):
    id: int
    url: str
    domain: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    proceeding_status: str
    pub_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    scraped_at: Optional[datetime] = None


class ArticleListResponse(BaseModel):
    articles: List[ArticleResponse]
    count: int
    page: int
    limit: int
    total: Optional[int] = None


class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: float = Field(default_factory=time.time)

# API Routes


@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint providing API status"""
    return {
        "status": "ok",
        "message": "Article Database API is running",
        "timestamp": time.time()
    }


@app.get("/articles", response_model=ArticleListResponse)
async def get_articles(
    status: str = Query(
        "ReadyForReview", description="Filter by article status"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    Get articles with pagination, filtered by status
    """
    try:
        offset = (page - 1) * limit

        # Get connection from the pool
        conn = db_client.get_connection()
        cursor = conn.cursor()

        # Get articles
        cursor.execute("""
            SELECT id, url, domain, title, content, proceeding_status, pub_date, created_at, scraped_at
            FROM articles
            WHERE proceeding_status = %s
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, (status, limit, offset))

        articles = []
        for row in cursor.fetchall():
            articles.append({
                "id": row[0],
                "url": row[1],
                "domain": row[2],
                "title": row[3],
                "content": row[4],
                "proceeding_status": row[5],
                "pub_date": row[6],
                "created_at": row[7],
                "scraped_at": row[8]
            })

        # Get total count for pagination
        cursor.execute(
            "SELECT COUNT(*) FROM articles WHERE proceeding_status = %s",
            (status,)
        )
        total = cursor.fetchone()[0]

        # Clean up
        cursor.close()
        db_client.release_connection(conn)

        return {
            "articles": articles,
            "count": len(articles),
            "page": page,
            "limit": limit,
            "total": total
        }
    except Exception as e:
        logger.error(f"Error retrieving articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


@app.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: int):
    """
    Get a specific article by ID
    """
    try:
        # Get connection from the pool
        conn = db_client.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, url, domain, title, content, proceeding_status, pub_date, created_at, scraped_at
            FROM articles
            WHERE id = %s
        """, (article_id,))

        row = cursor.fetchone()

        # Clean up
        cursor.close()
        db_client.release_connection(conn)

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Article with ID {article_id} not found"
            )

        return {
            "id": row[0],
            "url": row[1],
            "domain": row[2],
            "title": row[3],
            "content": row[4],
            "proceeding_status": row[5],
            "pub_date": row[6],
            "created_at": row[7],
            "scraped_at": row[8]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


@app.get("/domains/failed", response_model=Dict[str, Any])
async def get_failed_domains():
    """
    Get domains with failed article processing
    """
    try:
        failed_domains = db_client.get_failed_domains()
        return {"domains": failed_domains, "count": len(failed_domains)}
    except Exception as e:
        logger.error(f"Error retrieving failed domains: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


@app.get("/status/counts", response_model=Dict[str, Any])
async def get_status_counts():
    """
    Get counts of articles by status
    """
    try:
        # Get connection from the pool
        conn = db_client.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT proceeding_status, COUNT(*) as count
            FROM articles
            GROUP BY proceeding_status
            ORDER BY count DESC
        """)

        status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Clean up
        cursor.close()
        db_client.release_connection(conn)

        # Get total count
        total = sum(status_counts.values())

        return {
            "counts": status_counts,
            "total": total
        }
    except Exception as e:
        logger.error(f"Error retrieving status counts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


@app.get("/check-url", response_model=Dict[str, bool])
async def check_url(url: str = Query(..., description="URL to check in database")):
    """
    Check if a URL exists in the database
    """
    try:
        exists = db_client.check_url_in_database(url)
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Error checking URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the API server
    """
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Setup database if not already set up
    try:
        db_client.setup_database()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        logger.warning(
            "Continuing with API startup despite database setup error")

    # Get host and port from environment variables or use defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    # Start API server
    logger.info(f"Starting API server on {host}:{port}")
    start_api_server(host, port)
