# Article Transfer System

This system transfers articles from the news-db database to the reader-ultimate database.

## Overview

The article transfer system:

1. Queries the `news-db` database for articles with `proceeding_status = 'ReadyForReview'`
2. Transfers these articles to the `reader-ultimate` database's `articles` table
3. Updates the status in the `news-db` to `'Transferred'` once successfully moved

## Components

- `src/main.py` - Main entry point that handles the article transfer process
- `src/database/news_db_client.py` - Client for interacting with the news-db database
- `src/database/reader_db_client.py` - Client for interacting with the reader-ultimate database

## Docker Setup

The system is designed to work with Docker containers:

- Both databases should be on the same Docker network (`reader_network`)
- The article transfer service connects to both databases using their container names

## Running the System

### Using Docker Compose

The easiest way to run the system is with Docker Compose:

```bash
docker-compose up -d
```

This will start:

- The reader-ultimate database
- The pgAdmin interface
- The article transfer service

### Running Standalone

If you want to run the transfer script directly:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python src/main.py
   ```

### Script Options

The script supports several command-line options:

- `--news-host`: News DB hostname (default: 'news-db')
- `--news-port`: News DB port (default: 5432)
- `--news-dbname`: News DB database name (default: 'postgres')
- `--news-user`: News DB username (default: 'postgres')
- `--news-password`: News DB password (default: 'postgres')
- `--reader-host`: Reader DB hostname (default: 'reader-ultimate')
- `--reader-port`: Reader DB port (default: 5432)
- `--reader-dbname`: Reader DB database name (default: 'reader_db')
- `--reader-user`: Reader DB username (default: 'READER-postgres')
- `--reader-password`: Reader DB password (default: 'READER-postgres')
- `--continuous`: Run in continuous mode with polling
- `--interval`: Polling interval in seconds (default: 300)

Example with custom settings:

```bash
python src/main.py --news-host localhost --news-port 5432 --reader-host localhost --reader-port 5433 --continuous --interval 60
```

## Logging

Logs are written to:

- Console output
- `article_transfer.log` file

## Troubleshooting

If you encounter connection issues:

1. Ensure both databases are running:

   ```bash
   docker ps | grep reader-ultimate
   docker ps | grep news-db
   ```

2. Verify network connectivity:

   ```bash
   docker network inspect reader_network
   ```

3. Make sure both containers are on the same network:
   ```bash
   # Connect containers to the network if needed
   ./connect_networks.sh
   ```

# Reader Ultimate - Backend Database Setup

This repository contains the database setup for the Reader Ultimate application. It creates and initializes PostgreSQL with pgvector extension and sets up 7 database tables according to the schema in `src/database/data-info.md`.

## Prerequisites

- Docker
- Docker Compose

## Tables Created

1. **articles** - Processed articles from the scraper
2. **entities** - Power players (people, organizations) extracted from articles
3. **article_entities** - Junction table linking articles to entities
4. **embeddings** - Gemini-generated vectors for articles
5. **clusters** - Groups of articles by topic
6. **essays** - News Feed paragraphs and Rabbit Hole layers
7. **essay_entities** - Junction table linking essays to entities

## Running with Docker Compose

```bash
# Build and start the containers
docker-compose up -d

# Check logs
docker-compose logs -f
```

## Running Locally (Development)

To run the application locally for development:

1. Set up PostgreSQL with pgvector:

   ```bash
   docker run -d \
     --name reader-db \
     -e POSTGRES_DB=reader_db \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=postgres \
     -p 5432:5432 \
     ankane/pgvector:latest
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the database:
   ```bash
   python src/database/db_setup.py
   ```

## Environment Variables

The application uses the following environment variables:

- `READER_DB_HOST`: PostgreSQL host (default: "localhost")
- `READER_DB_PORT`: PostgreSQL port (default: "5432")
- `READER_DB_NAME`: Database name (default: "reader_db")
- `READER_DB_USER`: Database user (default: "postgres")
- `READER_DB_PASSWORD`: Database password (default: "postgres")
- `WAIT_FOR_DB`: Whether to wait for the database to be ready (default: "false")
- `WAIT_SECONDS`: How many seconds to wait for the database (default: "5")

# Docker Networking Setup for Reader and Scraper Projects

This guide explains how to set up proper Docker networking between the Reader and Scraper projects,
ensuring reliable communication between the `article-transfer` service and the `news-api` service.

## Overview

The system consists of two separate Docker Compose projects:

1. **Scraper-Ultimate**: Contains the `news-api` service that provides article data
2. **Reader-Ultimate/backend**: Contains the `article-transfer` service that consumes article data from `news-api`

To enable seamless communication between these services, we use a shared Docker network (`reader_network`)
managed automatically by Docker Compose.

## One-Time Setup

**None required!** Docker Compose handles network creation automatically.

## Starting the Services

**Important:** You MUST start the services in the following order to ensure the shared network is created correctly:

1. **First, start the Reader-Ultimate/backend project:**
   This project is configured to create the `reader_network` if it doesn't exist.

   ```bash
   # Navigate to the backend directory
   cd /path/to/Reader-Ultimate/backend
   # Start the backend services
   docker-compose up -d
   ```

2. **Second, start the Scraper-Ultimate project:**
   This project connects its `news-api` service to the existing `reader_network`.
   ```bash
   # Navigate to the Scraper-Ultimate directory
   cd /path/to/Scraper-Ultimate
   # Start the scraper and API services
   docker-compose up -d
   ```

## How It Works

- The `backend/docker-compose.yml` defines and manages the `reader_network`.
- The `Scraper-Ultimate/docker-compose.yml` declares `reader_network` as `external: true` and attaches `news-api` to it.
- Docker's built-in DNS resolves service names (`news-api`, `postgres`) across this shared network.
- The `article-transfer` service connects to `news-api` using the hostname `news-api`.
- No manual network creation or connection steps are required - Docker Compose handles it automatically.
- The `network_checker.py` script (run with `--skip-network-check`) verifies service reachability.

## Troubleshooting

If you encounter connectivity issues:

1. Verify the startup order was followed (backend first, then scraper).
2. Verify both projects are running:
   ```bash
   docker ps
   ```
3. Check that the `reader_network` exists and the correct containers are attached:
   ```bash
   docker network inspect reader_network
   ```
   (You should see containers from both projects, including `news-api` and `article-transfer`).
4. Restart the services, ensuring the correct order:
   ```bash
   # In Scraper-Ultimate directory
   docker-compose down
   # In Reader-Ultimate/backend directory
   docker-compose down
   # Start backend first, then scraper
   cd /path/to/Reader-Ultimate/backend && docker-compose up -d
   cd /path/to/Scraper-Ultimate && docker-compose up -d
   ```
5. Check logs for connectivity errors:
   ```bash
   docker logs article-transfer
   docker logs news-api
   ```

## Changes Made

This setup replaces previous approaches which used:

- Manual setup scripts (`setup_network.sh`, `connect_networks.ps1`)
- The `network-connector` service
- Complex Docker introspection and fallback logic

The new approach follows Docker best practices for automated multi-container networking and provides:

- More reliable connectivity
- Improved security (no Docker socket mounting)
- Cleaner, more maintainable code
- Fully automated network management by Docker Compose
