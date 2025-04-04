# Reader-Ultimate Docker Setup Guide

This guide explains how to set up the Reader-Ultimate database with pgAdmin for easy management.

## Components

1. **PostgreSQL Database with pgvector** (container name: reader-db)

   - Contains 7 tables for the application
   - Uses pgvector extension for embeddings
   - Port: 5433 (external), 5432 (internal)
   - Credentials: postgres/postgres

2. **News Database** (external)

   - Source database with news articles
   - Accessed via localhost:5432 or API on localhost:8000
   - Not part of this docker-compose setup

3. **pgAdmin** (container name: reader-pgadmin)

   - Web-based administration tool for PostgreSQL
   - Port: 5051 (external), 80 (internal)
   - Login: admin@reader.com/admin

4. **Backend Service** (container name: reader-backend)

   - Initializes database tables and provides core functionality
   - Connected to the reader-db database
   - Mounts source code for easy development

5. **Article Transfer Service** (container name: article-transfer)
   - Transfers articles from news-db to reader-db
   - Connects to news-api (localhost:8000) and reader-db
   - Mounts source code for easy development

## Environment Variables

The system uses these key environment variables (from .env file):

- **Reader DB**:

  - Host: localhost (outside Docker) or postgres (inside Docker)
  - Port: 5433 (outside Docker) or 5432 (inside Docker)
  - Database: reader_db
  - User/Password: postgres/postgres

- **News API**:
  - URL: http://localhost:8000

## Accessing pgAdmin

1. Open your browser and navigate to: http://localhost:5051
2. Login credentials:
   - Email: admin@reader.com
   - Password: admin

## Connecting pgAdmin to Databases

### 1. Connecting to Reader-DB Database

1. After logging in, right-click on "Servers" in the left sidebar and select "Register > Server"
2. In the **General** tab:
   - Name: Reader-DB
3. In the **Connection** tab:
   - Host name/address: postgres (Use the container name)
   - Port: 5432 (Internal port within Docker network)
   - Maintenance database: reader_db
   - Username: postgres
   - Password: postgres
   - Save password: Yes (check the box)
4. Click "Save"

### 2. Connecting to News Database (External)

1. Right-click on "Servers" again and select "Register > Server"
2. In the **General** tab:
   - Name: News-DB
3. In the **Connection** tab:
   - Host name/address: localhost
   - Port: 5432
   - Maintenance database: postgres (or your news database name)
   - Username: postgres
   - Password: postgres (or your news-db password)
   - Save password: Yes (check the box)
4. Click "Save"

## Database Structure

### Reader-DB Database

The database includes 7 tables:

1. **articles** - Stores processed articles from the scraper
   - Fields: id, scraper_id, title, content, pub_date, domain, processed_at, is_hot, cluster_id
2. **entities** - Tracks powerful players extracted via NER

   - Fields: id, name, type, influence_score, mentions, first_seen, last_seen

3. **article_entities** - Junction table linking articles to entities

   - Fields: article_id, entity_id, mention_count

4. **embeddings** - Stores Gemini-generated embeddings for articles (vector type)

   - Fields: id, article_id, embedding, created_at

5. **clusters** - Groups articles by topic

   - Fields: id, centroid, article_count, created_at, is_hot

6. **essays** - Stores News Feed paragraphs and Rabbit Hole layers

   - Fields: id, type, article_id, content, layer_depth, created_at, tags

7. **essay_entities** - Junction table linking essays to entities
   - Fields: essay_id, entity_id

## Network Configuration

The setup uses two Docker networks:

1. **reader_network**: Internal network connecting all Reader-Ultimate containers
2. **external-network** (scraper-ultimate_app_network): Connects to the external news-api service

## Managing the Docker Environment

### Start the containers

```bash
docker-compose up -d
```

### Stop the containers

```bash
docker-compose down
```

### View container logs

```bash
# View PostgreSQL logs
docker logs reader-db

# View pgAdmin logs
docker logs reader-pgadmin

# View Backend logs
docker logs reader-backend

# View Article Transfer logs
docker logs article-transfer
```

### Restart a specific container

```bash
# Restart PostgreSQL
docker restart reader-db

# Restart pgAdmin
docker restart reader-pgadmin

# Restart Backend
docker restart reader-backend

# Restart Article Transfer
docker restart article-transfer
```

## Accessing from Code

### Inside Docker containers:

```python
# For reader-db (from within Docker network)
reader_client = ReaderDBClient(
    host="postgres",  # Container name
    port=5432,        # Internal port
    dbname="reader_db",
    user="postgres",
    password="postgres"
)

# For news-api (from within Docker network)
news_client = NewsAPIClient(api_base_url="http://localhost:8000")
```

### From local development machine:

```python
# For reader-db (from outside Docker)
reader_client = ReaderDBClient(
    host="localhost",
    port=5433,  # Mapped external port
    dbname="reader_db",
    user="postgres",
    password="postgres"
)

# For news-api
news_client = NewsAPIClient(api_base_url="http://localhost:8000")
```

## Troubleshooting

### Connection Issues

If pgAdmin cannot connect to PostgreSQL:

1. **Verify Network Communication**

   ```bash
   # Test connection to Reader-DB
   docker exec reader-pgadmin ping -c 2 postgres
   ```

2. **Check Container Status**

   ```bash
   docker ps | grep reader-db
   ```

3. **Verify Database is Running**

   ```bash
   # Check Reader-DB
   docker exec reader-db psql -U postgres -d reader_db -c "SELECT version();"
   ```

### Port Conflicts

If you see "port is already allocated" errors:

1. Check for existing containers using the port: `docker ps`
2. Modify the port mapping in docker-compose.yml
   - For example, change "5433:5432" to "5434:5432" if port 5433 is in use

### Role or Database Does Not Exist

If you see errors like "role 'postgres' does not exist" or "database 'reader_db' does not exist":

1. Remove the volume and restart the containers to reinitialize the database:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

## Data Persistence

Database data is stored in Docker volume `postgres_data` which persists between container restarts.
