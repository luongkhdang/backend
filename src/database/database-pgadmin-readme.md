# Database and pgAdmin Setup Guide

This guide explains how to connect to and manage the PostgreSQL database using pgAdmin in the Reader-Ultimate project.

## Overview

The Reader-Ultimate backend uses:

- **PostgreSQL with pgVector** extension for storing articles and vector embeddings
- **pgAdmin** as a web-based management tool

Both are running as Docker containers in the same network.

## Access pgAdmin

1. Start the containers using Docker Compose:

   ```bash
   docker-compose up -d
   ```

2. Open pgAdmin in your web browser:
   - URL: http://localhost:5051
   - Default credentials:
     - Email: admin@reader.com
     - Password: admin

## Network Setup

### Understanding Container Networking

The `connect_networks.ps1` script is used to connect existing containers to the `reader_network` so they can communicate with each other. This is particularly important when you have multiple Docker Compose projects that need to interact.

### Do I need to run connect_networks.ps1 every time?

- **First-time setup**: Yes, run it once after starting containers
- **After container restarts**: Yes, if containers lose network connectivity
- **After Docker service restarts**: Yes, network connections may be lost

### Making Network Connections Persistent

The main reason you need to run the script repeatedly is that Docker doesn't persist connections between different Docker Compose projects by default. For more persistent connections:

1. **Create a shared network first** (before starting any containers):

   ```powershell
   docker network create --driver bridge reader_network
   ```

2. **Use this pre-created network** in all your docker-compose files with the `external: true` flag:

   ```yaml
   networks:
     reader_network:
       external: true
   ```

3. This way, all projects will use the same persistent network without needing reconnection.

### Automating Network Connections

To avoid manually running the script each time, you can:

1. **Create a startup script** that combines Docker Compose and network connection:

   Create a file named `start-services.ps1`:

   ```powershell
   # Start the Docker Compose services
   docker-compose up -d

   # Wait for containers to initialize
   Start-Sleep -Seconds 5

   # Run the network connection script
   .\connect_networks.ps1

   Write-Host "All services started and connected!"
   ```

2. **Add Docker Compose dependencies** instead of using the script:

   Modify your `docker-compose.yml` to include the external containers in the same network:

   ```yaml
   networks:
     reader_network:
       name: reader_network
       external: false # Create the network if it doesn't exist

   # In your service definitions:
   services:
     postgres:
       # ... other configuration ...
       networks:
         - reader_network
   ```

3. **Use Docker events to trigger the script** (advanced):

   Create a monitoring script that watches for container start events and automatically runs the connection script.

### Using VS Code Docker Extension

If you're using Visual Studio Code, the Docker extension provides an easy way to manage your containers:

1. Install the VS Code Docker extension
2. Create a `.vscode/tasks.json` file with custom tasks:

   ```json
   {
     "version": "2.0.0",
     "tasks": [
       {
         "label": "Start Reader Services",
         "type": "shell",
         "command": "docker-compose up -d && powershell -ExecutionPolicy Bypass -File .\\connect_networks.ps1",
         "problemMatcher": []
       },
       {
         "label": "Stop Reader Services",
         "type": "shell",
         "command": "docker-compose down",
         "problemMatcher": []
       }
     ]
   }
   ```

3. Run these tasks from the VS Code Command Palette (Ctrl+Shift+P): `Tasks: Run Task` > `Start Reader Services`

### Windows Task Scheduler (Alternative)

You can set up the Windows Task Scheduler to run your script:

1. Open Task Scheduler
2. Create a new task
3. Set the trigger to "At startup" or "At log on"
4. Action: Start a program
5. Program/script: `powershell.exe`
6. Arguments: `-ExecutionPolicy Bypass -File "C:\path\to\start-services.ps1"`

## Register the Reader Database Server

1. In pgAdmin, right-click on "Servers" in the left panel and select "Register > Server"

2. In the **General** tab:

   - Name: `Reader DB` (or any name you prefer)

3. In the **Connection** tab:

   - Host name/address: `postgres` (the container name as defined in docker-compose.yml)
   - Port: `5432` (the internal port used within the Docker network)
   - Maintenance database: `reader_db`
   - Username: `postgres`
   - Password: `postgres`

4. In the **Advanced** tab:

   - No changes needed for basic use

5. Click "Save" to connect to the database

## Database Structure

The Reader-Ultimate database includes the following tables:

| Table Name         | Description                                 |
| ------------------ | ------------------------------------------- |
| `articles`         | Stores article content with metadata        |
| `embeddings`       | Stores vector embeddings for articles       |
| `entities`         | Named entities extracted from articles      |
| `article_entities` | Junction table linking articles to entities |
| `clusters`         | Article clusters based on embeddings        |
| `essays`           | Generated summaries and analyses            |
| `essay_entities`   | Junction table linking essays to entities   |

## Common pgAdmin Operations

### Browse Table Data

1. Navigate to `Servers > Reader DB > Databases > reader_db > Schemas > public > Tables`
2. Right-click on a table (e.g., `articles`) and select "View/Edit Data > All Rows"

### Run SQL Queries

1. Click the "Query Tool" button in the toolbar or right-click on the `reader_db` database and select "Query Tool"
2. Enter your SQL query and click the "Execute/Refresh" button

### Sample Queries

#### Count articles by domain

```sql
SELECT domain, COUNT(*) as article_count
FROM articles
GROUP BY domain
ORDER BY article_count DESC;
```

#### Find most recent articles

```sql
SELECT id, title, domain, pub_date, processed_at
FROM articles
ORDER BY pub_date DESC
LIMIT 20;
```

#### Find articles with embeddings

```sql
SELECT a.id, a.title, a.domain, e.created_at
FROM articles a
JOIN embeddings e ON a.id = e.article_id
ORDER BY e.created_at DESC
LIMIT 20;
```

## Connecting from External Applications

When connecting from:

- **Another Docker container** in the same network: Use `postgres` as hostname, port `5432`
- **Docker host machine**: Use `localhost` as hostname, port `5433` (mapped in docker-compose.yml)
- **External application**: Use the Docker host IP address, port `5433`

### Connection String Examples

- From Docker container:

  ```
  postgresql://postgres:postgres@postgres:5432/reader_db
  ```

- From host machine:
  ```
  postgresql://postgres:postgres@localhost:5433/reader_db
  ```

## Troubleshooting

### Cannot connect to the database server

- Ensure the Docker containers are running: `docker-compose ps`
- Check Docker network connectivity: `docker network inspect reader_network`
- Verify no port conflicts on host: `netstat -an | grep 5433`

### Network Connectivity Issues

If containers can't communicate with each other:

1. **Check network existence**:

   ```powershell
   docker network ls | Select-String "reader_network"
   ```

2. **Verify container network connections**:

   ```powershell
   docker network inspect reader_network
   ```

   Look for your containers in the "Containers" section of the output.

3. **Manual container connection** (if missing):

   ```powershell
   docker network connect reader_network container-name
   ```

4. **Check for network conflicts**:
   If you have multiple Docker Compose projects with the same network name but different network IDs, this can cause issues.
   ```powershell
   docker network ls
   ```
5. **Docker Desktop users**: You can visualize networks in Docker Desktop by:
   - Opening Docker Desktop
   - Going to "Containers" tab
   - Selecting a container
   - Viewing the "Networks" section in container details

### Common Network Error Messages and Solutions

| Error Message                                             | Possible Solution                                                                |
| --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `network XXX not found`                                   | Run `docker network create reader_network` and restart containers                |
| `endpoint with name X already exists in network Y`        | The connection already exists. This is usually just a warning and can be ignored |
| `Container not running`                                   | The target container must be started before connecting to network                |
| `Error response from daemon: Could not attach to network` | Docker daemon issue - try restarting Docker Desktop                              |

### pgAdmin seems slow or unresponsive

- Refresh the browser
- Check Docker container logs: `docker-compose logs pgadmin`
- Restart the pgAdmin container: `docker-compose restart pgadmin`

### Database errors in pgAdmin

- Check the server logs: `docker-compose logs postgres`
- Verify database volume permissions
- Ensure pgVector extension is installed properly

## Backup and Restore

### Create a backup

```sql
-- In pgAdmin Query Tool
COPY (SELECT * FROM articles) TO '/tmp/articles_backup.csv' WITH CSV HEADER;
```

### Advanced backup using pg_dump

```bash
# From Docker host
docker-compose exec postgres pg_dump -U postgres -d reader_db -f /tmp/reader_db_backup.sql
docker cp <container_id>:/tmp/reader_db_backup.sql ./backup/
```

## Additional Resources

- [pgAdmin Documentation](https://www.pgadmin.org/docs/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgVector GitHub Repository](https://github.com/pgvector/pgvector)
