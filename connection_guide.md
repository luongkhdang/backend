# PostgreSQL Connection Guide

## Connection Information

- **Host**: localhost
- **Port**: 5433
- **Username**: READER-postgres
- **Password**: READER-postgres
- **Database**: reader_db

## Working Methods to Connect

### 1. Using Python

Python is the most reliable way to connect to the database. You can use the following code:

```python
import psycopg2

params = {
    'host': 'localhost',
    'port': '5433',
    'user': 'READER-postgres',
    'password': 'READER-postgres',
    'dbname': 'reader_db'
}

conn = psycopg2.connect(**params)
cur = conn.cursor()
# Execute queries here
cur.close()
conn.close()
```

### 2. Using Docker Exec

You can execute psql commands directly in the Docker container:

```bash
docker exec reader-ultimate psql -U "READER-postgres" -d reader_db -c "YOUR QUERY HERE"
```

### 3. GUI Tools (DataGrip, pgAdmin, etc.)

When using GUI tools like DataGrip or pgAdmin, use the following settings:

- Host: localhost
- Port: 5433
- User: READER-postgres
- Password: READER-postgres
- Database: reader_db

## Troubleshooting

If you encounter connection issues, try the following steps:

1. **Check if the container is running**:

   ```bash
   docker-compose ps
   ```

2. **Restart the container**:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Check that port 5433 is open**:

   ```bash
   netstat -ano | findstr 5433
   ```

4. **Check container logs**:

   ```bash
   docker logs reader-ultimate
   ```

5. **Test connection with Python**:
   Use the Python script provided in this repository to check if the database is accessible.

## Common Issues

- **Connection refused**: Make sure the container is running and the port is properly mapped
- **Authentication failed**: Verify username and password
- **Database not found**: Check that you're connecting to the correct database name
