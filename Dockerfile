FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV READER_DB_HOST=postgres
ENV READER_DB_PORT=5432
ENV READER_DB_NAME=reader_db
ENV READER_DB_USER=postgres
ENV READER_DB_PASSWORD=postgres
ENV WAIT_FOR_DB=true
ENV WAIT_SECONDS=10

# Command to run the application
CMD ["python", "src/database/db_setup.py"] 