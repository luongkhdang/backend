FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    dnsutils \
    iputils-ping \
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Create logs directory for error reports
RUN mkdir -p /app/logs && chmod 777 /app/logs

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
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "src/database/db_setup.py"] 