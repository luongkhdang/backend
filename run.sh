#!/bin/bash

# This script runs the article transfer process

# Create necessary directories
mkdir -p src/database

# Ensure src/database is a package
touch src/database/__init__.py
touch src/__init__.py

# Install dependencies
pip install -r requirements.txt

# Run the transfer script in continuous mode
python src/main.py --continuous --interval 60 