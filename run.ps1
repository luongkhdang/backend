# This script runs the article transfer process

# Create necessary directories
New-Item -ItemType Directory -Force -Path src/database | Out-Null

# Ensure src/database is a package
New-Item -ItemType File -Force -Path src/database/__init__.py | Out-Null
New-Item -ItemType File -Force -Path src/__init__.py | Out-Null

# Install dependencies
pip install -r requirements.txt

# Run the transfer script in continuous mode
python src/main.py --continuous --interval 60 