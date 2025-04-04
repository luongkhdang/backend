#!/bin/bash

# Run the connect_networks.sh script
echo "Running network connection script..."
bash /app/connect_networks.sh

# Then run the main Python application with any arguments passed to the container
echo "Starting main application..."
exec python /app/src/main.py "$@" 