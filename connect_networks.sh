#!/bin/bash

# This script connects existing containers to the reader_network
# so they can communicate with each other

echo "Connecting containers to reader_network..."

# Create the network if it doesn't exist
if ! docker network inspect reader_network > /dev/null 2>&1; then
    echo "Creating reader_network..."
    docker network create --driver bridge reader_network
fi

# Connect the news-db container if it exists and isn't already connected
if docker ps | grep -q "news-db"; then
    echo "Connecting news-db to reader_network..."
    docker network connect reader_network news-db || echo "news-db already connected or error connecting"
else
    echo "news-db container not found"
fi

# Connect the pgadmin container if it exists and isn't already connected
if docker ps | grep -q "pgadmin"; then
    echo "Connecting pgadmin to reader_network..."
    docker network connect reader_network pgadmin || echo "pgadmin already connected or error connecting"
else
    echo "pgadmin container not found"
fi

# Connect the reader-ultimate container if it exists and isn't already connected
if docker ps | grep -q "reader-ultimate"; then
    echo "Connecting reader-ultimate to reader_network..."
    docker network connect reader_network reader-ultimate || echo "reader-ultimate already connected or error connecting"
else
    echo "reader-ultimate container not found"
fi

echo "Done connecting containers to network"
echo ""
echo "To verify connections, use:"
echo "docker network inspect reader_network" 