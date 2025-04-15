#!/bin/bash

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Function to handle errors
handle_error() {
    echo "======================================================"
    echo "🔴 Error encountered during build process!"
    echo "======================================================"
    echo "Please check the Docker build logs for details."
    echo ""
    exit 1
}

# Set trap to catch errors
trap 'handle_error' ERR

# Check if base image exists
if [[ "$(docker images -q article-transfer-base:latest 2> /dev/null)" == "" ]]; then
    echo "Base image not found. Building base image first..."
    echo "======================================================"
    echo "⚠️  This will take several minutes for initial setup"
    echo "======================================================"
    docker build -t article-transfer-base:latest -f base.Dockerfile . || handle_error
    echo "✅ Base image built successfully!"
elif [ "$1" == "--rebuild-base" ] || [ "$1" == "-b" ]; then
    echo "Rebuilding base image..."
    echo "======================================================"
    echo "⚠️  This will take several minutes"
    echo "======================================================"
    docker build -t article-transfer-base:latest -f base.Dockerfile . || handle_error
    echo "✅ Base image rebuilt successfully!"
else
    echo "Using existing base image. Use '--rebuild-base' flag to rebuild it."
fi

# Build application image
echo "Building application image..."
docker-compose build || handle_error

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "To start the application:"
echo "docker-compose up" 