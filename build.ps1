# Enable BuildKit for better caching
$env:DOCKER_BUILDKIT=1

# Function to handle errors
function Handle-Error {
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Red
    Write-Host "Error encountered during build process!" -ForegroundColor Red
    Write-Host "======================================================" -ForegroundColor Red
    Write-Host "Please check the Docker build logs for details."
    Write-Host ""
    exit 1
}

# Check if base image exists
$baseImageExists = docker images -q article-transfer-base:latest 2>$null
if ($baseImageExists -eq $null -or $baseImageExists -eq "") {
    Write-Host "Base image not found. Building base image first..."
    Write-Host "======================================================" -ForegroundColor Yellow
    Write-Host "This will take several minutes for initial setup" -ForegroundColor Yellow
    Write-Host "======================================================" -ForegroundColor Yellow
    
    try {
        docker build -t article-transfer-base:latest -f base.Dockerfile .
        if ($LASTEXITCODE -ne 0) { Handle-Error }
    }
    catch {
        Handle-Error
    }
    
    Write-Host "Base image built successfully!" -ForegroundColor Green
}
elseif ($args[0] -eq "--rebuild-base" -or $args[0] -eq "-b") {
    Write-Host "Rebuilding base image..."
    Write-Host "======================================================" -ForegroundColor Yellow
    Write-Host "This will take several minutes" -ForegroundColor Yellow
    Write-Host "======================================================" -ForegroundColor Yellow
    
    try {
        docker build -t article-transfer-base:latest -f base.Dockerfile .
        if ($LASTEXITCODE -ne 0) { Handle-Error }
    }
    catch {
        Handle-Error
    }
    
    Write-Host "Base image rebuilt successfully!" -ForegroundColor Green
}
else {
    Write-Host "Using existing base image. Use '--rebuild-base' flag to rebuild it."
}

# Build application image
Write-Host "Building application image..."
try {
    docker-compose build
    if ($LASTEXITCODE -ne 0) { Handle-Error }
}
catch {
    Handle-Error
}

Write-Host ""
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:"
Write-Host "docker-compose up" 