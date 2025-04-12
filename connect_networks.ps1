# This script connects existing containers to the reader_network
# so they can communicate with each other

Write-Host "Connecting containers to reader_network..."

# Create the network if it doesn't exist
try {
    docker network inspect reader_network | Out-Null
} catch {
    Write-Host "Creating reader_network..."
    docker network create --driver bridge reader_network
}

# Connect the news-db container if it exists and isn't already connected
if (docker ps | Select-String -Pattern "news-db") {
    Write-Host "Connecting news-db to reader_network..."
    try {
        docker network connect reader_network news-db
    } catch {
        Write-Host "news-db already connected or error connecting"
    }
} else {
    Write-Host "news-db container not found"
}

# Connect the pgadmin container if it exists and isn't already connected
if (docker ps | Select-String -Pattern "pgadmin") {
    Write-Host "Connecting pgadmin to reader_network..."
    try {
        docker network connect reader_network pgadmin
    } catch {
        Write-Host "pgadmin already connected or error connecting"
    }
} else {
    Write-Host "pgadmin container not found"
}

# Connect the reader-ultimate container if it exists and isn't already connected
if (docker ps | Select-String -Pattern "reader-ultimate") {
    Write-Host "Connecting reader-ultimate to reader_network..."
    try {
        docker network connect reader_network reader-ultimate
    } catch {
        Write-Host "reader-ultimate already connected or error connecting"
    }
} else {
    Write-Host "reader-ultimate container not found"
}

# PowerShell script to connect the news-api container to reader_network
# This helps establish communication between separate Docker Compose projects

# Check if network exists
$networkExists = docker network ls | Select-String "reader_network"
if (-not $networkExists) {
    Write-Host "Creating reader_network..." -ForegroundColor Yellow
    docker network create --driver bridge reader_network
}

# Check if news-api container is running
$containerExists = docker ps | Select-String "news-api"
if (-not $containerExists) {
    Write-Host "WARNING: news-api container is not running!" -ForegroundColor Red
    Write-Host "Please start the news-api container first, then run this script again."
    exit 1
}

# Try to connect news-api to the reader_network
try {
    Write-Host "Connecting news-api container to reader_network..." -ForegroundColor Cyan
    docker network connect reader_network news-api
    Write-Host "Successfully connected news-api to reader_network." -ForegroundColor Green
} catch {
    # If it's already connected, this is fine
    if ($_.Exception.Message -like "*endpoint with name news-api already exists*") {
        Write-Host "news-api is already connected to reader_network." -ForegroundColor Green
    } else {
        Write-Host "Error connecting news-api to reader_network: $_" -ForegroundColor Red
    }
}

# Make sure our containers are connected to the network
Write-Host "Verifying network connections..." -ForegroundColor Cyan
$networkInfo = docker network inspect reader_network -f "{{json .Containers}}" | ConvertFrom-Json

# Get container names in the network
$containerNames = $networkInfo.PSObject.Properties | ForEach-Object { 
    $_.Value.Name 
}

Write-Host "Containers in reader_network: $($containerNames -join ', ')" -ForegroundColor Yellow

# Check for essential containers
$essentialContainers = @("news-api", "article-transfer", "reader-db", "postgres", "reader-pgadmin")
foreach ($container in $essentialContainers) {
    if ($containerNames -contains $container) {
        Write-Host "✅ $container is connected to the network" -ForegroundColor Green
    } else {
        Write-Host "❌ $container is NOT connected to the network" -ForegroundColor Red
    }
}

# Provide troubleshooting info
Write-Host "`nNetwork connection complete. If you still have connection issues:"
Write-Host "1. Restart the article-transfer container: docker restart article-transfer"
Write-Host "2. Check logs: docker logs article-transfer"
Write-Host "3. Run API connectivity test: docker exec article-transfer python src/utils/api_tester.py"
Write-Host "4. Verify network settings: docker network inspect reader_network"

Write-Host "Done connecting containers to network"
Write-Host ""
Write-Host "To verify connections, use:"
Write-Host "docker network inspect reader_network" 