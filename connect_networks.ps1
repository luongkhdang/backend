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

Write-Host "Done connecting containers to network"
Write-Host ""
Write-Host "To verify connections, use:"
Write-Host "docker network inspect reader_network" 