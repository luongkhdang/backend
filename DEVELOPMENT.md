# Docker Development Guide

This project uses a two-stage Docker build process to speed up development. This approach separates heavy dependencies from application code, significantly reducing rebuild times during development.

## Why Two-Stage Builds?

The project depends on several large libraries and models:

- PyTorch (~2GB)
- Transformers (~500MB)
- spaCy + en_core_web_lg model (~1GB total)
- Clustering libraries (~300MB combined)

Re-downloading and installing these dependencies for every code change would make development painfully slow.

## Build Process

We use two Docker images:

1. **Base Image (`article-transfer-base`)**

   - Contains all heavy dependencies (PyTorch, Transformers, spaCy, etc.)
   - Built from `base.Dockerfile`
   - Rarely changes (only when dependencies change)

2. **Application Image (`article-transfer-app`)**
   - Contains only application code
   - Built from regular `Dockerfile`
   - Builds quickly for code changes

## Development Workflow

### First-Time Setup

The first time you set up the project:

```bash
# Linux/Mac
chmod +x build.sh  # Make the script executable
./build.sh         # This will build both the base and application images

# Windows
.\build.ps1
```

The base image will be built automatically as it doesn't exist yet. This initial build will take some time (10-20 minutes) as it downloads and installs all the heavy dependencies.

### Regular Development

For normal development (code changes only):

```bash
# Linux/Mac
docker-compose up --build

# Windows
docker-compose up --build
```

Or use the build script without the rebuild flag:

```bash
# Linux/Mac
./build.sh
docker-compose up

# Windows
.\build.ps1
docker-compose up
```

This rebuilds only the application image, which is much faster (usually seconds).

### Adding or Updating Dependencies

When you modify dependencies (e.g., in `requirements.txt`):

1. Update `requirements.txt` or `base.Dockerfile` as needed
2. Rebuild the base image:

   ```bash
   # Linux/Mac
   ./build.sh --rebuild-base

   # Windows
   .\build.ps1 --rebuild-base
   ```

## Troubleshooting

### Base Image Not Found

If you see an error like this:

```
failed to solve: article-transfer-base:latest: failed to resolve source metadata for docker.io/library/article-transfer-base:latest
```

It means the base image hasn't been built yet. Run the build script without any arguments to build it:

```bash
# Linux/Mac
./build.sh

# Windows
.\build.ps1
```

### Module "**version**" Attribute Errors

Some Python packages don't expose a `__version__` attribute. If you see an error like:

```
AttributeError: module 'hdbscan' has no attribute '__version__'
```

You'll need to modify the verification step in `base.Dockerfile` to avoid accessing the non-existent attribute.

### Docker Build Resource Issues

If your build fails due to resource constraints:

1. **Check Docker Desktop Resources**: Go to Docker Desktop → Settings → Resources

   - Increase Memory (at least 4GB recommended)
   - Increase CPU (at least 2 cores recommended)
   - Increase Swap (at least 1GB recommended)

2. **Free Up Disk Space**: Try cleaning unused Docker resources:

   ```bash
   # Remove unused images, containers, networks
   docker system prune

   # Also remove unused volumes (careful!)
   docker system prune --volumes
   ```

### BuildKit Cache Mounting Issues

On some systems, the BuildKit cache mounting might fail. If you see errors related to cache mounting, modify `base.Dockerfile` to remove the `--mount=type=cache` instruction and use regular `pip install` commands instead.

### Multiple Dependency Attempts

The base image intentionally installs dependencies in a specific order to avoid conflicts. If you're adding new dependencies, consider their placement carefully to avoid breaking existing ones.

## Cache Volumes

The Docker Compose configuration includes a volume for Hugging Face model caching:

```yaml
volumes: huggingface_cache:/root/.cache/huggingface
```

This ensures that downloaded models persist between container restarts.

## Mount Points

The src directory is mounted as a volume to allow live code changes:

```yaml
volumes:
  - ./src:/app/src
```

This means you can edit code in your IDE and see changes without rebuilding.

## Windows PowerShell Execution Policy

Windows PowerShell scripts may require execution policy changes:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
