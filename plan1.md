# Plan to Enable and Verify GPU Utilization (plan1.md)

**Goal:** Ensure the `article-transfer` service can access and utilize the host's NVIDIA GPU (GTX 1060) for NLP tasks in Step 1.7, resolving the observed 0% utilization.

**Phase 1: GPU Verification Utility (`src/utils/gpu_client.py`)**

1.  **Create File:** Create the file `src/utils/gpu_client.py`.
2.  **Implement Check Function:** Add a function `check_gpu_availability()`:
    - Import `torch` and `logging`.
    - Check `torch.cuda.is_available()`.
    - If `True`:
      - Log INFO messages detailing the number of GPUs found (`torch.cuda.device_count()`) and the name of the default GPU (`torch.cuda.get_device_name(0)`).
      - Log INFO confirming "PyTorch CUDA setup appears correct."
      - Return `True`.
    - If `False`:
      - Log a WARNING message: "PyTorch cannot find CUDA. GPU acceleration will NOT be available. Check Docker setup, NVIDIA drivers, CUDA toolkit in the base image, and PyTorch CUDA build."
      - Return `False`.
3.  **Add Main Block:** Include an `if __name__ == "__main__":` block that calls `check_gpu_availability()` so the script can be run directly for testing.

**Phase 2: Docker Configuration**

1.  **Base Image Requirements (Prerequisite - User Verification Needed):**

    - **Verify/Update Base Image:** The Docker base image used for `article-transfer` (e.g., `article-transfer-base:latest`) **must** have the following installed:
      - NVIDIA Container Toolkit support (often handled by the base OS image if using official NVIDIA images or setting up manually).
      - A PyTorch version compiled with CUDA support compatible with the host's NVIDIA driver version and the GTX 1060's architecture. (e.g., check PyTorch installation command - it should specify a CUDA version like `cu118` or `cu121`).

2.  **`docker-compose.yml` Modifications:**
    - **Target Service:** Locate the `article-transfer` service definition.
    - **Enable GPU Access:** Add the `deploy` section to grant the container access to the GPU. This is the modern approach.
      ```yaml
      services:
        article-transfer:
          # ... existing configuration ...
          deploy:
            resources:
              reservations:
                devices:
                  - driver: nvidia
                    count: 1 # Request 1 GPU
                    capabilities: [gpu, compute, utility]
          # Potentially add environment variables (less common with 'deploy' but can help)
          # environment:
          #   # ... existing environment variables ...
          #   NVIDIA_VISIBLE_DEVICES: all
          #   NVIDIA_DRIVER_CAPABILITIES: compute,utility
          # ... rest of service definition ...
      ```
    - _(Self-correction: The `deploy` key is preferred over older methods like `runtime: nvidia`)_.

**Phase 3: Integration & Verification**

1.  **`src/main.py` Integration:**
    - Import the `check_gpu_availability` function from `src/utils/gpu_client`.
    - Call `check_gpu_availability()` near the beginning of the `main()` function (e.g., after parsing arguments or logging startup).
    - Log the boolean result to clearly indicate if the GPU check passed during container startup.
2.  **Build and Run:**
    - Rebuild the `article-transfer` image: `docker-compose build article-transfer`
    - Run the service: `docker-compose up article-transfer` (or the full stack).
3.  **Check Logs:** Observe the startup logs for the output of the `check_gpu_availability` function. This will confirm if PyTorch can see the GPU _inside the container_.
4.  **Monitor Utilization:** If the logs indicate GPU availability, monitor GPU utilization using host tools (like `nvidia-smi`) _while Step 1.7 is actively summarizing long articles_. Utilization should now be > 0% during those specific tasks.

**Notes:**

- This plan focuses on enabling Docker access and verifying PyTorch's view. The Python code in `LocalNLPClient` already attempts GPU usage.
- The most common failure points are the Docker base image lacking CUDA-enabled PyTorch or the `docker-compose.yml` not correctly requesting GPU resources.
- The `gpu_client.py` script provides a simple, isolated way to test the core PyTorch CUDA detection within the container environment.
