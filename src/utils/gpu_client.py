"""
gpu_client.py - Utility for checking GPU availability and configuration

This module provides functions to check if a compatible NVIDIA GPU and CUDA
setup are available for PyTorch within the current environment (e.g., Docker container).

Exported functions:
- check_gpu_availability() -> bool:
  Checks if CUDA is available and logs details about the GPU setup.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_gpu_availability() -> bool:
    """
    Check if PyTorch can access a CUDA-enabled GPU.

    Logs details about the GPU setup if available, or a warning if not.

    Returns:
        bool: True if a CUDA GPU is available, False otherwise.
    """
    try:
        # Attempt to import torch dynamically to avoid hard dependency if not used
        import torch
    except ImportError:
        logger.error("PyTorch is not installed. Cannot check for GPU.")
        return False

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA Available: Yes. Found {gpu_count} GPU(s).")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Default GPU Name: {gpu_name}")
        except Exception as e:
            logger.warning(f"Could not get GPU name: {e}")
        logger.info(
            "PyTorch CUDA setup appears correct. GPU acceleration should be available.")
        return True
    else:
        logger.warning(
            "PyTorch cannot find CUDA. GPU acceleration will NOT be available.")
        logger.warning(
            "Check Docker setup (deploy/resources/devices or runtime), base image (NVIDIA drivers, CUDA toolkit), and ensure PyTorch was installed with CUDA support.")
        return False


if __name__ == "__main__":
    print("Running GPU Availability Check...")
    available = check_gpu_availability()
    if available:
        print("GPU check PASSED. PyTorch can access CUDA.")
    else:
        print("GPU check FAILED. PyTorch cannot access CUDA.")
