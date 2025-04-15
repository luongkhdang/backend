"""
dimensionality.py - Dimensionality reduction module for article embeddings

This module provides functions for performing dimensionality reduction on article embeddings
using techniques like UMAP, t-SNE, and PCA for visualization purposes.

Exported functions:
- reduce_dimensions(embeddings: np.ndarray, method: str = 'umap', n_components: int = 2, 
                  config: Optional[Dict] = None) -> np.ndarray
- reduce_and_store_dimensions(article_ids: List[int], embeddings: np.ndarray, 
                            method: str = 'umap', n_components: int = 2, 
                            config: Optional[Dict] = None) -> bool
- get_method_default_config(method: str) -> Dict
- get_available_methods() -> List[str]
                            
Related files:
- src/database/modules/dimensionality.py - For storing and retrieving dimensionality data
- src/steps/step2/interpretation.py - For visualization using the reduced dimensions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import os
import json
from datetime import datetime

# Import the database module for storing coordinates
from src.database.modules.dimensionality import (
    store_coordinates,
    batch_store_coordinates,
    update_coordinates_config,
    get_coordinates_config
)
from src.database.reader_db_client import ReaderDBClient

# Configure logger
logger = logging.getLogger(__name__)

# Initialize the database client
db_client = ReaderDBClient()

# Default configurations for different dimensionality reduction methods
DEFAULT_CONFIGS = {
    "umap": {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42
    },
    "tsne": {
        "perplexity": 30.0,
        "early_exaggeration": 12.0,
        "learning_rate": "auto",
        "n_iter": 1000,
        "random_state": 42
    },
    "pca": {
        "svd_solver": "auto",
        "random_state": 42
    }
}

# Path for saving models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..", "..", "models", "dimensionality")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_available_methods() -> List[str]:
    """
    Returns a list of available dimensionality reduction methods.

    Returns:
        List[str]: List of method names
    """
    return list(DEFAULT_CONFIGS.keys())


def get_method_default_config(method: str) -> Dict:
    """
    Gets the default configuration for a specific dimensionality reduction method.

    Args:
        method (str): The dimensionality reduction method (umap, tsne, pca)

    Returns:
        Dict: Default configuration parameters

    Raises:
        ValueError: If the method is not supported
    """
    method = method.lower()
    if method not in DEFAULT_CONFIGS:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}. "
                         f"Supported methods are: {', '.join(DEFAULT_CONFIGS.keys())}")

    return DEFAULT_CONFIGS[method].copy()


def _create_model(method: str, n_components: int, config: Optional[Dict] = None) -> Any:
    """
    Creates a dimensionality reduction model based on the specified method and configuration.

    Args:
        method (str): The dimensionality reduction method (umap, tsne, pca)
        n_components (int): Number of dimensions for the output
        config (Optional[Dict]): Configuration parameters for the method

    Returns:
        Any: The initialized model

    Raises:
        ValueError: If the method is not supported
    """
    method = method.lower()
    if method not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unsupported dimensionality reduction method: {method}")

    # Use default config as base and update with provided config
    method_config = get_method_default_config(method)
    if config:
        method_config.update(config)

    # Create model based on method
    if method == "umap":
        return UMAP(n_components=n_components, **method_config)
    elif method == "tsne":
        return TSNE(n_components=n_components, **method_config)
    elif method == "pca":
        return PCA(n_components=n_components, **method_config)

    raise ValueError(f"Method implementation missing for: {method}")


def reduce_dimensions(embeddings: np.ndarray, method: str = 'umap',
                      n_components: int = 2, config: Optional[Dict] = None,
                      save_model: bool = False) -> np.ndarray:
    """
    Reduces dimensions of embeddings using the specified method.

    Args:
        embeddings (np.ndarray): Array of embeddings to reduce
        method (str): Dimensionality reduction method (umap, tsne, pca)
        n_components (int): Number of dimensions for the output
        config (Optional[Dict]): Configuration parameters for the method
        save_model (bool): Whether to save the fitted model

    Returns:
        np.ndarray: Reduced dimensionality coordinates

    Raises:
        ValueError: If embeddings are invalid or method is not supported
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings cannot be None or empty")

    method = method.lower()
    logger.info(
        f"Reducing dimensions with {method} to {n_components} components")

    # Create model
    model = _create_model(method, n_components, config)

    # Fit and transform data
    try:
        if method == "tsne":
            # t-SNE doesn't have a separate transform method
            reduced_coords = model.fit_transform(embeddings)
        else:
            reduced_coords = model.fit_transform(embeddings)

            # Save the model if requested (not applicable for t-SNE)
            if save_model:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(
                    MODELS_DIR, f"{method}_{n_components}d_{timestamp}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved {method} model to {model_path}")

                # Save configuration
                config_to_save = config or get_method_default_config(method)
                config_to_save.update({
                    "n_components": n_components,
                    "model_path": model_path,
                    "timestamp": timestamp
                })
                db_client.update_coordinates_config(config_to_save, method)

        return reduced_coords

    except Exception as e:
        logger.error(
            f"Error during dimensionality reduction with {method}: {str(e)}")
        raise


def reduce_and_store_dimensions(article_ids: List[int], embeddings: np.ndarray,
                                method: str = 'umap', n_components: int = 2,
                                config: Optional[Dict] = None,
                                save_model: bool = True) -> bool:
    """
    Reduces dimensions of embeddings and stores the results in the database.

    Args:
        article_ids (List[int]): List of article IDs corresponding to the embeddings
        embeddings (np.ndarray): Array of embeddings to reduce
        method (str): Dimensionality reduction method (umap, tsne, pca)
        n_components (int): Number of dimensions for the output
        config (Optional[Dict]): Configuration parameters for the method
        save_model (bool): Whether to save the fitted model

    Returns:
        bool: True if successful, False otherwise

    Raises:
        ValueError: If article_ids and embeddings lengths don't match
    """
    if len(article_ids) != len(embeddings):
        raise ValueError(
            "Number of article IDs must match number of embeddings")

    try:
        # Perform dimensionality reduction
        reduced_coords = reduce_dimensions(
            embeddings, method, n_components, config, save_model
        )

        # Prepare data for database storage
        coordinates_data = []
        timestamp = datetime.now().isoformat()

        for i, article_id in enumerate(article_ids):
            # Convert coordinates to a list for JSON storage
            coords = reduced_coords[i].tolist()

            # Add metadata
            metadata = {
                "timestamp": timestamp,
                "method": method,
                "n_components": n_components
            }

            if config:
                metadata["config"] = config

            coordinates_data.append({
                "article_id": article_id,
                "method": method,
                "coordinates": coords,
                "metadata": metadata
            })

        # Store coordinates in database
        db_client.batch_store_coordinates(coordinates_data)
        logger.info(
            f"Stored {len(coordinates_data)} coordinate sets in database using {method}")
        return True

    except Exception as e:
        logger.error(f"Error reducing and storing dimensions: {str(e)}")
        return False


def load_model(method: str) -> Tuple[Any, Dict]:
    """
    Loads the most recent saved model for the specified method.

    Args:
        method (str): Dimensionality reduction method (umap, tsne, pca)

    Returns:
        Tuple[Any, Dict]: Tuple containing the loaded model and its configuration

    Raises:
        ValueError: If no model is found for the method
    """
    method = method.lower()

    # Get the configuration for the method
    config = db_client.get_coordinates_config(method)
    if not config or "model_path" not in config:
        raise ValueError(f"No saved model found for method: {method}")

    model_path = config["model_path"]
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        return model, config
    except Exception as e:
        logger.error(f"Error loading model for {method}: {str(e)}")
        raise


def transform_new_embeddings(embeddings: np.ndarray, method: str = 'umap') -> np.ndarray:
    """
    Transforms new embeddings using a previously trained dimensionality reduction model.

    Args:
        embeddings (np.ndarray): Array of embeddings to transform
        method (str): Dimensionality reduction method (umap, pca)

    Returns:
        np.ndarray: Reduced dimensionality coordinates

    Raises:
        ValueError: If method is not supported for transformation or no model exists
    """
    method = method.lower()

    # t-SNE doesn't support transform on new data, only UMAP and PCA do
    if method == "tsne":
        raise ValueError(
            "t-SNE doesn't support transforming new data. Use UMAP or PCA instead.")

    try:
        # Load the model
        model, config = load_model(method)

        # Transform the embeddings
        reduced_coords = model.transform(embeddings)
        return reduced_coords

    except Exception as e:
        logger.error(
            f"Error transforming new embeddings with {method}: {str(e)}")
        raise
