"""
localnlp_client.py - Client for Local NLP Model Inference

This module provides a client class to interact with local NLP models,
primarily for text summarization using the Hugging Face transformers library.

Exported Classes:
- LocalNLPClient:
    - __init__(self, model_name: str = "facebook/bart-large-cnn")
    - summarize_text(self, text: str, max_summary_tokens: int, min_summary_tokens: int) -> Optional[str]

Related Files:
- src/steps/step1.py: Uses this client for summarizing long articles.
- requirements.txt: Contains necessary dependencies (transformers, torch, etc.)
"""

import logging
import os
from typing import Optional
from transformers import pipeline, AutoTokenizer, set_seed
import torch
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducibility if needed (optional)
# set_seed(42)


class LocalNLPClient:
    """Client to handle local NLP model operations like summarization."""

    # Define model limits - BART models typically have a 1024 token input limit
    MODEL_MAX_INPUT_TOKENS = 1024  # Set to match model's true limit

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the LocalNLPClient by loading the specified model and tokenizer.

        Args:
            model_name: The name of the Hugging Face model to load (default: facebook/bart-large-cnn).
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        # Add a lock for thread safety
        self.lock = threading.Lock()

        # Determine device (CPU or GPU if available)
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU
        device_name = "GPU" if self.device == 0 else "CPU"
        logger.info(
            f"Initializing LocalNLPClient with model '{self.model_name}' on {device_name}")

        try:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(
                f"Successfully loaded tokenizer for '{self.model_name}'")

            # Load the summarization pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device  # Specify the device
            )
            logger.info(
                f"Successfully loaded summarization pipeline for '{self.model_name}'")

        except Exception as e:
            logger.error(
                f"Failed to load model or tokenizer '{self.model_name}': {e}", exc_info=True)
            # Depending on requirements, you might want to raise the exception
            # or allow the application to continue without summarization capability.
            # raise

    def summarize_text(self, text: str, max_summary_tokens: int, min_summary_tokens: int) -> Optional[str]:
        """
        Summarize the input text using the loaded model.
        Truncates the input text to the model's maximum input length (1024 tokens).
        Uses a lock to ensure thread safety when accessing the tokenizer and pipeline.

        Args:
            text: The text content to summarize.
            max_summary_tokens: The maximum number of tokens for the generated summary.
            min_summary_tokens: The minimum number of tokens for the generated summary.

        Returns:
            The summarized text as a string, or None if summarization failed or the client is not initialized.
        """
        if not self.pipeline or not self.tokenizer:
            logger.error(
                "Summarization pipeline or tokenizer not initialized. Cannot summarize.")
            return None

        original_char_len = len(text)
        logger.debug(
            f"Attempting to summarize text with original length: {original_char_len} chars.")

        # Adjust max_summary_tokens based on input length to avoid the warning
        input_token_estimate = len(text) // 4  # Rough estimate of token count
        adjusted_max_tokens = min(max_summary_tokens, input_token_estimate)

        # Ensure we don't go below min_summary_tokens
        if adjusted_max_tokens < min_summary_tokens:
            adjusted_max_tokens = min_summary_tokens

        if adjusted_max_tokens != max_summary_tokens:
            logger.debug(
                f"Adjusted max_tokens from {max_summary_tokens} to {adjusted_max_tokens} based on input length")

        try:
            # Use the lock to ensure thread safety
            with self.lock:
                # --- Truncation ---
                # Encode, truncate, and decode to get text truncated to model's max input length
                inputs = self.tokenizer(
                    text, max_length=self.MODEL_MAX_INPUT_TOKENS, truncation=True, return_tensors="pt")
                truncated_text = self.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=True)
                truncated_char_len = len(truncated_text)
                truncated_token_len = len(inputs["input_ids"][0])

                if truncated_char_len < original_char_len:
                    logger.debug(
                        f"Input text truncated from {original_char_len} chars to {truncated_char_len} chars ({truncated_token_len} tokens) for model limit.")
                else:
                    logger.debug(
                        f"Input text length ({truncated_char_len} chars, {truncated_token_len} tokens) within model limit.")

                # --- Summarization ---
                # Generate summary using the pipeline with the adjusted token limit
                summary_output = self.pipeline(
                    truncated_text,
                    max_length=adjusted_max_tokens,
                    min_length=min_summary_tokens,
                    do_sample=False,
                    truncation=True  # Ensure truncation is enabled
                )

                # Extract the summary text
                summary = summary_output[0]['summary_text']
                summary_char_len = len(summary)
                # Estimate summary token length (optional, less precise than encoding)
                summary_token_len = len(self.tokenizer.encode(summary))

            # Log outside the lock
            logger.debug(
                f"Successfully summarized text. Summary length: {summary_char_len} chars (~{summary_token_len} tokens). Target range: {min_summary_tokens}-{adjusted_max_tokens} tokens.")
            return summary

        except Exception as e:
            logger.error(
                f"Error during text summarization: {e}", exc_info=True)
            return None
