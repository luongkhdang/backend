"""
localnlp_client.py - Client for Local NLP Model Inference

This module provides a client class to interact with local NLP models,
primarily for text summarization using the Hugging Face transformers library.
It supports chunk-based summarization for very long articles.

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
    """Client to handle local NLP model operations like summarization of text chunks."""

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

            # Load the summarization pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device  # Specify the device
            )

        except Exception as e:
            logger.error(
                f"Failed to load model or tokenizer '{self.model_name}': {e}", exc_info=True)
            # Depending on requirements, you might want to raise the exception
            # or allow the application to continue without summarization capability.
            # raise

    def summarize_text(self, text: str, max_summary_tokens: int, min_summary_tokens: int) -> Optional[str]:
        """
        Summarize the input text chunk using the loaded model.

        This method is designed to work with text chunks of approximately 1000 tokens,
        which is suitable for the BART model's capacity. No explicit truncation is 
        needed for properly sized chunks, as the pipeline handles truncation internally.

        For very long articles, use this method on each chunk separately, then concatenate
        the summaries to create a comprehensive summary.

        Args:
            text: The text chunk to summarize.
            max_summary_tokens: The maximum number of tokens for the chunk's summary.
            min_summary_tokens: The minimum number of tokens for the chunk's summary.

        Returns:
            The summarized text as a string, or None if summarization failed or the client is not initialized.
        """
        if not self.pipeline or not self.tokenizer:
            logger.error(
                "Summarization pipeline or tokenizer not initialized. Cannot summarize.")
            return None

        original_char_len = len(text)

        # Manually truncate the input text based on the model's max input tokens
        # Use the lock for tokenizer safety if called from multiple threads
        with self.lock:
            try:
                # Tokenize the input text, truncating to the model's max input length
                inputs = self.tokenizer(
                    text,
                    max_length=self.MODEL_MAX_INPUT_TOKENS,
                    truncation=True,
                    return_tensors="pt"  # Return PyTorch tensors
                )
                truncated_text = self.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=True
                )
                truncated_token_count = len(inputs["input_ids"][0])
                if truncated_token_count >= self.MODEL_MAX_INPUT_TOKENS:
                    logger.warning(
                        f"Input text chunk was truncated from {original_char_len} chars "
                        f"to {truncated_token_count} tokens (max: {self.MODEL_MAX_INPUT_TOKENS}) "
                        f"before summarization.")
            except Exception as e:
                logger.error(
                    f"Error during input tokenization/truncation: {e}", exc_info=True)
                return None  # Cannot proceed if truncation fails

        # Adjust max_summary_tokens based on the *truncated* input length
        # Use the truncated token count directly
        input_token_count = truncated_token_count
        adjusted_max_tokens = min(
            # Keep the heuristic, but base it on actual tokens
            max_summary_tokens, input_token_count // 2)

        # Ensure we don't go below min_summary_tokens
        if adjusted_max_tokens < min_summary_tokens:
            adjusted_max_tokens = min_summary_tokens

        if adjusted_max_tokens != max_summary_tokens:
            pass

        try:
            # Use the lock to ensure thread safety for the pipeline call
            with self.lock:
                # Generate summary using the pipeline with the truncated text
                # No need for truncation=True here as we already truncated manually
                summary_output = self.pipeline(
                    truncated_text,  # Use the truncated text
                    max_length=adjusted_max_tokens,
                    min_length=min_summary_tokens,
                    do_sample=False
                    # truncation=True # Removed, handled manually above
                )

                # Extract the summary text
                summary = summary_output[0]['summary_text']
                summary_char_len = len(summary)
                # Estimate summary token length
                summary_token_len = len(self.tokenizer.encode(summary))

            return summary

        except Exception as e:
            logger.error(
                f"Error during text chunk summarization: {e}", exc_info=True)
            return None
