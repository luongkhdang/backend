"""
localnlp_client.py - Client for Local NLP Model Inference

This module provides a client class to interact with local NLP models,
primarily for text summarization using the Hugging Face transformers library.
It supports both individual and batch processing for efficient GPU utilization.

Exported Classes:
- LocalNLPClient:
    - __init__(self, model_name: str = "facebook/bart-large-cnn")
    - summarize_text(self, text: str, max_summary_tokens: int, min_summary_tokens: int) -> Optional[str]
    - summarize_batch(self, texts: List[str], max_summary_tokens: int, min_summary_tokens: int) -> List[Optional[str]]

Related Files:
- src/steps/step1.py: Uses this client for summarizing long articles.
- requirements.txt: Contains necessary dependencies (transformers, torch, datasets, etc.)
"""

import logging
import os
from typing import Optional, List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import torch
from torch.utils.data import Dataset, DataLoader
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducibility if needed (optional)
# set_seed(42)

# Simple dataset class for text summarization


class SummaryDataset(Dataset):
    """Dataset for batch text summarization."""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Truncate and tokenize text
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        # Remove batch dimension added by tokenizer
        return {k: v.squeeze(0) for k, v in inputs.items()}


class LocalNLPClient:
    """Client to handle local NLP model operations like summarization of text chunks."""

    # Define model limits - BART models typically have a 1024 token input limit
    # Use a safety margin to avoid edge cases with position embeddings
    MODEL_MAX_INPUT_TOKENS = 1000  # Slightly less than 1024 for safety margin

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the LocalNLPClient by loading the specified model and tokenizer.

        Args:
            model_name: The name of the Hugging Face model to load (default: facebook/bart-large-cnn).
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        # Add a lock for thread safety
        self.lock = threading.Lock()

        # Determine device (CPU or GPU if available)
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU
        device_name = "GPU" if self.device == 0 else "CPU"

        # Determine optimal batch size based on GPU memory if available
        self.optimal_batch_size = 1  # Default for CPU
        if self.device != -1:
            try:
                # Get available GPU memory and set batch size accordingly
                # These are conservative estimates for BART-large model
                free_memory_mb = self._get_gpu_free_memory_mb()

                if free_memory_mb > 10000:  # >10GB
                    self.optimal_batch_size = 16
                elif free_memory_mb > 8000:  # >8GB
                    self.optimal_batch_size = 12
                elif free_memory_mb > 6000:  # >6GB
                    self.optimal_batch_size = 8
                elif free_memory_mb > 4000:  # >4GB
                    self.optimal_batch_size = 6
                elif free_memory_mb > 2000:  # >2GB
                    self.optimal_batch_size = 4
                else:
                    self.optimal_batch_size = 2  # Conservative for lower memory

                logger.info(
                    f"Auto-configured batch size: {self.optimal_batch_size} based on {free_memory_mb}MB free GPU memory")
            except Exception as e:
                logger.warning(
                    f"Error determining optimal batch size: {e}. Using default of 4.")
                self.optimal_batch_size = 4  # Conservative default if memory detection fails

        logger.info(
            f"Initializing LocalNLPClient with model '{self.model_name}' on {device_name} with batch size {self.optimal_batch_size}")

        try:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model directly for batch processing
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                # Avoid deprecated warning from huggingface_hub
                resume_download=None  # Let HF Hub use default behavior
            )

            # Move model to appropriate device
            if self.device != -1:  # GPU
                self.model = self.model.to(f"cuda:{self.device}")

            # Load the summarization pipeline for single-item processing
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device  # Specify the device
            )

            logger.info(
                f"Successfully loaded model and tokenizer on {device_name}")

        except Exception as e:
            logger.error(
                f"Failed to load model or tokenizer '{self.model_name}': {e}", exc_info=True)
            # Depending on requirements, you might want to raise the exception
            # or allow the application to continue without summarization capability.
            # raise

    def _get_gpu_free_memory_mb(self) -> int:
        """Get free memory in MB for the GPU device being used."""
        if not torch.cuda.is_available():
            return 0

        try:
            # Get device index
            device_idx = 0 if self.device == 0 else self.device

            # Get free memory in bytes and convert to MB
            free_memory_bytes = torch.cuda.get_device_properties(
                device_idx).total_memory - torch.cuda.memory_allocated(device_idx)
            free_memory_mb = free_memory_bytes / (1024 * 1024)

            logger.info(
                f"Detected {free_memory_mb:.0f}MB free GPU memory on device {device_idx}")
            return int(free_memory_mb)
        except Exception as e:
            logger.warning(f"Error getting GPU memory: {e}")
            return 4000  # Conservative default of 4GB

    def summarize_text(self, text: str, max_summary_tokens: int, min_summary_tokens: int) -> Optional[str]:
        """
        Summarize the input text chunk using the loaded model.

        This method is designed to work with text chunks of approximately 1000 tokens,
        which is suitable for the BART model's capacity. The method handles truncation
        to ensure the input doesn't exceed the model's position embedding limits.

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

        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for summarization.")
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
                truncated_token_count = len(inputs["input_ids"][0])

                # Only log if actual truncation happened
                if truncated_token_count >= self.MODEL_MAX_INPUT_TOKENS or original_char_len > 3000:
                    logger.warning(
                        f"Input text chunk was truncated from {original_char_len} chars "
                        f"to {truncated_token_count} tokens (max: {self.MODEL_MAX_INPUT_TOKENS}) "
                        f"before summarization.")

                # Decode back to text to ensure we have valid truncated text
                truncated_text = self.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=True
                )

            except Exception as e:
                logger.error(
                    f"Error during input tokenization/truncation: {e}", exc_info=True)
                return None  # Cannot proceed if truncation fails

        # Set summary length parameters
        # Use a direct approach without complex adjustments
        summary_max_length = max_summary_tokens
        summary_min_length = min_summary_tokens

        # Ensure min length is not greater than max length
        if summary_min_length > summary_max_length:
            summary_min_length = summary_max_length // 2

        try:
            # Use the lock to ensure thread safety for the pipeline call
            with self.lock:
                # Generate summary using the pipeline with the truncated text
                summary_output = self.pipeline(
                    truncated_text,  # Use the truncated text
                    max_length=summary_max_length,
                    min_length=summary_min_length,
                    do_sample=False,
                    # Explicitly set num_beams for more deterministic results
                    num_beams=4
                )

                # Extract the summary text
                summary = summary_output[0]['summary_text']

                # Log summary stats for debugging if needed
                # logger.debug(f"Summary generated: {len(summary)} chars")

            return summary

        except IndexError as e:
            # This specific error is often related to positional embedding limits
            logger.error(
                f"IndexError during summarization (likely position embedding limit): {e}")

            # Attempt a more aggressive truncation as a fallback
            try:
                with self.lock:
                    # Even more conservative token limit
                    safer_token_limit = self.MODEL_MAX_INPUT_TOKENS - 50
                    safer_inputs = self.tokenizer(
                        text,
                        max_length=safer_token_limit,
                        truncation=True,
                        return_tensors="pt"
                    )
                    safer_text = self.tokenizer.decode(
                        safer_inputs["input_ids"][0], skip_special_tokens=True
                    )
                    logger.info(
                        f"Attempting summarization with more conservative truncation: {len(safer_text)} chars")

                    safer_output = self.pipeline(
                        safer_text,
                        max_length=summary_max_length,
                        min_length=summary_min_length,
                        do_sample=False,
                        num_beams=4
                    )
                    return safer_output[0]['summary_text']
            except Exception as fallback_e:
                logger.error(
                    f"Fallback summarization also failed: {fallback_e}")
                return None

        except Exception as e:
            logger.error(
                f"Error during text chunk summarization: {e}", exc_info=True)
            return None

    def summarize_batch(self, texts: List[str], max_summary_tokens: int, min_summary_tokens: int) -> List[Optional[str]]:
        """
        Batch process multiple text chunks for summarization using GPU efficiently.

        This method uses DataLoader to process texts in batches for optimal GPU utilization.

        Args:
            texts: List of text chunks to summarize
            max_summary_tokens: Maximum number of tokens for each summary
            min_summary_tokens: Minimum number of tokens for each summary

        Returns:
            List of summarized texts (None for any that failed)
        """
        if not self.model or not self.tokenizer:
            logger.error(
                "Model or tokenizer not initialized. Cannot summarize batch.")
            return [None] * len(texts)

        if not texts:
            logger.warning("Empty batch provided for summarization.")
            return []

        # Filter out empty texts
        filtered_texts = []
        empty_indices = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                filtered_texts.append(text)
            else:
                empty_indices.append(i)

        logger.info(
            f"Processing batch of {len(filtered_texts)} texts for summarization")

        # If all texts were empty, return early
        if not filtered_texts:
            return [None] * len(texts)

        try:
            # Create dataset for batch processing
            dataset = SummaryDataset(
                filtered_texts, self.tokenizer, self.MODEL_MAX_INPUT_TOKENS)

            # Use the optimal batch size determined at initialization time
            batch_size = min(self.optimal_batch_size, len(filtered_texts))
            logger.info(
                f"Using batch size of {batch_size} for {len(filtered_texts)} texts")

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                # Use multiple workers if CPU-bound preprocessing is a bottleneck
                num_workers=0
            )

            # Set up generation parameters
            gen_kwargs = {
                "max_length": max_summary_tokens,
                "min_length": min_summary_tokens,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "num_beams": 4,
                "length_penalty": 2.0,
            }

            # Move model to GPU if available
            device = torch.device(
                f"cuda:{self.device}" if self.device != -1 else "cpu")
            model = self.model.to(device)

            # Process batch by batch
            all_summaries = []
            with torch.no_grad():  # Disable gradient calculation for inference
                for batch in dataloader:
                    # Move batch to the correct device
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Generate summaries
                    output_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs
                    )

                    # Decode summaries
                    batch_summaries = self.tokenizer.batch_decode(
                        output_ids,
                        skip_special_tokens=True
                    )

                    all_summaries.extend(batch_summaries)

            # Reinsert None for empty texts
            result = all_summaries.copy()
            for idx in empty_indices:
                result.insert(idx, None)

            logger.info(
                f"Successfully generated {len(all_summaries)} summaries in batch mode")
            return result

        except Exception as e:
            logger.error(
                f"Error during batch summarization: {e}", exc_info=True)
            return [None] * len(texts)
