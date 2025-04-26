# Implementation Plan: Gemini Thinking and Grounding

## 1. Goal

Enable Gemini Thinking and Grounding features for supported models in the application, using the `google-genai` Python SDK. The activation of these features for a specific model call will depend on configuration flags read from environment variables (`*_THINKING`, `*_GROUNDING`).

## 2. Identify Supported Models (`GeminiClient.__init__`)

- Modify `GeminiClient.__init__` to read `*_THINKING` and `*_GROUNDING` environment variables alongside `*_MODEL_ID` and `*_RPM` for each generation model prefix (e.g., `GEMINI_FLASH_EXP_THINKING`, `GEMINI_FLASH_GROUNDING`).
- Store these boolean flags (defaulting to `False` if the environment variable is missing or not 'true') within the model configuration structure (e.g., `self.gen_model_details = { 'model_id': {'rpm': ..., 'thinking': ..., 'grounding': ...} }`).
- Log the detected thinking/grounding capabilities for each loaded model at DEBUG level.

## 3. Set API Version (`GeminiClient.__init__`)

- Check if _any_ loaded generation model has its `thinking` flag set to `True`.
- If thinking is enabled for any model, initialize the `google_genai.Client` instance with `http_options={'api_version': 'v1alpha'}` to support the thinking feature.
  ```python
  # Example within __init__
  needs_v1alpha = any(details.get('thinking', False) for details in self.gen_model_details.values())
  http_opts = {'api_version': 'v1alpha'} if needs_v1alpha else {}
  self.client = google_genai.Client(api_key=api_key, http_options=http_opts)
  ```
- Log whether the client is initialized with `v1alpha`.

## 4. Pass Capability Flags (`GeminiClient` -> `generator.py`)

- Modify `GeminiClient.analyze_articles_with_prompt` and `GeminiClient.generate_essay_from_prompt`.
- Inside these methods, retrieve the `thinking` and `grounding` boolean flags for the `effective_model_id` being used by looking them up in `self.gen_model_details`.
- Update the calls to the corresponding `generator.py` functions (`generator_analyze_articles`, `generator_generate_text`) to pass these retrieved flags as new arguments (e.g., `thinking_enabled=model_flags['thinking']`, `grounding_enabled=model_flags['grounding']`).
- Update the function signatures in `generator.py` to accept `thinking_enabled: bool = False` and `grounding_enabled: bool = False`.

## 5. Dynamic Configuration (`generator.py`)

- Inside `analyze_articles_with_prompt` and `generate_text_from_prompt` in `generator.py`:
- Use the passed `thinking_enabled` and `grounding_enabled` flags to dynamically configure the API call.
- **Thinking Config:** If `thinking_enabled` is `True`, create the thinking configuration object and add it to the main `GenerateContentConfig`.

  ```python
  # Example within generator.py functions
  gen_config_kwargs = {
      "temperature": temperature,
      "max_output_tokens": max_output_tokens,
      "automatic_function_calling": {'disable': True}
  }
  if thinking_enabled:
      try:
          # Ensure google_genai_types is imported
          gen_config_kwargs["thinking_config"] = google_genai_types.ThinkingConfig(include_thoughts=True)
          logger.debug(f"[{model_name}] Enabling Thinking mode (include_thoughts=True)")
      except AttributeError:
          logger.warning(f"[{model_name}] Could not enable Thinking: google_genai_types.ThinkingConfig not found. Check google-genai version and imports.")


  gen_config = google_genai_types.GenerateContentConfig(**gen_config_kwargs)
  ```

- **Grounding Config:** If `grounding_enabled` is `True`, create the `tools` list for grounding. Pass this list directly to the `client.aio.models.generate_content` call.

  ```python
  # Example within generator.py functions
  tools_list = None
  if grounding_enabled:
      try:
          # Ensure google_genai_types is imported
          tools_list = [google_genai_types.Tool(
              google_search_retrieval=google_genai_types.GoogleSearchRetrieval()
              # Add other grounding tools here if needed, e.g., specific data sources
          )]
          logger.debug(f"[{model_name}] Enabling Grounding via Google Search Retrieval tool.")
      except AttributeError:
           logger.warning(f"[{model_name}] Could not enable Grounding: google_genai_types.Tool or GoogleSearchRetrieval not found. Check google-genai version and imports.")


  # Pass tools_list to the generate_content call
  response = await client.aio.models.generate_content(
      model=f'models/{model_name}',
      contents=contents,
      config=gen_config,
      tools=tools_list # Pass the tools list here
  )
  ```

## 6. Response Handling (`generator.py`)

- Modify the response processing logic in `analyze_articles_with_prompt` and `generate_text_from_prompt`.
- After receiving the `response` object:
  - Check if `response.candidates` exists and has content.
  - Iterate through `response.candidates[0].content.parts`.
  - Initialize `generated_text = ""` and `thoughts_log = []`.
  - For each `part`:
    - Check if `hasattr(part, 'thought') and part.thought is True`.
    - If it's a thought, append `part.text` to `thoughts_log`.
    - Otherwise (it's regular content), append `part.text` to `generated_text`.
  - Log the combined `thoughts_log` content at DEBUG level (e.g., `logger.debug(f"[{model_name}] Thoughts: {' | '.join(thoughts_log)}")`).
  - Use the final `generated_text` for subsequent processing (e.g., JSON parsing, returning the result). Handle cases where `generated_text` might be empty even if thoughts were produced.
  - **Note:** This plan does not include storing the detailed `grounding_metadata` (e.g., search results, source links) returned by the API when grounding is enabled. Only the final generated text content is extracted for use.

## 7. Environment Variables (`docker-compose.yml`)

- Ensure the `*_THINKING` and `*_GROUNDING` environment variables are correctly defined (e.g., set to `"true"` or `"false"`) for each relevant model in `docker-compose.yml`.

## 8. Testing and Verification

- Run the pipeline with a model configured with `THINKING: true` (e.g., `gemini-2.0-flash-exp` based on current compose file). Check DEBUG logs for "[model_id] Enabling Thinking mode" and "[model_id] Thoughts: ..." messages.
- Run with a model configured with `GROUNDING: true`. Check DEBUG logs for "[model_id] Enabling Grounding..." messages. Verifying grounding _effectiveness_ requires analyzing the output quality for factuality and relevance, which is beyond simple log checking.
- Run with a model where both flags are `false` (e.g., `gemini-2.0-flash-lite`). Verify that the thinking/grounding enablement logs do _not_ appear and that the features are not used.
- Ensure that JSON parsing or other downstream processing of the `generated_text` still works correctly after separating out thoughts.
