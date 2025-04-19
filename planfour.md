# Plan Four (Revised): Advanced Gemini Analysis for Step 4

**Objective:** Add dedicated advanced analysis functionality to the existing Gemini API client (`src/gemini/gemini_client.py`) for `step4.py` and future analysis needs, then modify `step4.py` to prepare article data, process it through this enhanced client using a specific prompt (`src/prompts/step4.txt`), and save the resulting analysis JSON from Gemini to the output directory.

**Phase 1: Add Analysis Method to `GeminiClient` (`src/gemini/gemini_client.py`)**

1.  **Modify `GeminiClient` Class:** Add a new asynchronous method `analyze_articles_with_prompt` to the existing `GeminiClient` class in `src/gemini/gemini_client.py`.
2.  **Method `analyze_articles_with_prompt`:**
    - **Signature:** `async def analyze_articles_with_prompt(self, articles_data: List[Dict[str, Any]], prompt_file_path: str, model_name: str, system_instruction: Optional[str] = None, temperature: float = 0.2, max_output_tokens: int = 8192, retries: int = 3, initial_delay: float = 1.0) -> Optional[Dict[str, Any]]`
    - **Inputs:**
      - `articles_data`: List of prepared article dictionaries.
      - `prompt_file_path`: Path to the prompt file (e.g., `src/prompts/step4.txt`).
      - `model_name`: Specific Gemini model (e.g., `models/gemini-2.0-flash-thinking-exp-01-21`).
      - `system_instruction`: by default: "The AI agent should adopt an academic persona—specifically, that of a seasoned political science professor at Stanford, who is also a leading expert in political and economic affairs with access to insider information by virtue of sitting on the directive board of the current ruling party. Aware of the risks of censorship and the precariousness of its position, the agent must carefully navigate these constraints, striving to present factual information in a way that encourages independent thought. Rather than drawing explicit conclusions, it subtly unveils the truth through evidence and context, allowing the audience to arrive at their own interpretations. At its core, the agent is an educator, committed to the intellectual growth of the next generation. It recognizes that failing to uphold this responsibility would be a betrayal of its duty as a noble scholar and mentor.".
      - `temperature`, `max_output_tokens`, `retries`, `initial_delay`: Control parameters.
    - **Functionality:**
      - Load prompt template text from `prompt_file_path`. Handle `FileNotFoundError`.
      - Determine the `actual_system_instruction` (use override or class default).
      - Serialize `articles_data` to a compact JSON string (`input_json_str`). Handle `TypeError`.
      - Inject `input_json_str` into the prompt template (replace `{INPUT_DATA_JSON}`).
      - Implement retry loop (`for attempt in range(retries)`):
        - Apply rate limiting using the existing `self.rate_limiter` (`await self.rate_limiter.wait_if_needed_async(model_name)`).
        - Instantiate the model: `model = genai.GenerativeModel(model_name)`.
        - Call `model.generate_content_async(...)` asynchronously:
          - `contents=[full_prompt_text]`
          - `generation_config=genai.types.GenerationConfig(...)`:
            - Set `temperature=temperature`, `max_output_tokens=max_output_tokens`.
            - **Crucially, set `response_mime_type='application/json'` to request JSON output.**
          - `safety_settings=...` (use appropriate settings if needed).
          - Include `system_instruction=actual_system_instruction` if the API supports it directly in the config or call (check `google-generativeai` documentation for the exact parameter name and location).
        - Handle potential API exceptions (rate limits, timeouts, `google.api_core.exceptions`, etc.). Log errors and apply exponential backoff (`await asyncio.sleep(delay)`).
        - On success, register the call using the existing `self.rate_limiter` (`await self.rate_limiter.register_call_async(model_name)`).
        - Get the response text: `response.text`.
        - Attempt to parse the text: `parsed_json = json.loads(response.text)`. Handle `json.JSONDecodeError`.
        - **Validate Structure:** Check if `parsed_json` is a dictionary and contains the expected top-level key (e.g., `"article_groups"` based on the prompt). If validation passes, return `parsed_json`.
        - If parsing or validation fails within an attempt, log the issue and continue to the next retry (if any).
    - **Output:** Return the _parsed and validated_ JSON dictionary (containing `article_groups`) or `None` if all retries fail or critical errors occur.
    - **Integration:** Ensure this method utilizes the existing `self.rate_limiter` and potentially `self.api_key` (though `genai.configure` handles the key globally).
3.  **Documentation:** Add comprehensive header comments and docstrings to the new method within `gemini_client.py`. Update the class docstring if necessary.

**Phase 2: Refactor `step4.py`**

1.  **Convert to Async:** As planned (change `run` to `async def`, import `asyncio`, update `__main__` block).
2.  **Import:** Ensure the import remains `from src.gemini.gemini_client import GeminiClient`.
3.  **Prepare Data for Gemini:**
    - After retrieving articles, iterate and create the `articles_for_analysis` list containing dictionaries with _only_ the fields specified in `src/prompts/step4.txt`: `article_id`, `title`, `domain`, `pub_date`, `cluster_id`, `frame_phrases`, `top_entities` (ensure entities only include fields needed by the prompt: `entity_id`, `name`, `entity_type`, `is_influential_context`, `mention_count`).
4.  **Call Gemini Client:**
    - Instantiate `client = GeminiClient()`.
    - Call `analysis_result = await client.analyze_articles_with_prompt(...)` with:
      - `articles_data=articles_for_analysis`
      - `prompt_file_path="src/prompts/step4.txt"`
      - `model_name="models/gemini-2.0-flash-thinking-exp-01-21"` (or fetch from env/config).
5.  **Process and Save Response:**
    - Check if `analysis_result` is `None`. If so, log error, set status failure, and return.
    - The result is already parsed and validated JSON.
    - Create output directory (`src/output/`).
    - Generate timestamped filename (e.g., `step4_analysis_output_YYYYMMDD_HHMMSS.json`).
    - Write `analysis_result` directly to the file using `json.dump(analysis_result, f, indent=2)`.
6.  **Update Status Reporting:** As planned (include counts, update output file path, set success/error).
7.  **Documentation:** Update header and comments in `step4.py`.

**Phase 3: Update `step4.txt` Prompt**

1.  **Placeholder:** Ensure the prompt contains a clear placeholder like `{INPUT_DATA_JSON}` for the article list injection.
2.  **Clarity:** Review prompt instructions to ensure they align with the expected input structure (`articles_for_analysis`) and desired output format (JSON object with `article_groups`). Emphasize the need for valid JSON output.
