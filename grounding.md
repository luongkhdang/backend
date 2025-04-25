# Implementation Plan: Gemini Features for Step 5 Essay Generation

## 1. Goal

Enhance the RAG essay generation in `src/steps/step5.py` by leveraging advanced Gemini API features:

1.  **(Optional - Currently deemed unsuitable for Step 5 prompt)** Google Search Grounding: To potentially improve factual accuracy and recency by grounding responses in real-time web search results.
2.  **(Experimental)** Thinking Mode: To potentially improve reasoning capabilities by using models that generate their thinking process alongside the final answer.

## 2. Mechanism

- **Grounding:** Utilize the `google-generativeai` library's support for grounding tools.
- **Thinking Mode:** Utilize specific experimental models (e.g., `gemini-2.0-flash-thinking-exp`) and parse the multi-part response to isolate the final answer from the generated "thoughts".

## 3. Affected Files

- `src/gemini/modules/generator.py`: To modify the core API call (model selection, grounding tool, response parsing for thoughts).
- `src/gemini/gemini_client.py`: To pass flags (grounding, thinking) and handle potentially modified responses/return types.
- `src/steps/step5.py`: To control features via environment variables and handle potential metadata (grounding) for saving.
- `src/database/modules/schema.py`: To potentially add columns for grounding metadata or thinking process logs in the `essays` table.
- `src/database/db_setup.py`: To apply schema changes if made.
- `src/database/modules/haystack_db.py`: To handle saving any new metadata.
- `docker-compose.yml`: To add environment variables for enabling/disabling grounding and thinking mode, and selecting the thinking model.

## 4. Implementation Steps

### a. (Schema - Optional but Recommended)

- Edit `src/database/modules/schema.py`.
- **(Grounding):** Add a `grounding_metadata JSONB NULL` column to the `essays` table definition.
- **(Thinking - Optional):** Consider adding a `thinking_process TEXT NULL` or `thinking_metadata JSONB NULL` column if storing the thoughts is desired (currently, the plan is to discard them based on the prompt).
- Ensure `src/database/db_setup.py` correctly applies schema changes.

### b. (Generator Module: `src/gemini/modules/generator.py`)

- Add `from google.ai import generativelanguage as glm` (needed for grounding tool).
- Modify `generate_text_from_prompt` signature:
  - Add `use_grounding: bool = False`.
  - Add `use_thinking_mode: bool = False`.
  - Add `thinking_model_name: Optional[str] = None`.
- **Model Selection:**
  - If `use_thinking_mode` is true and `thinking_model_name` is provided, use `thinking_model_name`.
  - Otherwise, use the standard `model_name` argument.
- **(Grounding) Conditional Grounding Tool:**
  - Prepare the `google_search_retriever` tool as previously planned _only_ if `use_grounding` is true.
- **API Call Modification:**
  - Pass `tools=tools` to `model.generate_content_async` if grounding is enabled.
  - **Thinking Budget:** _Investigate_ where to place the `thinking_budget: 1024` parameter. It might belong in `generation_config` or be specific to the thinking model setup. Add it if a valid parameter is found, e.g., `generation_config["thinking_budget"] = 1024` (Exact parameter name needs confirmation).
- **Response Handling:**
  - After the API call:
    - **(Thinking Mode):** If `use_thinking_mode` was true, parse `response.candidates[0].content.parts`. Identify the final answer part (often the last text part, but needs verification) and discard the preceding "thought" parts. Assign the final answer text to `generated_text`.
    - **(Standard Mode):** Extract `generated_text` as previously planned.
    - **(Grounding):** Extract `grounding_metadata` as previously planned _only_ if `use_grounding` was true.
- **Return Value:** Update the function's return statement and type annotation to potentially include grounding metadata: `return generated_text, grounding_metadata` -> `Tuple[Optional[str], Optional[Dict]]`. (Thinking process is discarded).

### c. (Gemini Client: `src/gemini/gemini_client.py`)

- Modify `generate_essay_from_prompt` signature:
  - Add `use_grounding: bool = False`.
  - Add `use_thinking_mode: bool = False`.
- **Model Logic:**
  - Determine the appropriate model name based on `use_thinking_mode`. Get the thinking model name from an environment variable (e.g., `GEMINI_THINKING_MODEL`).
- **Pass Arguments:** Pass `use_grounding`, `use_thinking_mode`, and the selected `thinking_model_name` (if applicable) to `generator_generate_text`.
- Update the method's return type annotation to `-> Tuple[Optional[str], Optional[Dict]]` and return the tuple received.

### d. (Step 5 Orchestrator: `src/steps/step5.py`)

- Add logic to check environment variables:
  - `ENABLE_GROUNDING = os.getenv("ENABLE_GROUNDING_STEP5", "false").lower() == "true"`
  - `ENABLE_THINKING_MODE = os.getenv("ENABLE_THINKING_MODE_STEP5", "false").lower() == "true"`
- In `process_group`, modify the call to `gemini_client.generate_essay_from_prompt`:
  ```python
  essay_text, grounding_info = await gemini_client.generate_essay_from_prompt(
      full_prompt_text=final_prompt,
      use_grounding=ENABLE_GROUNDING,
      use_thinking_mode=ENABLE_THINKING_MODE
  )
  ```
- Update the `essay_data` dictionary:
  ```python
  essay_data = {
      # ... other fields ...
      "grounding_metadata": grounding_info, # If grounding used
      "generation_settings": { # Add thinking mode info
          # ... existing settings ...
          "thinking_mode_enabled": ENABLE_THINKING_MODE,
          "thinking_model_used": # Get actual model used from client/config
          # Add thinking_budget if confirmed
      },
      # ... rest of the fields ...
  }
  ```

### e. (Database Saving: `src/database/modules/haystack_db.py`)

- Modify the `save_essay` function to handle `grounding_metadata` key.
- Update the SQL `INSERT` statement for the `grounding_metadata` column.
- If storing thoughts, modify schema and INSERT for `thinking_process`.

### f. (Configuration: `docker-compose.yml`)

- Add environment variables under the `article-transfer` service:
  - `ENABLE_GROUNDING_STEP5: "false"` (or `"true"`)
  - `ENABLE_THINKING_MODE_STEP5: "true"` (or `"false"`)
  - `GEMINI_THINKING_MODEL: "models/gemini-2.0-flash-thinking-exp"` (or the desired thinking model)

## 5. Verification

- Run the pipeline with Step 5 and desired features enabled.
- Monitor logs for grounding/thinking mode activation messages and successful response parsing.
- Inspect the `essays` table for relevant metadata.
- Compare essay quality and reasoning capabilities with different feature combinations.
- Verify the `thinking_budget` parameter functionality if confirmed.
