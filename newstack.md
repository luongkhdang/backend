# Implementation Plan: Tech Stack Upgrade (Haystack 2.x & google-genai)

**Goal:** Upgrade key libraries to newer versions, notably migrating from `google-generativeai` to `google-genai` and updating Haystack components.

**Target Versions:**

- `haystack-ai`: 2.13.1
- `google-ai-haystack`: >=5.1.0
- `sentence-transformers`: 4.1.0
- `google-genai`: 1.12.1 (Replacing `google-generativeai` 0.8.5)

**Affected Files:**

- `src/gemini/gemini_client.py` (High Impact)
- `src/gemini/modules/generator.py` (High Impact)
- `src/haystack/haystack_client.py` (Medium Impact)
- `src/utils/task_manager.py` (Medium Impact)
- `src/utils/rate_limit.py` (Low-Medium Impact)
- `src/main.py` (Low Impact)
- Dependency Files (`requirements.txt`, `Dockerfile`, `docker-compose.yml`) (High Impact)

---

## Phase 3: Dependent Module Updates

1.  **Update `src/utils/task_manager.py`:**
    - **`_run_single_task`:** Modify the call to `gemini_client.generate_text_with_prompt_async` to match the new signature defined in Phase 2.
    - **Result Handling:** Adjust how results are extracted and checked for errors based on the new response structure from `google-genai`.
    - **Error Handling:** Update exception handling for API calls made within the task manager, aligning with Phase 2 changes.
    - **Model Cooldown Logic:** Review if the rate limit error detection (`is_rate_limit_error`) and cooldown mechanism still function correctly with `google-genai` error messages/types.
2.  **Review `src/utils/rate_limit.py`:**
    - **Necessity:** Determine if the custom rate limiter is still strictly necessary or if `google-genai` or `google-ai-haystack` provide sufficient built-in handling. Assume it's kept for now for fine-grained control.
    - **Model Names:** Ensure model names used as keys match those used in the refactored `GeminiClient` and expected by `google-genai`.
    - **Integration:** Verify that `wait_if_needed_async` and `register_call_async` are still called correctly from the refactored `GeminiClient` and `generator.py`.

## Phase 4: Haystack Integration & Verification

1.  **Update `src/haystack/haystack_client.py`:**
    - **Imports:** Verify all `haystack` and `haystack_integrations` imports are compatible with `haystack-ai==2.13.1` and `google-ai-haystack>=5.1.0`. Pay attention to component paths (e.g., `haystack.components.rankers`, `haystack_integrations.components.generators.google_ai`).
    - **`get_ranker`:** Ensure `TransformersSimilarityRanker` initialization is correct for Haystack 2.13.1 and uses `sentence-transformers==4.1.0` effectively.
    - **`get_prompt_builder`:** Verify `PromptBuilder` usage aligns with Haystack 2.13.1.
    - **`get_gemini_generator`:**
      - Confirm `GoogleAIGeminiGenerator` initialization (from `google-ai-haystack`) is correct.
      - Ensure the model name passed (`GEMINI_FLASH_THINKING_MODEL`) is valid for `google-genai`.
      - Verify `generation_kwargs` are appropriate for the specified model and library version.
    - **`run_article_retrieval_and_ranking`:**
      - Check `Document` dataclass usage (`from haystack.dataclasses import Document`) aligns with Haystack 2.13.1.
      - Ensure the `ranker.run()` call signature is correct.

## Phase 5: Orchestration & Testing

1.  **Review `src/main.py`:**
    - Check calls to `run_step3`, `run_step4`, `run_step5`. Ensure any changes to the return values or error handling in these async steps are correctly handled in `main`.
2.  **Integration Testing:**
    - **Step 3 (Entity Extraction):** Run the pipeline with `RUN_STEP3=true`. Verify entities are extracted correctly, check logs for `google-genai` related errors, monitor API usage and rate limiting.
    - **Step 4 (Data Export):** Run with `RUN_STEP4=true`. This step might use `analyze_articles_with_prompt`. Verify its successful execution and output.
    - **Step 5 (RAG Essay):** Run with `RUN_STEP5=true`. This involves `haystack_client` (retrieval/ranking) and potentially `gemini_client` or `GoogleAIGeminiGenerator` (generation). Verify the full RAG flow, check prompt assembly, and essay generation.
    - **Monitor Logs:** Closely monitor application logs during testing for any new errors or warnings related to the upgraded libraries.
    - **API Quotas:** Monitor Gemini API quotas during testing.

## Phase 6: Documentation & Cleanup

1.  **Update Code Comments & Docstrings:**
    - Modify headers and docstrings in affected files (`gemini_client.py`, `generator.py`, etc.) to reflect the use of `google-genai` and updated Haystack versions.
    - Remove comments related to `google.generativeai`.
2.  **Remove Obsolete Code:** Delete any helper functions or logic specifically tied to the old `google.generativeai` library that are no longer needed.
3.  **Update Project README:** If applicable, update the project README with the new library versions.

---

**Important Considerations:**

- **API Key:** Ensure the `GEMINI_API_KEY` environment variable is correctly set and valid for the `google-genai` library.
- **Error Handling:** `google-genai` might have different exception types and error messages compared to `google-generativeai`. Thoroughly test error conditions (e.g., invalid API key, quota exceeded, invalid model name).
- **Performance:** Monitor the performance (latency, throughput) of Gemini API calls after migrating to `google-genai`.
- **Haystack Migration Guide:** Refer to the official Haystack 1.x to 2.x migration guide for any subtle API changes or best practices for version 2.13.1.
- **Google-genai Documentation:** Consult the `google-genai` library documentation for detailed API references and examples.
