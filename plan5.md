Okay, to make the `analyze_articles_with_prompt` function in `gemini_advanced.txt` behave more like the `generate_text_with_prompt` functions in `gemini.txt` (specifically, to parse a JSON response), you'll need to modify its code to include JSON parsing logic and adjust its expected return type.

Here's how you can approach the fix, using the `generate_text_with_prompt_async` function in `gemini.txt` as a reference:

1.  **Update the Prompt:**

    - **Crucially**, ensure the prompt template file used by `analyze_articles_with_prompt` (specified by `prompt_file_path`, likely `src/prompts/step4.txt` [cite: 204]) explicitly instructs the Gemini model to format its response as a JSON object. Without this instruction, the model will likely return plain text, and the parsing step will fail.

2.  **Modify `analyze_articles_with_prompt` in `gemini_advanced.txt`:**

    - **Change Return Type:** Update the function signature's return type annotation from `Optional[str]` to `Optional[Dict[str, Any]]` to reflect that it will return a parsed dictionary[cite: 209].
    - **Add JSON Parsing Logic:** After retrieving the `generated_text` successfully, insert the JSON parsing logic similar to that in `gemini.txt`:
      - Find the start and end of the potential JSON object within `generated_text`:
        ```python
        start_index = generated_text.find('{')
        end_index = generated_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_part = generated_text[start_index:end_index+1]
            # Try to parse the JSON
            try:
                parsed_json = json.loads(json_part.strip())
                logger.info(
                    f"Successfully parsed JSON response from {model_name}")
                return parsed_json  # Return the parsed dictionary
            except json.JSONDecodeError as json_err:
                logger.warning(
                    f"JSON parsing error from {model_name}: {json_err}. Raw JSON part: {json_part[:100]}...")
                # Decide how to handle parsing errors - currently returns None below
        else:
            logger.warning(
                f"Generated text from {model_name} does not appear to contain a valid JSON structure: {generated_text[:100]}...")
        ```
    - **Adjust Return Statements:** Ensure that successful parsing returns the `parsed_json` dictionary. If parsing fails (either `JSONDecodeError` or no `{...}` structure found), the function should return `None`. You'll need to remove the final `return generated_text` statement [cite: 225] and ensure all failure paths (including the original error handling) return `None`.
    - **Update Docstrings and Logging:** Modify the function's docstring [cite: 201, 209] and any relevant logging messages to reflect that it now expects, parses, and returns a JSON dictionary instead of raw text.

3.  **Consider Retry Logic (Optional but Recommended):**
    - The current `analyze_articles_with_prompt` in `gemini_advanced.txt` is set for only one attempt (`retries: int = 1` [cite: 206]). If receiving valid JSON is critical, you might want to adopt the more robust retry and model fallback logic present in `generate_text_with_prompt_async`. This would involve handling `JSONDecodeError` or the lack of JSON structure as potential reasons to retry the API call (possibly with the same or a fallback model).

By implementing these changes, particularly ensuring the prompt requests JSON and adding the parsing logic, the `analyze_articles_with_prompt` function will align its response handling with the `generate_text_with_prompt` methods in `gemini.txt`.
