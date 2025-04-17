That's a great thought – optimizing the response structure itself is a valid technique to save tokens. Let's analyze the current structure and explore ways to make it more compact.

**Analysis of Current Token Usage:**

The example output (8036 tokens) is generated from a relatively verbose JSON structure:

- **Keys:** Full descriptive names like `entity_name`, `mention_count_article`, `is_influential_context`, `supporting_snippets`, `extracted_entities` are used repeatedly for each entity.
- **Values:** Strings (names, types, snippets), booleans (`true`/`false`), integers.
- **Structure:** Standard JSON syntax (`{}`, `[]`, `:`, `,`, `"`).
- **Whitespace:** The pretty-printing (indentation, newlines) adds readability but consumes tokens.

**Optimization Strategies:**

We can definitely reduce the token count without necessarily losing the core information. Here are some options, ranging from simple to more complex:

1.  **Minimize Whitespace:** Explicitly instruct the model to produce compact JSON. This is often the easiest win.
2.  **Use Shorter Key Names:** Abbreviate the keys. This saves significant tokens when repeated across many entities.
    - `en`: `entity_name`
    - `et`: `entity_type`
    - `mc`: `mention_count_article`
    - `ic`: `is_influential_context` (could even be `inf`)
    - `ss`: `supporting_snippets`
    - `ents`: `extracted_entities` (for the top-level key)
    - `fr`: `frame_phrases`
3.  **Use Numeric Representations:**
    - **Booleans:** Represent `is_influential_context` as `1` (true) / `0` (false) instead of the words `true`/`false`.
    - **(More Advanced) Entity Types:** Map the string types to integers (e.g., 0: PERSON, 1: ORGANIZATION, 2: GOVERNMENT_AGENCY, etc.). This requires maintaining the mapping outside the LLM (in your application code) but saves tokens for every entity.
4.  **Alternative Structures (More Complex):**
    - **List of Lists:** Instead of `[{"en": "Name", "et": 0, ...}, ...]`, use `[["Name", 0, count, inf, [snip1]], ...]`. This removes key repetition entirely but makes parsing dependent on list order.
    - **Consider carefully:** These might be harder for the LLM to generate consistently and add complexity to your downstream parsing logic.

We aim to create sufficient token headroom, so we can modify the prompt to also request 3-5 descriptive `frame_phrases`. storing these potentially in a new TEXT[] column on the articles table.
