# Implementation Plan: Enhancing Step 3 Semantic Extraction

**Goal:** Improve the semantic richness of data extracted in Step 3 by utilizing existing Gemini API calls more effectively to classify entity roles, differentiate relationship contexts, and select higher-quality evidence snippets.

**Approach:** Modify the Step 3 Gemini prompt (`entity_extraction_prompt.txt`) to request more specific information and update the response parsing logic in `src/steps/step3/__init__.py` to handle the refined output. This approach avoids adding new complex NLP models, focusing on leveraging the existing infrastructure.

---

## Phase 1: Prompt Refinement ✅ COMPLETE

**Objective:** Update the Gemini prompt to explicitly request entity roles, relationship context types, and concise, relevant evidence snippets.

1.  **Modify `src/prompts/entity_extraction_prompt.txt`:** ✅ COMPLETE
    - **Entity Roles:** Added instructions for Gemini to determine the specific role of each entity mentioned in connection with identified _events_ and _policies_.
      - Defined a controlled vocabulary within the prompt (e.g., for events: `ORGANIZER`, `PARTICIPANT`, `SPEAKER`, `SUBJECT`, `MENTIONED`; for policies: `AUTHOR`, `ENFORCER`, `SUBJECT`, `MENTIONED`).
      - Instructed Gemini to default to `MENTIONED` if a specific role cannot be determined confidently.
    - **Relationship Context Types:** Added instructions for Gemini to classify the _context type_ for each identified _entity relationship_.
      - Defined a controlled vocabulary within the prompt (e.g., `COLLABORATION`, `CONFLICT`, `FINANCIAL`, `LEGAL`, `MEMBERSHIP`, `FAMILY`, `POLITICAL_ALIGNMENT`, `GEOGRAPHIC_PROXIMITY`, `OTHER`).
      - Instructed Gemini to select the single most fitting context type.
    - **Evidence Snippet Quality:** Added instructions for Gemini to provide only the _single, most representative sentence_ as the evidence snippet (`evi`) for each _entity relationship_. The sentence should clearly demonstrate the relationship between the two entities.
    - **Update JSON Output Specification:** Updated the expected JSON structure in the prompt examples, incorporating fields for `role` (within `ev_mentions` and `pol_mentions` entity lists) and modified `evi` field in `rel_contexts` to expect a single string (the best sentence).

## Phase 2: Response Parsing Logic Update ✅ COMPLETE

**Objective:** Modify the Python code in Step 3 to correctly parse and store the new information requested from Gemini.

1.  **Modify `src/steps/step3/__init__.py` (`_store_results` function):** ✅ COMPLETE
    - **Parse Entity Roles:**
      - Updated code to check if entity mentions are in the new format (dictionaries with `en` and `role` fields) or old format (simple strings)
      - Added logic to extract the role field from the new format while maintaining backward compatibility
      - Modified the calls to `link_event_entity()` and `link_policy_entity()` to pass the extracted role instead of the hardcoded default
    - **Parse Relationship Context Type:**
      - The existing code already correctly extracts and uses the context type from Gemini's response
    - **Parse Evidence Snippet:**
      - Updated the code to handle both formats: the new format with a single evidence string and the old format with a list of strings
      - Added logic to properly extract the single evidence snippet in both cases

## Phase 3: Database Module Confirmation ✅ COMPLETE

**Objective:** Quickly verify that the database interaction functions handle the new data correctly.

1.  **Review `src/database/modules/events.py`, `policies.py`, `relationships.py`:** ✅ COMPLETE
    - **Confirmed** that `link_event_entity` already correctly accepts and stores the `role` parameter, with appropriate default to 'MENTIONED'
    - **Confirmed** that `link_policy_entity` already correctly accepts and stores the `role` parameter
    - **Confirmed** that `record_relationship_context` already correctly handles the `context_type` parameter and `evidence_snippet` as a string

## Phase 4: Testing and Validation

**Objective:** Ensure the changes function as expected and improve data quality.

1.  **Execute Modified Step 3:** Run the pipeline with the updated Step 3 on a representative sample of new articles.
2.  **Database Verification:**
    - Query `event_entities` and `policy_entities`: Verify that `role` columns now contain specific roles beyond just `'MENTIONED'`.
    - Query `entity_relationships`: Verify that `context_type` shows varied classifications and that `metadata->>'evidence'` contains concise, relevant sentences.
3.  **Log Monitoring:** Check logs for any new JSON parsing errors or warnings related to the updated extraction logic.
4.  **Qualitative Review:** Manually review a sample of the extracted roles, context types, and evidence snippets to assess the quality improvement.

---

## Implementation Summary

We have successfully enhanced the semantic richness of the entity extraction process by:

1. **Improving the Gemini Prompt:** The modified prompt now explicitly asks for specific entity roles, better relationship context classification, and higher quality evidence snippets.

2. **Updating the Response Parser:** The code now properly handles the enhanced output format while maintaining backward compatibility with the old format.

3. **Confirming Database Support:** We verified that the database functions already correctly handle the role, context type, and evidence snippet parameters.

These changes will significantly improve the quality of the extracted data by:

- Capturing more specific relationships between entities and events/policies
- Providing more precise classifications of entity relationships
- Improving the quality of evidence snippets by focusing on the single most representative sentence

Once the testing phase is complete, these enhancements will provide more valuable insights into entity relationships and their roles in different contexts.
