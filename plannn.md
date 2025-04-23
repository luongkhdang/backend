# Implementation Plan: Remaining Tasks for Relational Data Capture

This plan outlines the remaining steps required to fully implement the relational data capture functionality, building upon the already completed database schema and client updates (Phase 1 of the original plan).

**Goal:** Complete the system to extract and store structured relational information (event mentions, policy mentions, entity co-occurrence contexts) identified during `step3`, enabling further analysis and enrichment.

**Key Files Involved:**

- `src/prompts/entity_extraction_prompt.txt`: Modify the LLM prompt to extract relational _signals_ and _mentions_.
- `src/steps/step3/__init__.py`: Update processing logic to parse mentions and store basic linkage data.
- `src/database/schema.md`: Document new table structures.

**Remaining Phases:**

**Phase 1 (Was Phase 2): Modify Gemini Extraction Prompt**

1.  **Update Prompt (`src/prompts/entity_extraction_prompt.txt`):**
    - **Task Definition:** Adjust instructions to focus on identifying _mentions_:
      - **Event Mentions:** Identify mentions of potential events. Extract the _name/title_ as mentioned, a likely _type_ (e.g., MEETING, CONFLICT, STATEMENT), any _date_ mentioned nearby, and the _names of entities_ mentioned in the immediate context of the event mention.
      - **Policy Mentions:** Identify mentions of potential policies/laws/agreements. Extract the _name/title_ as mentioned, a likely _type_, any _date_ mentioned nearby, and the _names of entities_ mentioned in the immediate context.
      - **Entity Co-occurrence Contexts:** Identify sentences/paragraphs where _pairs of significant entities_ are mentioned together in a meaningful context (e.g., signing an agreement, engaging in conflict, issuing a joint statement). Extract the _names of the two entities_, a simple _context type_ (e.g., AGREEMENT*CONTEXT, CONFLICT_CONTEXT, JOINT_STATEMENT_CONTEXT, SHARED_LOCATION_CONTEXT), and the \_snippet* showing the co-occurrence.
    - **Output JSON Structure:** Modify the example JSON to reflect the focus on mentions:
      ```json
      {
        "ents": [ {"en": "Entity Name", "et": "Type", ...} ],
        "fr": [ ... ],
        "ev_mentions": [ // Mentions of potential events
          {
            "ti": "Event Title/Identifier Mentioned", // As found in text
            "ty": "Mentioned Type", // e.g., MEETING, STATEMENT
            "dt": "Mentioned Date or text", // Optional
            "ent_mens": ["Entity Name 1", "Entity Name 2"] // Entities mentioned in context
          }
        ],
        "pol_mentions": [ // Mentions of potential policies
          {
            "ti": "Policy Title/Identifier Mentioned",
            "ty": "Mentioned Type", // e.g., LAW, AGREEMENT
            "edt": "Mentioned Date or text", // Optional
            "ent_mens": ["Entity Name 1", "Entity Name 2"]
          }
        ],
        "rel_contexts": [ // Contexts where entities are mentioned together
          {
            "e1n": "Entity Name 1",
            "e2n": "Entity Name 2",
            "ctx_ty": "CONTEXT_TYPE", // e.g., AGREEMENT_CONTEXT, CONFLICT_CONTEXT
            "evi": ["Snippet showing co-mention context"] // Keep snippet
          }
        ]
      }
      ```
    - **Clarity and Constraints:** Emphasize extracting mentions _based solely on the text_. Avoid asking the LLM to infer details not present or assign complex relationship types/confidences based on this single article.

**Phase 2 (Was Phase 3): Update Step 3 Processing Logic**

1.  **Modify `_store_results` (`src/steps/step3/__init__.py`):**
    - **Parse New JSON:** Update parsing logic to handle the new keys: `ev_mentions`, `pol_mentions`, `rel_contexts` in the JSON response from Gemini.
    - **Process Event Mentions:**
      - Iterate through the parsed `ev_mentions` list.
      - For each `ev` object: Call `db_client.find_or_create_event` using `ev['ti']`, `ev['ty']`, `ev.get('dt')`. Get the resulting `event_id`.
      - If a valid `event_id` is obtained: Iterate through the `ev['ent_mens']` list. For each entity name `en`:
        - Find or create the entity using `entity_id = db_client.find_or_create_entity(name=en)`.
        - If a valid `entity_id` is obtained: Link the event and entity using `db_client.link_event_entity(event_id, entity_id, role='MENTIONED')`.
    - **Process Policy Mentions:**
      - Iterate through the parsed `pol_mentions` list.
      - For each `po` object: Call `db_client.find_or_create_policy` using `po['ti']`, `po['ty']`, `po.get('edt')`. Get the resulting `policy_id`.
      - If a valid `policy_id` is obtained: Iterate through the `po['ent_mens']` list. For each entity name `en`:
        - Find or create the entity using `entity_id = db_client.find_or_create_entity(name=en)`.
        - If a valid `entity_id` is obtained: Link the policy and entity using `db_client.link_policy_entity(policy_id, entity_id, role='MENTIONED')`.
    - **Process Relationship Contexts:**
      - Iterate through the parsed `rel_contexts` list.
      - For each `re` object:
        - Find or create the first entity: `entity_id_1 = db_client.find_or_create_entity(name=re['e1n'])`.
        - Find or create the second entity: `entity_id_2 = db_client.find_or_create_entity(name=re['e2n'])`.
        - Ensure both `entity_id_1` and `entity_id_2` are valid and not the same.
        - Extract the evidence snippet: `evidence_snippet = re['evi'][0] if re.get('evi') else None`.
        - Record the context: `db_client.record_relationship_context(entity_id_1, entity_id_2, re['ctx_ty'], article_id, evidence_snippet)` (ensure `article_id` is available in this scope).
    - **Error Handling & Counters:** Ensure robust error handling is maintained or added for these new processing steps. Update relevant counters (e.g., `events_linked`, `policies_linked`, `relationships_recorded`) for the summary report.

**Phase 3 (Was Phase 4): Testing and Documentation**

1.  **Update Schema Documentation (`src/database/schema.md`):**
    - Document the `events`, `event_entities`, `policy_details`, `policy_entities`, and `entity_relationships` tables accurately.
    - Clarify the intended initial population strategy (mentions from `step3`, potential later enrichment).
2.  **Testing:** Perform unit, integration, and end-to-end testing, paying close attention to:
    - Correct parsing of the new JSON structure from Gemini.
    - Correct creation and linking of events, policies, and relationships in the database.
    - Proper handling of errors during parsing and DB operations.
3.  **Prompt Engineering:** Refine the prompt in `src/prompts/entity_extraction_prompt.txt` if testing reveals issues with accurately capturing the desired _mentions_ and _context types_.
4.  **Acknowledge Enrichment Need:** Explicitly document (e.g., in `README.md` or the updated `schema.md`) that the relational tables are initially populated with mentions/contexts from individual articles via `step3`. Deeper analysis, description generation, confidence scoring, and complex relationship inference should be handled by a separate, future process.

**Success Metrics:**

- Gemini prompt updated and effectively extracts relational mentions.
- `step3` processing logic correctly parses and stores event, policy, and relationship data.
- `step3` runs without critical errors related to the new logic.
- Relational _mentions_ (event/policy links, entity co-occurrence contexts) are populated correctly in the database.
- Schema documentation (`schema.md`) is updated.
- The system is positioned for future enrichment steps based on the collected relational signals.
