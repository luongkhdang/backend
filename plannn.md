# Implementation Plan: Adding Relational Entity Tables (Revised)

This plan outlines the steps required to add new database tables for events, policies, and entity relationships. It focuses on using these tables to **structure and connect data within the database**, with `step3` identifying and storing relational _signals_ or _mentions_ rather than performing full relational extraction.

**Goal:** Enhance the database schema to capture structured relational information (event mentions, policy mentions, entity co-occurrence contexts) identified during `step3`, enabling further analysis and enrichment based on aggregated database data.

**Key Files Involved:**

- `src/database/modules/schema.py`: Define new table structures.
- `src/database/schema.md`: Document new table structures.
- `src/database/reader_db_client.py`: Add client methods for new tables.
- `src/database/modules/` (New files: `events.py`, `policies.py`, `relationships.py`): Implement DB logic for new tables.
- `src/prompts/entity_extraction_prompt.txt`: Modify the LLM prompt to extract relational _signals_ and _mentions_.
- `src/steps/step3/__init__.py`: Update processing logic to parse mentions and store basic linkage data.

**Phases:**

**Phase 1: Database Schema and Client Updates**

1.  **Define New Tables (`src/database/modules/schema.py`):**

    - Implement `events`, `event_entities`, `entity_relationships`, `policy_details`, and `policy_entities` tables as previously defined. The structure remains valid for storing the connections.
    - Ensure columns, constraints (PK, FK with `ON DELETE CASCADE`), data types, defaults, and indexes are correctly implemented.
    - Consider if `events.description` and `policy_details.description` should be nullable or populated later.
    - The `entity_relationships.relationship_type` will store simpler context types initially (e.g., COOPERATION_CONTEXT). `confidence_score` might be initially low or based on frequency.

2.  **Create New Database Modules:**

    - Create `src/database/modules/events.py`: Implement `find_or_create_event(conn, title, event_type, date_mention=None) -> event_id`, `link_event_entity(conn, event_id, entity_id, role='MENTIONED')`. The role will likely be basic initially.
    - Create `src/database/modules/policies.py`: Implement `find_or_create_policy(conn, title, policy_type, date_mention=None) -> policy_id`, `link_policy_entity(conn, policy_id, entity_id, role='MENTIONED')`.
    - Create `src/database/modules/relationships.py`: Implement `record_relationship_context(conn, entity_id_1, entity_id_2, context_type, article_id, evidence_snippet)`. This function will use `ON CONFLICT` to:
      - Insert if the pair (e1, e2) doesn't exist.
      - If it exists, increment `article_count`.
      - Append the new `article_id` and `evidence_snippet` to the `metadata` JSONB field.
      - Update `last_updated`.
      - Optionally adjust `confidence_score` based on rules (e.g., increase slightly with more articles).

3.  **Update `ReaderDBClient` (`src/database/reader_db_client.py`):**

    - Import the new modules.
    - Add corresponding public methods (e.g., `find_or_create_event`, `link_event_entity`, `record_relationship_context`).

4.  **Update Schema Documentation (`src/database/schema.md`):**
    - Document the new tables accurately.
    - Clarify the intended initial population strategy (mentions from `step3`, potential later enrichment).

**Phase 2: Modify Gemini Extraction Prompt**

1.  **Update Prompt (`src/prompts/entity_extraction_prompt.txt`):**
    - **Task Definition:** Adjust instructions to focus on identifying _mentions_:
      - **Event Mentions:** Identify mentions of potential events. Extract the _name/title_ as mentioned, a likely _type_ (e.g., MEETING, CONFLICT, STATEMENT), any _date_ mentioned nearby, and the _names of entities_ mentioned in the immediate context of the event mention.
      - **Policy Mentions:** Identify mentions of potential policies/laws/agreements. Extract the _name/title_ as mentioned, a likely _type_, any _date_ mentioned nearby, and the _names of entities_ mentioned in the immediate context.
      - **Entity Co-occurrence Contexts:** Identify sentences/paragraphs where _pairs of significant entities_ are mentioned together in a meaningful context (e.g., signing an agreement, engaging in conflict, issuing a joint statement). Extract the _names of the two entities_, a simple _context type_ (e.g., AGREEMENT_CONTEXT, CONFLICT_CONTEXT, JOINT_STATEMENT_CONTEXT, SHARED_LOCATION_CONTEXT), and the _snippet_ showing the co-occurrence.
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

**Phase 3: Update Step 3 Processing Logic**

1.  **Modify `_store_results` (`src/steps/step3/__init__.py`):**
    - **Parse New JSON:** Update parsing for `ev_mentions`, `pol_mentions`, `rel_contexts`.
    - **Process Event Mentions:**
      - Iterate through `ev_mentions`.
      - For each `ev`: Call `db_client.find_or_create_event` using `ev['ti']`, `ev['ty']`, `ev.get('dt')`. Get `event_id`.
      - If `event_id`: Iterate through `ev['ent_mens']`. For each `en`:
        - `entity_id = db_client.find_or_create_entity(name=en)`.
        - If `entity_id`: `db_client.link_event_entity(event_id, entity_id, role='MENTIONED')`.
    - **Process Policy Mentions:**
      - Iterate through `pol_mentions`.
      - For each `po`: Call `db_client.find_or_create_policy` using `po['ti']`, `po['ty']`, `po.get('edt')`. Get `policy_id`.
      - If `policy_id`: Iterate through `po['ent_mens']`. For each `en`:
        - `entity_id = db_client.find_or_create_entity(name=en)`.
        - If `entity_id`: `db_client.link_policy_entity(policy_id, entity_id, role='MENTIONED')`.
    - **Process Relationship Contexts:**
      - Iterate through `rel_contexts`.
      - For each `re`:
        - `entity_id_1 = db_client.find_or_create_entity(name=re['e1n'])`.
        - `entity_id_2 = db_client.find_or_create_entity(name=re['e2n'])`.
        - If `entity_id_1` and `entity_id_2` valid and `entity_id_1 != entity_id_2`:
          - `evidence_snippet = re['evi'][0] if re['evi'] else None`
          - `db_client.record_relationship_context(entity_id_1, entity_id_2, re['ctx_ty'], article_id, evidence_snippet)`.
    - **Error Handling & Counters:** Maintain robust error handling and update relevant counters.

**Phase 4: Testing, Refinement, and Future Enrichment**

1.  **Testing:** Perform unit, integration, and end-to-end testing as before.
2.  **Prompt Engineering:** Refine the prompt to accurately capture the desired _mentions_ and _context types_.
3.  **Acknowledge Enrichment Need:** Explicitly document (e.g., in `README.md` or `schema.md`) that the relational tables are initially populated with mentions/contexts from individual articles via `step3`. Deeper analysis, description generation, confidence scoring, and complex relationship inference (e.g., determining ALLY/ADVERSARY based on aggregated contexts over time) should be handled by a separate, future process or analysis step that operates on the aggregated data in these tables.

**Success Metrics:**

- New tables created successfully.
- `step3` runs without critical errors.
- Relational _mentions_ (event/policy links, entity co-occurrence contexts) are populated correctly.
- The system is positioned for future enrichment steps based on the collected relational signals.
- Documentation reflects the revised strategy.

This revised plan aligns better with using the database as the central point for relationship building, leveraging `step3` and Gemini primarily for signal extraction from individual articles.
