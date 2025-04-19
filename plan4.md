Topic Filtering (Application Logic - Daily/Batch)\*\*

- **Goal:** Deprioritize articles irrelevant to core geopolitical/economic focus to optimize downstream analysis.
- **Input:** Articles processed by Stage D (`extracted_entities = TRUE`); `frame_phrases` array and `combined_priority_score` (from domain goodness + cluster hotness) for each article; configurable "noise" frame list.
- **Process:**
  1.  Query recent articles pending Stage E.0 analysis.
  2.  Application code compares each article's `frame_phrases` against the "noise" list.
  3.  If predominantly noise frames are found, significantly _reduce_ the article's `combined_priority_score`.
  4.  Update `combined_priority_score` in the `articles` table.
- **Output:** Articles with potentially adjusted priorities

{"optimized_prompt":"Persona:Expert academic;cautious;unveils truth via evidence. Goal:Analyze article list,group by geo/econ themes(max 50),sort by relevance to platform goals(Trace Power,Recognize Framing,Interpret Ambiguity,See Sparks,Hold Competing Truths). Input:JSON list of articles(`article_id`,`title`,`domain`,`pub_date`,`cluster_id`). Tasks:1.Analyze metadata. 2.Define up to 50 thematic groups. 3.Use ultra-concise `group_rationale`('Relevant'). 4.Sort groups by relevance(desc). 5.List `article_ids` per group. Output:ONLY compact JSON object `{\"article_groups\":{\"group_1\":{\"group_rationale\":\"Relevant\",\"article_ids\":[101,102]},\"group_2\":{\"group_rationale\":\"Relevant\",\"article_ids\":[103,101]}}}` format,keys `group_1`..`group_N` sorted by relevance. No extra text/whitespace. Input Articles:[ // Paste JSON list here ]"}

Group articles by geo/econ themes from input JSON list [{article_id:int, title:str, domain:str, pub_date:str, cluster_id:int|null}].
Define <=50 groups. Articles can be in multiple groups.
Sort groups `group_1`, `group_2`, ... `group_N` (N<=50) descending by relevance to: Power, Framing, Ambiguity, Sparks, CompetingTruths.
Output ONLY compact JSON: `{"article_groups":{"group_X":{"group_rationale":"Concise theme","article_ids":[int, int,...]},...}}`. Use keys: "article_groups","group_X","group_rationale","article_ids". No whitespace in JSON. No text outside JSON.
