#!/usr/bin/env python3
"""
step5.py - RAG Essay Generation Orchestrator

This script orchestrates the Haystack RAG pipeline for generating analytical essays.
It performs the following steps for each article group defined in group.json:
1.  Load group data (article IDs, rationale).
2.  Fetch current article details (titles, URLs, potentially summaries/snippets).
3.  Identify key entities mentioned across the group's articles.
4.  Retrieve relevant structured data (events, policies, relationships) linked to key entities.
5.  Retrieve relevant historical context articles using Haystack retrieval/ranking.
6.  Select the most relevant context items (historical articles, structured data) - TO BE IMPLEMENTED.
7.  Assemble the final prompt using the selected context and the template.
8.  Call the Gemini API via GeminiClient to generate the essay text.
9.  Extract/Format the generated essay.
10. Save the essay and its metadata to the database.

Usage:
  python src/steps/step5.py [--group-file PATH] [--output-dir PATH]

Options:
  --group-file PATH  Path to the group definition JSON file (default: src/steps/step5/group.json)
  --output-dir PATH  Directory to save generated essays and logs (default: src/output/essays)
"""

import os
import json
import logging
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import sys

# Assuming standard project structure
from src.database.reader_db_client import ReaderDBClient
from src.gemini.gemini_client import GeminiClient
from src.haystack.haystack_client import run_article_retrieval_and_ranking
from haystack.dataclasses import Document  # For type hinting

# Configure logging
log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants --- (Can be moved to a config file later)
DEFAULT_GROUP_FILE = "src/steps/step5/group.json"
DEFAULT_OUTPUT_DIR = "src/output/essays"
DEFAULT_PROMPT_TEMPLATE = "src/prompts/haystack_prompt.txt"
CONTEXT_SELECTION_LIMIT = 40  # Total items (historical docs + structured data)
MAX_HISTORICAL_DOCS_IN_CONTEXT = 20
# Max each for events, policies, relationships
MAX_STRUCTURED_ITEMS_IN_CONTEXT = 20


def load_group_data(group_file_path: str) -> Optional[Dict[str, Any]]:
    """Load group definitions from the JSON file."""
    try:
        with open(group_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "article_groups" not in data:
                logger.error(
                    f"'article_groups' key not found in {group_file_path}")
                return None
            logger.info(f"Loaded group data from {group_file_path}")
            return data["article_groups"]
    except FileNotFoundError:
        logger.error(f"Group file not found: {group_file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {group_file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading group data: {e}", exc_info=True)
        return None


def get_current_article_details(db_client: ReaderDBClient, article_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch detailed context for current articles in the group, including:
    - title, domain, pub_date, content
    - Filtered frame_phrases ("Intriguing_angles", "Theories_and_interpretations")
    - Top 5 influential entities and their associated snippets
    """
    articles_details = []
    TARGET_FRAME_PHRASES = {"Intriguing_angles",
                            "Theories_and_interpretations"}
    MAX_ENTITIES_PER_ARTICLE = 5
    # MAX_SNIPPETS_PER_ENTITY = 3 # Limit snippets per entity if needed

    for article_id in article_ids:
        # Fetch core article data including content and frame phrases
        article_data = db_client.get_article_by_id(article_id)
        if not article_data:
            logger.warning(
                f"Could not retrieve data for article ID: {article_id}")
            continue

        detail = {
            "id": article_data.get("id"),
            "title": article_data.get("title", "N/A"),
            "domain": article_data.get("domain", "N/A"),
            "pub_date": str(article_data.get("pub_date")) if article_data.get("pub_date") else None,
            "content": article_data.get("content", ""),
            "filtered_frame_phrases": [],
            "top_entities_with_snippets": []
        }

        # Filter frame phrases
        frame_phrases = article_data.get("frame_phrases", [])
        if frame_phrases:
            detail["filtered_frame_phrases"] = [phrase for phrase in frame_phrases
                                                if any(target in phrase for target in TARGET_FRAME_PHRASES)]

        # Fetch top entities for this article
        # Using get_top_entities_with_influence_flag to get IDs and names
        top_entities = db_client.get_top_entities_with_influence_flag(
            article_id, limit=MAX_ENTITIES_PER_ARTICLE)

        if top_entities:
            entities_with_snippets_list = []
            for entity_info in top_entities:
                entity_id = entity_info.get('entity_id')
                entity_name = entity_info.get('name', 'Unknown Entity')
                if not entity_id:
                    continue

                # Fetch snippets for this entity within this article
                snippets = db_client.get_article_entity_snippets(
                    article_id=article_id, entity_id=entity_id)
                # Limit snippets if needed, e.g., snippets[:MAX_SNIPPETS_PER_ENTITY]
                snippet_texts = [s.get('snippet', '')
                                 for s in snippets if s and s.get('snippet')]

                entities_with_snippets_list.append({
                    "entity_name": entity_name,
                    "snippets": snippet_texts
                })
            detail["top_entities_with_snippets"] = entities_with_snippets_list

        articles_details.append(detail)

    logger.info(
        f"Fetched detailed context for {len(articles_details)} current articles.")
    return articles_details


def select_context_items(historical_docs: List[Document],
                         related_events: List[str],
                         related_policies: List[str],
                         related_relationships: List[str],
                         max_total: int = CONTEXT_SELECTION_LIMIT,
                         max_docs: int = MAX_HISTORICAL_DOCS_IN_CONTEXT,
                         max_structured: int = MAX_STRUCTURED_ITEMS_IN_CONTEXT) -> Dict[str, List[Any]]:
    """
    Selects the most relevant context items based on limits.
    Simple strategy: Prioritize historical docs, then fill with structured data types equally.
    More sophisticated strategies can be implemented later (e.g., ranking all items).

    Args:
        historical_docs: Ranked list of retrieved historical documents.
        related_events: Formatted strings of related events.
        related_policies: Formatted strings of related policies.
        related_relationships: Formatted strings of related relationships.
        max_total: Maximum total number of context items to return.
        max_docs: Maximum number of historical documents to include.
        max_structured: Maximum number of items per structured data type.

    Returns:
        Dict containing lists of selected items: 'historical_docs', 'events', 'policies', 'relationships'.
    """
    selected_context = {
        "historical_docs": [],
        "events": [],
        "policies": [],
        "relationships": []
    }
    current_total = 0

    # 1. Add historical documents (up to max_docs and max_total)
    docs_to_add = min(len(historical_docs), max_docs,
                      max_total - current_total)
    if docs_to_add > 0:
        selected_context["historical_docs"] = historical_docs[:docs_to_add]
        current_total += docs_to_add
        logger.debug(f"Selected {docs_to_add} historical documents.")

    # 2. Add structured data, trying to balance types
    structured_items = {
        "events": related_events,
        "policies": related_policies,
        "relationships": related_relationships
    }

    structured_types = list(structured_items.keys())
    items_added_structured = 0

    # Distribute remaining slots among structured types
    remaining_slots = max_total - current_total
    if remaining_slots > 0:
        slots_per_type = remaining_slots // len(
            structured_types) if structured_types else 0
        extra_slots = remaining_slots % len(
            structured_types) if structured_types else 0

        for i, type_key in enumerate(structured_types):
            limit_for_type = min(
                len(structured_items[type_key]),  # Available items
                max_structured,                # Max per type
                # Available slots for this type
                slots_per_type + (1 if i < extra_slots else 0)
            )

            if limit_for_type > 0:
                selected_context[type_key] = structured_items[type_key][:limit_for_type]
                items_added_structured += limit_for_type
                logger.debug(
                    f"Selected {limit_for_type} items for type '{type_key}'.")

    logger.info(
        f"Selected {len(selected_context['historical_docs'])} docs, {items_added_structured} structured items. Total: {len(selected_context['historical_docs']) + items_added_structured}")
    return selected_context


def assemble_final_context(group_rationale: str,
                           current_articles_summary: List[Dict[str, Any]],
                           selected_historical_docs: List[Document],
                           selected_events: List[str],
                           selected_policies: List[str],
                           selected_relationships: List[str],
                           prompt_template_path: str = DEFAULT_PROMPT_TEMPLATE) -> Optional[str]:
    """
    Assembles the final prompt string using the provided context and template.

    Args:
        group_rationale: The rationale for the article group.
        current_articles_summary: List of dicts with basic info of current articles.
        selected_historical_docs: List of selected Haystack Document objects for historical context.
        selected_events: List of formatted strings for selected related events.
        selected_policies: List of formatted strings for selected related policies.
        selected_relationships: List of formatted strings for selected entity relationships.
        prompt_template_path: Path to the prompt template file.

    Returns:
        The fully assembled prompt string, or None if the template cannot be loaded.
    """
    try:
        # Load the prompt template
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {prompt_template_path}")
        return None
    except Exception as e:
        logger.error(
            f"Error loading prompt template {prompt_template_path}: {e}")
        return None

    # Format Current articles context into a detailed string
    current_articles_str_list = []
    for i, article in enumerate(current_articles_summary):
        s = f"Current Article {i+1} (ID: {article.get('id', 'N/A')}):\n"
        s += f"  Title: {article.get('title', 'N/A')}\n"
        s += f"  Source: {article.get('domain', 'N/A')} ({article.get('pub_date', 'N/A')})\n"

        # Add filtered frame phrases
        if article.get('filtered_frame_phrases'):
            s += f"  Narrative Frames: {'; '.join(article['filtered_frame_phrases'])}\n"

        # Add top entities and snippets
        if article.get('top_entities_with_snippets'):
            s += "  Key Entities Mentioned:\n"
            for entity_info in article['top_entities_with_snippets']:
                s += f"    - {entity_info.get('entity_name', 'Unknown Entity')}:\n"
                if entity_info.get('snippets'):
                    for snippet in entity_info['snippets']:
                        s += f"      > Snippet: {snippet}\n"
                else:
                    s += "      (No specific snippets retrieved for this entity in this article)\n"

        # Add content (potentially truncated or summarized later if needed)
        s += f"  Content: {article.get('content', 'N/A')}\n"
        current_articles_str_list.append(s)
    current_summary_str = "\n---\n".join(current_articles_str_list)

    # Format Historical articles context into a detailed string
    # Requires fetching extra details for the selected historical docs
    TARGET_FRAME_PHRASES = {"Intriguing_angles",
                            "Theories_and_interpretations"}
    MAX_ENTITIES_PER_ARTICLE = 5

    historical_articles_str_list = []
    try:
        for i, doc in enumerate(selected_historical_docs):
            article_id = doc.meta.get('id')
            # Use title/domain/pub_date directly from doc.meta
            title = doc.meta.get('title', 'N/A')
            domain = doc.meta.get('domain', 'N/A')
            pub_date = doc.meta.get('pub_date', 'N/A')
            # Get frame_phrases directly from doc.meta
            frame_phrases = doc.meta.get(
                'frame_phrases', [])  # Default to empty list

            s = f"Historical Article {i+1} (ID: {article_id if article_id else 'N/A'}):\n"
            s += f"  Title: {title}\n"
            s += f"  Source: {domain} ({pub_date})\n"

            # Add filtered frame phrases (using phrases from meta)
            if frame_phrases:  # Check if list is not None and not empty
                filtered_frames = [phrase for phrase in frame_phrases
                                   if any(target in phrase for target in TARGET_FRAME_PHRASES)]
                if filtered_frames:
                    s += f"  Narrative Frames: {'; '.join(filtered_frames)}\n"
            # else: # Optional: log if frame_phrases were expected but missing
            #     logger.debug(f"No frame_phrases found in metadata for historical article ID: {article_id}")

            # --- Keep the logic for fetching Entities + Snippets using the main db_client ---
            # Need the main db_client passed into this function, or recreate it here if necessary.
            # Assuming db_client is available in this scope or passed in.
            # If not, we need to adjust how db_client is accessed.
            # For now, assuming db_client exists (e.g., passed from process_group)

            # Re-instantiate db_client if needed within the loop, or pass it in.
            # For simplicity here, let's assume db_client needs to be instantiated per loop iteration for now
            # Ideally, pass the main db_client instance to this function.
            # Re-creating - TODO: Refactor to pass main client
            db_client_temp = ReaderDBClient()
            try:
                if article_id:  # Only fetch entities if we have an ID
                    top_entities = db_client_temp.get_top_entities_with_influence_flag(
                        article_id, limit=MAX_ENTITIES_PER_ARTICLE)
                    if top_entities:
                        s += "  Key Entities Mentioned:\n"
                        for entity_info in top_entities:
                            entity_id = entity_info.get('entity_id')
                            entity_name = entity_info.get(
                                'name', 'Unknown Entity')
                            if not entity_id:
                                continue

                            snippets = db_client_temp.get_article_entity_snippets(
                                article_id=article_id, entity_id=entity_id)
                            snippet_texts = [
                                snip.get('snippet', '') for snip in snippets if snip and snip.get('snippet')]

                            s += f"    - {entity_name}:\n"
                            if snippet_texts:
                                for snippet in snippet_texts:
                                    s += f"      > Snippet: {snippet}\n"
                            else:
                                s += "      (No specific snippets retrieved for this entity in this article)\n"
            finally:
                db_client_temp.close()  # Close the temporary client
            # --- End Entities/Snippets ---

            # Add content (use content from the Haystack Document as it was retrieved)
            s += f"  Content: {doc.content}\n"
            historical_articles_str_list.append(s)

    finally:
        # Remove the final close for the removed temporary client
        # db_client_temp.close()
        pass  # No temporary client to close here anymore

    historical_context_str = "\n---\n".join(
        historical_articles_str_list) if historical_articles_str_list else "No relevant historical articles found or processed."

    # Structured data (join lists with newlines)
    events_str = "\n".join(
        f"- {event}" for event in selected_events) if selected_events else "No relevant events identified."
    policies_str = "\n".join(
        f"- {policy}" for policy in selected_policies) if selected_policies else "No relevant policies identified."
    relationships_str = "\n".join(
        f"- {rel}" for rel in selected_relationships) if selected_relationships else "No relevant entity relationships identified."

    # Substitute placeholders in the template
    try:
        # Use str.replace for safer substitution if format specifiers clash
        final_prompt = template\
            .replace("{group_rationale}", group_rationale)\
            .replace("{current_articles_summary}", current_summary_str)\
            .replace("{historical_articles}", historical_context_str)\
            .replace("{related_events}", events_str)\
            .replace("{related_policies}", policies_str)\
            .replace("{entity_relationships}", relationships_str)

        # Alternative using .format() - ensure placeholders don't clash with content
        # final_prompt = template.format(
        #     group_rationale=group_rationale,
        #     current_articles_summary=current_summary_str,
        #     historical_articles=historical_context_str,
        #     related_events=events_str,
        #     related_policies=policies_str,
        #     entity_relationships=relationships_str
        #     # Add {key_questions} later if needed/derived
        # )
        logger.info(
            f"Successfully assembled final prompt (length: {len(final_prompt)} chars)")
        return final_prompt
    except KeyError as e:
        logger.error(
            f"Missing placeholder in prompt template {prompt_template_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error formatting prompt template: {e}")
        return None


def calculate_file_hash(filepath: str) -> Optional[str]:
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(4096)  # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found for hashing: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return None

# --- Main Execution --- (Placeholder for now)


async def process_group(group_id: str, group_info: Dict[str, Any], db_client: ReaderDBClient, gemini_client: GeminiClient, output_dir: Path):
    """Processes a single group to generate an essay."""
    logger.info(f"--- Processing Group: {group_id} ---")
    article_ids = group_info.get("article_ids", [])
    group_rationale = group_info.get(
        "group_rationale", "No rationale provided.")

    if not article_ids:
        logger.warning(f"Skipping group {group_id}: No article IDs found.")
        return

    # 1. Fetch Current Article Details
    logger.debug(f"[{group_id}] Fetching current article details...")
    current_articles_summary = get_current_article_details(
        db_client, article_ids)
    if not current_articles_summary:
        logger.warning(
            f"Skipping group {group_id}: Failed to get details for current articles.")
        return
    logger.debug(
        f"[{group_id}] Found details for {len(current_articles_summary)} current articles.")

    # 2. Identify Key Entities
    logger.debug(f"[{group_id}] Identifying key entities...")
    key_entities_data = db_client.get_key_entities_for_group(
        article_ids, top_n=10)
    key_entity_ids = [entity[0] for entity in key_entities_data]
    key_entity_names = [entity[1] for entity in key_entities_data]
    logger.debug(
        f"[{group_id}] Identified {len(key_entity_ids)} key entities: {key_entity_names}")

    # 3. Fetch Structured Data (Events, Policies, Relationships)
    logger.debug(f"[{group_id}] Fetching related structured data...")
    related_events_str = db_client.get_formatted_related_events(
        key_entity_ids, limit=15)
    related_policies_str = db_client.get_formatted_related_policies(
        key_entity_ids, limit=15)
    related_relationships_str = db_client.get_formatted_related_relationships(
        key_entity_ids, limit=20)
    logger.debug(
        f"[{group_id}] Found {len(related_events_str)} events, {len(related_policies_str)} policies, {len(related_relationships_str)} relationships.")

    # 4. Fetch Historical Articles (using Haystack client)
    logger.debug(f"[{group_id}] Retrieving and ranking historical articles...")
    # Use the group rationale as the query for historical context
    # Pass article_ids for calculating average embedding
    historical_docs: List[Document] = run_article_retrieval_and_ranking(
        query_text=group_rationale, article_ids=article_ids)
    logger.debug(
        f"[{group_id}] Retrieved {len(historical_docs)} historical documents.")

    # 5. Select Context Items
    logger.debug(f"[{group_id}] Selecting context items...")
    selected_context = select_context_items(
        historical_docs,
        related_events_str,
        related_policies_str,
        related_relationships_str
    )

    # 6. Assemble Final Prompt
    logger.debug(f"[{group_id}] Assembling final prompt...")
    final_prompt = assemble_final_context(
        group_rationale=group_rationale,
        current_articles_summary=current_articles_summary,
        selected_historical_docs=selected_context["historical_docs"],
        selected_events=selected_context["events"],
        selected_policies=selected_context["policies"],
        selected_relationships=selected_context["relationships"]
        # prompt_template_path=DEFAULT_PROMPT_TEMPLATE # Use default
    )

    if not final_prompt:
        logger.error(
            f"[{group_id}] Failed to assemble prompt. Skipping generation.")
        return

    # --- Steps 7-10 (Generation, Formatting, Saving) Go Here ---
    # 7. Generate Essay
    logger.debug(f"[{group_id}] Generating essay...")
    # Determine the intended model (can be passed explicitly or defaults in client)
    # For simplicity, let's assume the client's default is used if not overridden.
    # The client method `generate_essay_from_prompt` selects the default.
    # We capture the default used by `generate_essay_from_prompt` for storage.
    intended_model_name = os.getenv("GEMINI_FLASH_THINKING_MODEL",
                                    "models/gemini-2.0-flash-exp")

    essay_text = await gemini_client.generate_essay_from_prompt(
        full_prompt_text=final_prompt
        # Pass explicit model if needed: model_name=intended_model_name
    )

    if not essay_text:
        logger.error(f"[{group_id}] Failed to generate essay text.")
        # Potentially retry or log failure more permanently
        return

    logger.info(
        f"[{group_id}] Successfully generated essay text (length: {len(essay_text)} chars). First 100: {essay_text[:100]}...")

    # 8. Extract/Format Essay (Currently, the prompt asks for just the essay text)
    # Basic formatting/cleanup could happen here if needed (e.g., trimming whitespace)
    final_essay = essay_text.strip()

    # 9. Save Essay to Database
    logger.debug(f"[{group_id}] Preparing essay data for saving...")

    # Calculate prompt template hash
    prompt_hash = calculate_file_hash(DEFAULT_PROMPT_TEMPLATE)

    essay_data = {
        "group_id": group_id,
        # "cluster_id": None, # Add if groups are linked to clusters
        "type": "rag_historical_essay",
        # Generate a simple title
        "title": f"Analysis of Group {group_id}: {group_rationale[:50]}...",
        "content": final_essay,
        "source_article_ids": article_ids,  # Original group article IDs
        # Store the *intended* model name used for the generation attempt
        "model_name": intended_model_name,
        "generation_settings": {  # Store key parameters used
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "context_selection_limit": CONTEXT_SELECTION_LIMIT,
            "max_historical_docs": MAX_HISTORICAL_DOCS_IN_CONTEXT,
            "max_structured_items": MAX_STRUCTURED_ITEMS_IN_CONTEXT,
            # Add GeminiClient generation params if needed/available (e.g., temperature)
        },
        # Token counts omitted for now (requires more complex plumbing)
        # "input_token_count": None,
        # "output_token_count": None,
        "tags": ["rag", "step5"],  # Example tags
        "prompt_template_hash": prompt_hash
    }

    logger.info(f"[{group_id}] Saving essay to database...")
    essay_id = db_client.save_essay(essay_data)

    if essay_id:
        logger.info(
            f"[{group_id}] Successfully saved essay with ID: {essay_id}")
        # Optionally save essay to file as well
        essay_filename = output_dir / f"group_{group_id}_essay_{essay_id}.txt"
        try:
            with open(essay_filename, 'w', encoding='utf-8') as f:
                f.write(final_essay)
            logger.info(f"[{group_id}] Saved essay text to {essay_filename}")
        except IOError as e:
            logger.warning(
                f"[{group_id}] Failed to save essay text to file: {e}")
    else:
        logger.error(f"[{group_id}] Failed to save essay to database.")

    # Placeholder for further steps (removed simulation sleep)
    # await asyncio.sleep(0.1)

    logger.info(f"--- Finished Processing Group: {group_id} ---")
    return essay_id is not None


async def run():
    """
    Main orchestrator for the RAG essay generation pipeline (Step 5).
    """
    logger.info("Starting Step 5: RAG Essay Generation")

    # Use default paths directly instead of parsing arguments
    group_file_path = DEFAULT_GROUP_FILE
    output_dir = Path(DEFAULT_OUTPUT_DIR)

    # Ensure output directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return 1  # Indicate failure

    # Load group definitions
    article_groups = load_group_data(group_file_path)
    if not article_groups:
        logger.error(
            "Failed to load or parse group definitions. Aborting Step 5.")
        return 1

    # Initialize clients (consider dependency injection later)
    try:
        db_client = ReaderDBClient()
        gemini_client = GeminiClient()
        logger.info("Database and Gemini clients initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize clients: {e}", exc_info=True)
        return 1

    # Process each group sequentially
    success_count = 0
    fail_count = 0
    for group_id, group_info in article_groups.items():
        logger.info(f"--- Processing Group: {group_id} ---")
        try:
            # Use await to call the async processing function
            essay_saved = await process_group(group_id, group_info, db_client, gemini_client, output_dir)
            if essay_saved:
                success_count += 1
                logger.info(
                    f"Successfully processed and saved essay for group {group_id}")
            else:
                fail_count += 1
                logger.error(f"Failed to process group {group_id}")
        except Exception as e:
            fail_count += 1
            logger.error(
                f"Unhandled exception processing group {group_id}: {e}", exc_info=True)
        logger.info(f"--- Finished Processing Group: {group_id} ---")

    # Close DB connection pool
    db_client.close()
    logger.info("Database connection pool closed.")

    logger.info(
        f"Step 5 Finished. Processed {len(article_groups)} groups. Success: {success_count}, Failed: {fail_count}")
    # Return 0 on success (even if some groups failed), 1 on major setup failure
    # Or return fail_count > 0 ? Let's return 0 if the process completed.
    return 0


if __name__ == "__main__":
    try:
        # Use asyncio.run() to execute the async run function
        exit_code = asyncio.run(run())
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(
            f"Step 5 failed with unhandled exception: {e}", exc_info=True)
        sys.exit(1)
