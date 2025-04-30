"""
essays.py - Essay-related database operations

This module provides functions for managing essays in the database,
including inserting, linking, and retrieving essays.

Exported functions:
- insert_essay(conn, essay: Dict[str, Any]) -> Optional[int]
  Inserts an essay into the database
- link_essay_entity(conn, essay_id: int, entity_id: int) -> bool
  Links an essay to an entity
- get_essay_by_id(conn, essay_id: int) -> Optional[Dict[str, Any]]
  Gets an essay by its ID
- get_essays_by_cluster(conn, cluster_id: int) -> List[Dict[str, Any]]
  Gets essays associated with a cluster

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for essay operations
- Entities in modules/entities.py
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def insert_essay(conn, essay: Dict[str, Any]) -> Optional[int]:
    """
    Insert an essay into the database using the revised schema.

    Args:
        conn: Database connection
        essay: The essay data to insert. Expected keys match the revised schema:
               'group_id', 'cluster_id', 'type', 'title', 'content',
               'source_article_ids', 'model_name', 'generation_settings',
               'input_token_count', 'output_token_count', 'tags', 'prompt_template_hash'.

    Returns:
        int or None: The new essay ID if successful, None otherwise
    """
    # Define the columns to insert/update (excluding 'id' and default 'created_at')
    columns = [
        'group_id', 'cluster_id', 'type', 'title', 'content',
        'source_article_ids', 'model_name', 'generation_settings',
        'input_token_count', 'output_token_count', 'tags', 'prompt_template_hash'
    ]
    update_columns = [col for col in columns if col not in (
        'group_id')]  # Exclude group_id from update

    # Prepare the values and placeholders
    values_tuple = (
        essay.get('group_id'),
        essay.get('cluster_id'),
        essay.get('type', 'unknown'),
        essay.get('title', ''),
        essay.get('content', ''),
        essay.get('source_article_ids'),
        essay.get('model_name'),
        json.dumps(essay.get('generation_settings')) if essay.get(
            'generation_settings') else None,
        essay.get('input_token_count'),
        essay.get('output_token_count'),
        essay.get('tags'),
        essay.get('prompt_template_hash')
    )

    # Construct the UPSERT query
    insert_cols_str = ", ".join(columns)
    placeholders_str = ", ".join(['%s'] * len(columns))
    # Construct the UPDATE SET part, excluding group_id
    update_set_str = ", ".join(
        [f"{col} = EXCLUDED.{col}" for col in update_columns])

    sql = f"""
        INSERT INTO essays ({insert_cols_str}, created_at)
        VALUES ({placeholders_str}, NOW()) -- Use NOW() for created_at on insert
        ON CONFLICT (group_id, DATE(created_at))
        DO UPDATE SET {update_set_str},
                      created_at = NOW() -- Explicitly update created_at on conflict too
        RETURNING id;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql, values_tuple)
        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"Successfully upserted essay {new_id} of type \'{essay.get('type', 'unknown')}\' for group \'{essay.get('group_id')}\'.")
        return new_id

    except Exception as e:
        logger.error(f"Error upserting essay: {e}", exc_info=True)
        conn.rollback()
        return None


def link_essay_entity(conn, essay_id: int, entity_id: int,
                      mention_count: int = 1,
                      relevance_score: Optional[float] = None,
                      first_mention_offset: Optional[int] = None) -> bool:
    """
    Link an essay to an entity, including new fields from the revised schema.

    Args:
        conn: Database connection
        essay_id: ID of the essay
        entity_id: ID of the entity
        mention_count: Number of times entity mentioned in this essay
        relevance_score: Calculated relevance of entity within this essay
        first_mention_offset: Character offset of the first mention

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Insert the link with new fields, update on conflict if needed (or just DO NOTHING)
        # Option 1: Simple insert, ignore conflict (assumes first link is definitive or updates happen elsewhere)
        cursor.execute("""
            INSERT INTO essay_entities (
                essay_id, entity_id, mention_count_in_essay, relevance_score, first_mention_offset
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (essay_id, entity_id) DO NOTHING;
        """, (
            essay_id,
            entity_id,
            mention_count,
            relevance_score,
            first_mention_offset
        ))

        # Option 2: Update on conflict (if subsequent mentions should update counts/scores)
        # cursor.execute("""
        #     INSERT INTO essay_entities (
        #         essay_id, entity_id, mention_count_in_essay, relevance_score, first_mention_offset
        #     )
        #     VALUES (%s, %s, %s, %s, %s)
        #     ON CONFLICT (essay_id, entity_id) DO UPDATE SET
        #         mention_count_in_essay = essay_entities.mention_count_in_essay + EXCLUDED.mention_count_in_essay,
        #         relevance_score = GREATEST(essay_entities.relevance_score, EXCLUDED.relevance_score), -- Example: keep highest score
        #         -- first_mention_offset typically wouldn't be updated
        #         first_mention_offset = COALESCE(essay_entities.first_mention_offset, EXCLUDED.first_mention_offset);
        # """, (
        #     essay_id,
        #     entity_id,
        #     mention_count,
        #     relevance_score,
        #     first_mention_offset
        # ))

        conn.commit()
        cursor.close()
        logger.debug(
            f"Attempted to link essay {essay_id} to entity {entity_id} with count {mention_count}.")
        return True

    except Exception as e:
        logger.error(
            f"Error linking essay {essay_id} to entity {entity_id}: {e}", exc_info=True)
        conn.rollback()
        return False


def get_essay_by_id(conn, essay_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an essay by its ID using the revised schema.

    Args:
        conn: Database connection
        essay_id: ID of the essay

    Returns:
        Dict[str, Any] or None: The essay data, or None if not found
    """
    try:
        cursor = conn.cursor()

        # Select all columns from the revised essays schema
        cursor.execute("""
            SELECT
                id, group_id, cluster_id, type, title, content,
                source_article_ids, model_name, generation_settings,
                input_token_count, output_token_count, created_at, tags,
                prompt_template_hash
            FROM essays
            WHERE id = %s
        """, (essay_id,))

        row = cursor.fetchone()

        if not row:
            cursor.close()
            return None

        # Convert row to dictionary using column names
        columns = [desc[0] for desc in cursor.description]
        essay_data = dict(zip(columns, row))

        # Parse generation_settings from JSON string back to dict/list
        if 'generation_settings' in essay_data and essay_data['generation_settings']:
            try:
                essay_data['generation_settings'] = json.loads(
                    essay_data['generation_settings'])
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse generation_settings JSON for essay {essay_id}")
                # Keep the raw string or set to None/Error indicator? Keep raw for now.
                pass

        cursor.close()
        return essay_data

    except Exception as e:
        logger.error(f"Error retrieving essay {essay_id}: {e}", exc_info=True)
        return None


def get_essays_by_cluster(conn, cluster_id: int) -> List[Dict[str, Any]]:
    """
    Get all essays associated with a cluster using the revised schema.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster

    Returns:
        List[Dict[str, Any]]: List of essays for the cluster
    """
    try:
        cursor = conn.cursor()

        # Select all columns from the revised essays schema
        cursor.execute("""
            SELECT
                id, group_id, cluster_id, type, title, content,
                source_article_ids, model_name, generation_settings,
                input_token_count, output_token_count, created_at, tags,
                prompt_template_hash
            FROM essays
            WHERE cluster_id = %s
            ORDER BY created_at DESC
        """, (cluster_id,))

        essays = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            essay_data = dict(zip(columns, row))
            # Parse generation_settings from JSON string back to dict/list
            if 'generation_settings' in essay_data and essay_data['generation_settings']:
                try:
                    essay_data['generation_settings'] = json.loads(
                        essay_data['generation_settings'])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse generation_settings JSON for essay {essay_data.get('id')}")
                    # Keep the raw string or set to None/Error indicator? Keep raw for now.
                    pass
            essays.append(essay_data)

        cursor.close()
        return essays

    except Exception as e:
        logger.error(
            f"Error retrieving essays for cluster {cluster_id}: {e}", exc_info=True)
        return []

# Note: The `save_essay` function used by step5 seems to be handled by
# the `haystack_db.py` module and its wrapper in `ReaderDBClient`.
# The `insert_essay` function here is updated for potential other uses
# or direct calls that might expect this module's interface.
