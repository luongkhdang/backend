"""
# Removed Haystack version comment as this module is now Haystack-agnostic
haystack_db.py - Database Integration Module for Haystack RAG Pipeline

This module contains functions for database operations specifically required by the
Haystack RAG pipeline, such as retrieving structured data related to entities
and saving generated essays. These functions are designed to be called by the
ReaderDBClient, which handles connection management.

Exported functions:
- get_key_entities_for_group(conn, article_ids, top_n): Identifies key entities for a group.
- get_related_events(conn, entity_ids, limit): Retrieves events related to key entities.
- get_related_policies(conn, entity_ids, limit): Retrieves policies related to key entities.
- get_related_relationships(conn, entity_ids, limit): Retrieves relationships involving key entities.
- save_essay(conn, essay_data): Saves a generated essay to the database.

Helper functions (internal):
- _format_event(event): Formats event dict to string.
- _format_policy(policy): Formats policy dict to string.
- _format_relationship(relationship): Formats relationship dict to string.

Related files:
- src/database/reader_db_client.py: Main database client that calls functions in this module.
- src/steps/step5.py: Script orchestrating the RAG process.
"""

from src.database.modules import essays
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the essays module to call its upsert function


def get_key_entities_for_group(conn, article_ids: List[int], top_n: int = 10) -> List[Tuple]:
    """
    Identifies the key entities mentioned across the articles in a group.

    Args:
        conn: Active database connection.
        article_ids: List of article IDs in the group.
        top_n: Maximum number of key entities to return.

    Returns:
        List[Tuple]: List of key entity tuples (id, name, type, articles_count, mentions_count, influence_score).
    """
    if not article_ids:
        logger.warning(
            "Empty article_ids list provided to get_key_entities_for_group")
        return []

    key_entities = []
    try:
        cursor = conn.cursor()
        query = """
            SELECT
                e.id,
                e.name,
                e.entity_type,
                COUNT(ae.article_id) as articles_in_group_count,
                SUM(ae.mention_count) as total_mentions_in_group,
                e.influence_score
            FROM entities e
            JOIN article_entities ae ON e.id = ae.entity_id
            WHERE ae.article_id = ANY(%s)
            GROUP BY e.id, e.name, e.entity_type, e.influence_score
            ORDER BY articles_in_group_count DESC, total_mentions_in_group DESC, e.influence_score DESC
            LIMIT %s
        """
        cursor.execute(query, (article_ids, top_n))
        key_entities = cursor.fetchall()
        cursor.close()
        logger.info(
            f"Retrieved {len(key_entities)} key entities for group with {len(article_ids)} articles")
        return key_entities
    except Exception as e:
        logger.error(
            f"Error in get_key_entities_for_group: {e}", exc_info=True)
        # Do not rollback here, let the caller handle transaction
        return []


def get_related_events(conn, entity_ids: List[int], limit: int = 15) -> List[Dict[str, Any]]:
    """
    Retrieves events related to the specified key entities.

    Args:
        conn: Active database connection.
        entity_ids: List of entity IDs to find related events for.
        limit: Maximum number of events to return.

    Returns:
        List[Dict[str, Any]]: List of event dictionaries.
    """
    if not entity_ids:
        logger.warning("Empty entity_ids list provided to get_related_events")
        return []

    events = []
    try:
        cursor = conn.cursor()
        query = """
            SELECT DISTINCT
                ev.id,
                ev.title,
                ev.event_type,
                ev.description,
                ev.date_mention,
                ev.last_mentioned_at
            FROM events ev
            JOIN event_entities ee ON ev.id = ee.event_id
            WHERE ee.entity_id = ANY(%s)
            ORDER BY ev.last_mentioned_at DESC
            LIMIT %s
        """
        cursor.execute(query, (entity_ids, limit))
        column_names = [desc[0] for desc in cursor.description]
        events = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        cursor.close()
        logger.info(
            f"Retrieved {len(events)} events related to {len(entity_ids)} entities")
        return events
    except Exception as e:
        logger.error(f"Error in get_related_events: {e}", exc_info=True)
        return []


def get_related_policies(conn, entity_ids: List[int], limit: int = 15) -> List[Dict[str, Any]]:
    """
    Retrieves policies related to the specified key entities.

    Args:
        conn: Active database connection.
        entity_ids: List of entity IDs to find related policies for.
        limit: Maximum number of policies to return.

    Returns:
        List[Dict[str, Any]]: List of policy dictionaries.
    """
    if not entity_ids:
        logger.warning(
            "Empty entity_ids list provided to get_related_policies")
        return []

    policies = []
    try:
        cursor = conn.cursor()
        query = """
            SELECT DISTINCT
                pd.id,
                pd.title,
                pd.policy_type,
                pd.description,
                pd.date_mention,
                pd.last_mentioned_at
            FROM policy_details pd
            JOIN policy_entities pe ON pd.id = pe.policy_id
            WHERE pe.entity_id = ANY(%s)
            ORDER BY pd.last_mentioned_at DESC
            LIMIT %s
        """
        cursor.execute(query, (entity_ids, limit))
        column_names = [desc[0] for desc in cursor.description]
        policies = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        cursor.close()
        logger.info(
            f"Retrieved {len(policies)} policies related to {len(entity_ids)} entities")
        return policies
    except Exception as e:
        logger.error(f"Error in get_related_policies: {e}", exc_info=True)
        return []


def get_related_relationships(conn, entity_ids: List[int], limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves relationships involving the specified key entities.

    Args:
        conn: Active database connection.
        entity_ids: List of entity IDs to find related relationships for.
        limit: Maximum number of relationships to return.

    Returns:
        List[Dict[str, Any]]: List of relationship dictionaries.
    """
    if not entity_ids:
        logger.warning(
            "Empty entity_ids list provided to get_related_relationships")
        return []

    relationships = []
    try:
        cursor = conn.cursor()
        query = """
            SELECT
                er.entity_id_1,
                e1.name as entity1_name,
                er.entity_id_2,
                e2.name as entity2_name,
                er.context_type,
                er.confidence_score,
                er.metadata,
                er.last_updated
            FROM entity_relationships er
            JOIN entities e1 ON er.entity_id_1 = e1.id
            JOIN entities e2 ON er.entity_id_2 = e2.id
            WHERE er.entity_id_1 = ANY(%s) OR er.entity_id_2 = ANY(%s)
            ORDER BY er.confidence_score DESC, er.last_updated DESC
            LIMIT %s
        """
        cursor.execute(query, (entity_ids, entity_ids, limit))
        column_names = [desc[0] for desc in cursor.description]
        relationships = [dict(zip(column_names, row))
                         for row in cursor.fetchall()]
        cursor.close()
        logger.info(
            f"Retrieved {len(relationships)} relationships involving {len(entity_ids)} entities")
        return relationships
    except Exception as e:
        logger.error(f"Error in get_related_relationships: {e}", exc_info=True)
        return []


def _format_event(event: Dict[str, Any]) -> str:
    """
    Formats an event dictionary into a readable string. (Internal helper)

    Args:
        event: Event dictionary.

    Returns:
        str: Formatted event string.
    """
    title = event.get('title', 'Untitled Event')
    event_type = event.get('event_type', 'Unknown Type')
    date_mention = event.get('date_mention', 'Unknown Date')
    description = event.get('description', 'No description available.')
    return f"Event: {title} ({event_type}), Mentioned around: {date_mention}. Description: {description}"


def _format_policy(policy: Dict[str, Any]) -> str:
    """
    Formats a policy dictionary into a readable string. (Internal helper)

    Args:
        policy: Policy dictionary.

    Returns:
        str: Formatted policy string.
    """
    title = policy.get('title', 'Untitled Policy')
    policy_type = policy.get('policy_type', 'Unknown Type')
    date_mention = policy.get('date_mention', 'Unknown Date')
    description = policy.get('description', 'No description available.')
    return f"Policy: {title} ({policy_type}), Mentioned around: {date_mention}. Description: {description}"


def _format_relationship(relationship: Dict[str, Any]) -> str:
    """
    Formats a relationship dictionary into a readable string. (Internal helper)

    Args:
        relationship: Relationship dictionary.

    Returns:
        str: Formatted relationship string.
    """
    entity1_name = relationship.get('entity1_name', 'Unknown Entity')
    entity2_name = relationship.get('entity2_name', 'Unknown Entity')
    context_type = relationship.get('context_type', 'Unknown Context')
    confidence_score = relationship.get('confidence_score', 0.0)

    metadata = relationship.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}

    snippets = metadata.get('evidence_snippets', [])
    snippet_text = ""
    if snippets and len(snippets) > 0:
        snippet = snippets[0]
        snippet_text = f" Evidence: \"{snippet}\""

    return f"Relationship: {entity1_name} and {entity2_name} ({context_type}, Confidence: {confidence_score:.2f}).{snippet_text}"


def save_essay(conn, essay_data: Dict[str, Any]) -> Optional[int]:
    """
    Saves a generated essay to the database using the upsert logic
    defined in the essays module.

    Args:
        conn: Active database connection.
        essay_data: Dictionary containing essay data.

    Returns:
        Optional[int]: Essay ID (new or existing) if successful, None otherwise.
    """
    if not essay_data or 'content' not in essay_data or 'group_id' not in essay_data:
        logger.error("Missing required fields in essay_data for save_essay")
        return None

    # Delegate directly to the essays module's upsert function
    # Transaction management (commit/rollback) is handled within essays.insert_essay
    try:
        essay_id = essays.insert_essay(conn, essay_data)
        return essay_id
    except Exception as e:
        # Log error at this level too, but rollback is handled by the called function
        logger.error(
            f"Error calling essays.insert_essay from haystack_db.save_essay: {e}", exc_info=True)
        # Ensure we still return None on failure
        return None
