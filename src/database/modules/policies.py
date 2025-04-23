"""
policies.py - Module for managing policies in the Reader database

This module provides functions for creating and querying policies and managing their
relationships with entities. Policies represent rules, regulations, guidelines, or
formal positions taken by entities.

Exported functions:
- initialize_tables(conn) -> bool
- find_or_create_policy(conn, title, policy_type, description) -> int
- link_policy_entity(conn, policy_id, entity_id, role) -> bool
- get_policy_by_id(conn, policy_id) -> Dict[str, Any]
- get_policies_by_entity(conn, entity_id, limit) -> List[Dict[str, Any]]
- get_recent_policies(conn, days, limit) -> List[Dict[str, Any]]

Related files:
- src/database/reader_db_client.py: Uses this module for policy operations
- src/database/modules/schema.py: Defines the database schema for policies
- src/database/modules/entities.py: Manages entities that interact with policies
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_tables(conn) -> bool:
    """
    Initialize policy_details and policy_entities tables if they don't exist.
    Called during schema initialization.

    Args:
        conn: Database connection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Create policy_details table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS policy_details (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            policy_type TEXT NOT NULL,
            date_mention TEXT,
            description TEXT,
            first_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mention_count INTEGER DEFAULT 1,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """)

        # Create policy_entities junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS policy_entities (
            policy_id INTEGER REFERENCES policy_details(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            role TEXT DEFAULT 'MENTIONED',
            first_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mention_count INTEGER DEFAULT 1,
            PRIMARY KEY (policy_id, entity_id)
        );
        """)

        # Create indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_policies_title ON policy_details(title);
        CREATE INDEX IF NOT EXISTS idx_policies_type ON policy_details(policy_type);
        CREATE INDEX IF NOT EXISTS idx_policies_first_mentioned ON policy_details(first_mentioned_at);
        CREATE INDEX IF NOT EXISTS idx_policy_entities_entity_id ON policy_entities(entity_id);
        """)

        conn.commit()
        logger.info("Policy tables initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing policy tables: {e}")
        conn.rollback()
        return False


def find_or_create_policy(conn, title: str, policy_type: str,
                          description: Optional[str] = None) -> Optional[int]:
    """
    Find an existing policy by title or create a new one if it doesn't exist.

    Args:
        conn: Database connection
        title: The title/name of the policy
        policy_type: The type of policy (e.g., REGULATION, GUIDELINE, POSITION)
        description: Optional brief description of the policy

    Returns:
        int or None: Policy ID if successful, None otherwise
    """
    try:
        # Check if policy already exists
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, mention_count FROM policy_details WHERE title = %s AND policy_type = %s",
            (title, policy_type)
        )
        result = cursor.fetchone()

        if result:
            # Policy already exists, update mention count and timestamp
            policy_id, mention_count = result

            cursor.execute(
                """
                UPDATE policy_details
                SET mention_count = %s,
                    last_mentioned_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (mention_count + 1, policy_id)
            )
            conn.commit()
            logger.info(
                f"Updated existing policy '{title}' with ID {policy_id}")
            return policy_id

        # Policy doesn't exist, create it
        cursor.execute(
            """
            INSERT INTO policy_details (title, policy_type, description, first_mentioned_at, last_mentioned_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (title, policy_type, description)
        )
        policy_id = cursor.fetchone()[0]
        conn.commit()

        logger.info(f"Created new policy '{title}' with ID {policy_id}")
        return policy_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Error in find_or_create_policy: {e}")
        return None


def link_policy_entity(conn, policy_id: int, entity_id: int,
                       role: str = 'MENTIONED') -> bool:
    """
    Link an entity to a policy with a specific role.

    Args:
        conn: Database connection
        policy_id: ID of the policy
        entity_id: ID of the entity
        role: The role of the entity in relation to the policy 
              (e.g., AUTHOR, ENFORCER, SUBJECT, MENTIONED)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Check if the link already exists
        cursor.execute(
            """
            SELECT mention_count FROM policy_entities 
            WHERE policy_id = %s AND entity_id = %s
            """,
            (policy_id, entity_id)
        )
        result = cursor.fetchone()

        if result:
            # Link exists, update mention count and last_mentioned_at
            mention_count = result[0]
            cursor.execute(
                """
                UPDATE policy_entities
                SET mention_count = %s,
                    last_mentioned_at = CURRENT_TIMESTAMP,
                    role = %s
                WHERE policy_id = %s AND entity_id = %s
                """,
                (mention_count + 1, role, policy_id, entity_id)
            )
        else:
            # Create a new link
            cursor.execute(
                """
                INSERT INTO policy_entities (policy_id, entity_id, role, first_mentioned_at, last_mentioned_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (policy_id, entity_id, role)
            )

        conn.commit()
        logger.info(
            f"Linked entity {entity_id} to policy {policy_id} with role '{role}'")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error in link_policy_entity: {e}")
        return False


def get_policy_by_id(conn, policy_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a policy by its ID, including related entities.

    Args:
        conn: Database connection
        policy_id: ID of the policy

    Returns:
        Dict or None: Policy data with related entities or None if not found
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get policy data
        cursor.execute(
            """
            SELECT id, title, policy_type, date_mention, description, 
                   first_mentioned_at, last_mentioned_at, mention_count, metadata
            FROM policy_details
            WHERE id = %s
            """,
            (policy_id,)
        )
        policy = cursor.fetchone()

        if not policy:
            logger.warning(f"Policy with ID {policy_id} not found")
            return None

        # Get related entities
        cursor.execute(
            """
            SELECT e.id, e.name, e.entity_type, pe.role, pe.mention_count 
            FROM entities e
            JOIN policy_entities pe ON e.id = pe.entity_id
            WHERE pe.policy_id = %s
            ORDER BY pe.mention_count DESC
            """,
            (policy_id,)
        )
        entities = cursor.fetchall()

        # Convert policy to dict and add entities
        policy_dict = dict(policy)
        policy_dict['entities'] = [dict(entity) for entity in entities]

        return policy_dict

    except Exception as e:
        logger.error(f"Error in get_policy_by_id: {e}")
        return None


def get_policies_by_entity(conn, entity_id: int,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get policies associated with a specific entity.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        limit: Maximum number of policies to return

    Returns:
        List[Dict[str, Any]]: List of policies associated with the entity
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        # Get policies for the entity
        cursor.execute(
            f"""
            SELECT p.id, p.title, p.policy_type, p.date_mention,
                   p.first_mentioned_at, p.last_mentioned_at, p.mention_count,
                   pe.role, pe.mention_count AS entity_mention_count
            FROM policy_details p
            JOIN policy_entities pe ON p.id = pe.policy_id
            WHERE pe.entity_id = %s
            ORDER BY p.last_mentioned_at DESC
            {limit_clause}
            """,
            (entity_id,)
        )
        policies = cursor.fetchall()

        # Convert to list of dicts
        return [dict(policy) for policy in policies]

    except Exception as e:
        logger.error(f"Error in get_policies_by_entity: {e}")
        return []


def get_recent_policies(conn, days: int = 30,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get policies created within the specified number of days.

    Args:
        conn: Database connection
        days: Number of days to look back
        limit: Maximum number of policies to return

    Returns:
        List[Dict[str, Any]]: List of recent policies
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        # Calculate the date threshold
        date_threshold = datetime.now() - timedelta(days=days)

        # Get recent policies based on last_mentioned_at
        cursor.execute(
            f"""
            SELECT id, title, policy_type, date_mention, description,
                   first_mentioned_at, last_mentioned_at, mention_count, metadata
            FROM policy_details
            WHERE last_mentioned_at >= %s
            ORDER BY last_mentioned_at DESC, mention_count DESC
            {limit_clause}
            """,
            (date_threshold,)
        )
        policies = cursor.fetchall()

        # Convert to list of dicts
        return [dict(policy) for policy in policies]

    except Exception as e:
        logger.error(f"Error in get_recent_policies: {e}")
        return []
