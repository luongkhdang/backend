"""
events.py - Module for managing events in the Reader database

This module provides functions for creating and querying events and managing their
relationships with entities. Events represent significant occurrences such as
meetings, conflicts, statements, etc. that involve entities in the dataset.

Exported functions:
- initialize_tables(conn) -> bool
- find_or_create_event(conn, title, event_type, date_mention) -> int
- link_event_entity(conn, event_id, entity_id, role) -> bool
- get_event_by_id(conn, event_id) -> Dict[str, Any]
- get_events_by_entity(conn, entity_id, limit) -> List[Dict[str, Any]]
- get_recent_events(conn, days, limit) -> List[Dict[str, Any]]

Related files:
- src/database/reader_db_client.py: Uses this module for event operations
- src/database/modules/schema.py: Defines the database schema for events
- src/database/modules/entities.py: Manages entities that participate in events
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
    Initialize events and event_entities tables if they don't exist.
    Called during schema initialization.

    Args:
        conn: Database connection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Create events table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            event_type TEXT NOT NULL,
            date_mention TEXT,
            description TEXT,
            first_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mention_count INTEGER DEFAULT 1,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """)

        # Create event_entities junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_entities (
            event_id INTEGER REFERENCES events(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            role TEXT DEFAULT 'MENTIONED',
            first_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            mention_count INTEGER DEFAULT 1,
            PRIMARY KEY (event_id, entity_id)
        );
        """)

        # Create indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_title ON events(title);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_first_mentioned ON events(first_mentioned_at);
        CREATE INDEX IF NOT EXISTS idx_event_entities_entity_id ON event_entities(entity_id);
        """)

        conn.commit()
        logger.info("Events tables initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing events tables: {e}")
        conn.rollback()
        return False


def find_or_create_event(conn, title: str, event_type: str,
                         date_mention: Optional[str] = None) -> Optional[int]:
    """
    Find an existing event by title or create a new one if it doesn't exist.

    Args:
        conn: Database connection
        title: The title/name of the event
        event_type: The type of event (e.g., MEETING, CONFLICT, STATEMENT)
        date_mention: Optional date mentioned in the text in original format

    Returns:
        int or None: Event ID if successful, None otherwise
    """
    try:
        # Check if event already exists
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, mention_count FROM events WHERE title = %s AND event_type = %s",
            (title, event_type)
        )
        result = cursor.fetchone()

        if result:
            # Event already exists, update mention count
            event_id, mention_count = result

            cursor.execute(
                """
                UPDATE events
                SET mention_count = %s,
                    last_mentioned_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (mention_count + 1, event_id)
            )
            conn.commit()
            logger.info(f"Updated existing event '{title}' with ID {event_id}")
            return event_id

        # Event doesn't exist, create it
        cursor.execute(
            """
            INSERT INTO events (title, event_type, date_mention, first_mentioned_at, last_mentioned_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (title, event_type, date_mention)
        )
        event_id = cursor.fetchone()[0]
        conn.commit()

        logger.info(f"Created new event '{title}' with ID {event_id}")
        return event_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Error in find_or_create_event: {e}")
        return None


def link_event_entity(conn, event_id: int, entity_id: int,
                      role: str = 'MENTIONED') -> bool:
    """
    Link an entity to an event with a specific role.

    Args:
        conn: Database connection
        event_id: ID of the event
        entity_id: ID of the entity
        role: The role of the entity in the event (e.g., MENTIONED, ORGANIZER, PARTICIPANT)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Check if the link already exists
        cursor.execute(
            """
            SELECT mention_count FROM event_entities 
            WHERE event_id = %s AND entity_id = %s
            """,
            (event_id, entity_id)
        )
        result = cursor.fetchone()

        if result:
            # Link exists, update mention count and last_mentioned_at
            mention_count = result[0]
            cursor.execute(
                """
                UPDATE event_entities
                SET mention_count = %s,
                    last_mentioned_at = CURRENT_TIMESTAMP,
                    role = %s
                WHERE event_id = %s AND entity_id = %s
                """,
                (mention_count + 1, role, event_id, entity_id)
            )
        else:
            # Create a new link
            cursor.execute(
                """
                INSERT INTO event_entities (event_id, entity_id, role, first_mentioned_at, last_mentioned_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (event_id, entity_id, role)
            )

        conn.commit()
        logger.info(
            f"Linked entity {entity_id} to event {event_id} with role '{role}'")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error in link_event_entity: {e}")
        return False


def get_event_by_id(conn, event_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an event by its ID, including related entities.

    Args:
        conn: Database connection
        event_id: ID of the event

    Returns:
        Dict or None: Event data with related entities or None if not found
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get event data
        cursor.execute(
            """
            SELECT id, title, event_type, date_mention, description, 
                   first_mentioned_at, last_mentioned_at, mention_count, metadata
            FROM events
            WHERE id = %s
            """,
            (event_id,)
        )
        event = cursor.fetchone()

        if not event:
            logger.warning(f"Event with ID {event_id} not found")
            return None

        # Get related entities
        cursor.execute(
            """
            SELECT e.id, e.name, e.entity_type, ee.role, ee.mention_count 
            FROM entities e
            JOIN event_entities ee ON e.id = ee.entity_id
            WHERE ee.event_id = %s
            ORDER BY ee.mention_count DESC
            """,
            (event_id,)
        )
        entities = cursor.fetchall()

        # Convert event to dict and add entities
        event_dict = dict(event)
        event_dict['entities'] = [dict(entity) for entity in entities]

        return event_dict

    except Exception as e:
        logger.error(f"Error in get_event_by_id: {e}")
        return None


def get_events_by_entity(conn, entity_id: int,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get events associated with a specific entity.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        limit: Maximum number of events to return

    Returns:
        List[Dict[str, Any]]: List of events associated with the entity
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        # Get events for the entity
        cursor.execute(
            f"""
            SELECT e.id, e.title, e.event_type, e.date_mention,
                   e.first_mentioned_at, e.last_mentioned_at, e.mention_count,
                   ee.role, ee.mention_count AS entity_mention_count
            FROM events e
            JOIN event_entities ee ON e.id = ee.event_id
            WHERE ee.entity_id = %s
            ORDER BY e.last_mentioned_at DESC
            {limit_clause}
            """,
            (entity_id,)
        )
        events = cursor.fetchall()

        # Convert to list of dicts
        return [dict(event) for event in events]

    except Exception as e:
        logger.error(f"Error in get_events_by_entity: {e}")
        return []


def get_recent_events(conn, days: int = 30,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get events first mentioned within the specified number of days.

    Args:
        conn: Database connection
        days: Number of days to look back
        limit: Maximum number of events to return

    Returns:
        List[Dict[str, Any]]: List of recent events
    """
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        # Calculate the date threshold
        date_threshold = datetime.now() - timedelta(days=days)

        # Get recent events based on last_mentioned_at
        cursor.execute(
            f"""
            SELECT id, title, event_type, date_mention, description,
                   first_mentioned_at, last_mentioned_at, mention_count, metadata
            FROM events
            WHERE last_mentioned_at >= %s
            ORDER BY last_mentioned_at DESC, mention_count DESC
            {limit_clause}
            """,
            (date_threshold,)
        )
        events = cursor.fetchall()

        # Convert to list of dicts
        return [dict(event) for event in events]

    except Exception as e:
        logger.error(f"Error in get_recent_events: {e}")
        return []
