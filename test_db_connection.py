import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
load_dotenv()


def test_news_db_connection():
    """Test connection to the news-db database."""
    print("Testing NEWS DB connection...")

    # Get connection parameters from environment or use defaults
    host = os.getenv("NEWS_DB_HOST", "news-db")
    port = os.getenv("NEWS_DB_PORT", "5432")
    dbname = os.getenv("NEWS_DB_NAME", "postgres")
    user = os.getenv("NEWS_DB_USER", "postgres")
    password = os.getenv("NEWS_DB_PASSWORD", "postgres")

    try:
        import psycopg2

        # Try to connect
        print(f"Connecting to {host}:{port}/{dbname} as {user}...")

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )

        print("Connection successful!")

        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"PostgreSQL version: {version}")

        # Get database info
        cursor.execute("SELECT current_database(), current_user;")
        db_info = cursor.fetchone()
        print(f"Current database: {db_info[0]}")
        print(f"Current user: {db_info[1]}")

        # Get schema info
        cursor.execute("SELECT schema_name FROM information_schema.schemata;")
        schemas = [row[0] for row in cursor.fetchall()]
        print(f"Schemas: {', '.join(schemas)}")

        # Get table info from public schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables: {', '.join(tables)}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"ERROR connecting to News DB: {e}")
        return False


def test_reader_db_connection():
    """Test connection to the reader-db database."""
    print("\nTesting READER DB connection...")

    # Get connection parameters from environment or use defaults
    host = os.getenv("READER_DB_HOST", "reader-ultimate")
    port = os.getenv("READER_DB_PORT", "5432")
    dbname = os.getenv("READER_DB_NAME", "reader_db")
    user = os.getenv("READER_DB_USER", "READER-postgres")
    password = os.getenv("READER_DB_PASSWORD", "READER-postgres")

    try:
        import psycopg2

        # Try to connect
        print(f"Connecting to {host}:{port}/{dbname} as {user}...")

        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )

        print("Connection successful!")

        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"PostgreSQL version: {version}")

        # Get database info
        cursor.execute("SELECT current_database(), current_user;")
        db_info = cursor.fetchone()
        print(f"Current database: {db_info[0]}")
        print(f"Current user: {db_info[1]}")

        # Get schema info
        cursor.execute("SELECT schema_name FROM information_schema.schemata;")
        schemas = [row[0] for row in cursor.fetchall()]
        print(f"Schemas: {', '.join(schemas)}")

        # Get table info from public schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables: {', '.join(tables)}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"ERROR connecting to Reader DB: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test database connections')
    parser.add_argument('--news-only', action='store_true',
                        help='Test only the news-db connection')
    parser.add_argument('--reader-only', action='store_true',
                        help='Test only the reader-db connection')

    args = parser.parse_args()

    news_success = False
    reader_success = False

    if args.news_only:
        news_success = test_news_db_connection()
    elif args.reader_only:
        reader_success = test_reader_db_connection()
    else:
        news_success = test_news_db_connection()
        reader_success = test_reader_db_connection()

    if (args.news_only and news_success) or (args.reader_only and reader_success) or (news_success and reader_success):
        print("\nConnections tests PASSED!")
        return 0
    else:
        print("\nConnection tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
