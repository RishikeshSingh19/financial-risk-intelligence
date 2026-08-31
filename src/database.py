import os

import psycopg


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "credit_risk_db")
DB_USER = os.getenv("DB_USER", "rishikeshsingh")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_connection():
    """Create and return a PostgreSQL database connection."""
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )