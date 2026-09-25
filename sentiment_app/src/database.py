"""
SQLite database layer.
Stores every prediction (input text, predicted label, confidence, timestamp)
so the app can show a history table and basic stats.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """Create the predictions table if it doesn't already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(text: str, label: str, confidence: float):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (input_text, predicted_label, confidence, created_at) VALUES (?, ?, ?, ?)",
        (text, label, confidence, datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 20):
    """Return the most recent N predictions, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT input_text, predicted_label, confidence, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_stats():
    """Return total predictions and positive/negative counts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 'Positive'")
    positive = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 'Negative'")
    negative = cursor.fetchone()[0]

    conn.close()
    return {"total": total, "positive": positive, "negative": negative}
