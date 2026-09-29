"""
database.py
------------
Manages the SQLite prediction-history database.
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "predictions.db")


def init_db():
    """Creates the database folder/file and the predictions table if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            cgpa REAL,
            projects INTEGER,
            internships INTEGER,
            prediction TEXT,
            probability REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_next_student_id():
    """Generates the next Student ID in the format ST001, ST002, ..."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return f"ST{count + 1:03d}"


def insert_prediction(student_id, cgpa, projects, internships, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
            (student_id, cgpa, projects, internships, prediction, probability, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student_id,
            cgpa,
            projects,
            internships,
            prediction,
            probability,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_predictions():
    """Returns all prediction history rows, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT student_id, cgpa, projects, internships, prediction, probability, created_at
        FROM predictions
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def count_predictions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_history():
    """Deletes all rows from the predictions table (keeps the table itself)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
