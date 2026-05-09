"""
backend/app/database.py
-----------------------
Persistencia liviana en SQLite para auditoría y monitoreo de predicciones.
"""

import os
import sqlite3
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "predictions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            text        TEXT,
            label       TEXT,
            probability REAL
        )
    """)
    conn.commit()
    return conn


def log_prediction(text: str, label: str, probability: float):
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO predictions (timestamp, text, label, probability) VALUES (?, ?, ?, ?)",
            (datetime.datetime.utcnow().isoformat(), text[:500], label, probability)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # No fallar la predicción por errores de BD


def get_stats() -> dict:
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'")
        spam_count = cur.fetchone()[0]

        cur.execute("SELECT AVG(probability) FROM predictions WHERE label='spam'")
        avg_prob_spam = cur.fetchone()[0] or 0.0

        cur.execute("""
            SELECT DATE(timestamp) as day, COUNT(*) as cnt
            FROM predictions
            GROUP BY day ORDER BY day DESC LIMIT 7
        """)
        daily = [{"date": r[0], "count": r[1]} for r in cur.fetchall()]

        conn.close()
        return {
            "total_predictions": total,
            "spam_count":        spam_count,
            "ham_count":         total - spam_count,
            "spam_rate":         round(spam_count / max(total, 1), 4),
            "avg_spam_prob":     round(avg_prob_spam, 4),
            "daily_last_7":      daily,
        }
    except Exception as e:
        return {"error": str(e)}
