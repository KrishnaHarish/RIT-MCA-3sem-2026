import sqlite3
import datetime

DB_NAME = 'waste_management.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table for sensor data
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    distance REAL,
                    fill_percentage REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def log_data(distance, fill_percentage):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO sensor_data (distance, fill_percentage) VALUES (?, ?)", 
              (distance, fill_percentage))
    conn.commit()
    conn.close()

def get_recent_data(limit=10):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_data():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM sensor_data ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]
