import sqlite3
import time
import random
import pandas as pd
import logging


DB_PATH = 'realtime_data.db'



def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Creates the daily_logs table if it doesn't exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                mood_log TEXT,
                steps REAL,
                sleep_duration REAL,
                audio_level REAL,
                hrv REAL,
                bp_systolic REAL,
                bp_diastolic REAL
            )
        ''')
        conn.commit()
        logging.info(f"Database '{DB_PATH}' initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
    finally:
        conn.close()

def write_single_data_point(data):
    """Writes a single, potentially simulated, data point to the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
       
        full_data = {
            'user_id': data.get('user_id', 'user1_data_baseline'),
            'timestamp': data.get('timestamp', int(time.time())),
            'mood_log': data.get('mood_log', random.choice(["Feeling good today.", "A bit stressed.", "Productive morning.", "Tired."])),
            'steps': data.get('steps', random.randint(100, 500)),
            'sleep_duration': data.get('sleep_duration', random.uniform(25000, 30000)),
            'audio_level': data.get('audio_level', random.uniform(65, 80)),
            'hrv': data.get('hrv', random.randint(40, 70)),
            'bp_systolic': data.get('bp_systolic', random.uniform(118, 125)),
            'bp_diastolic': data.get('bp_diastolic', random.uniform(75, 82))
        }

        cursor.execute('''
            INSERT INTO daily_logs (user_id, timestamp, mood_log, steps, sleep_duration, audio_level, hrv, bp_systolic, bp_diastolic)
            VALUES (:user_id, :timestamp, :mood_log, :steps, :sleep_duration, :audio_level, :hrv, :bp_systolic, :bp_diastolic)
        ''', full_data)
        
        conn.commit()
        logging.info(f"Successfully wrote data point for user '{full_data['user_id']}'.")
        return True
    except sqlite3.Error as e:
        logging.error(f"Database write error: {e}")
        return False
    finally:
        conn.close()

def get_latest_data(user_id, limit=5):
    """Retrieves the last N data points for a specific user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM daily_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
       
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logging.error(f"Database read error in get_latest_data: {e}")
        return []
    finally:
        conn.close()

def get_all_user_data_as_df(user_id):
    """
    Connects to the SQLite DB and fetches all historical data for a specific user,
    returning it as a Pandas DataFrame.
    """
    conn = get_db_connection()
    try:
        
        query = "SELECT * FROM daily_logs WHERE user_id = ?"
        df = pd.read_sql_query(query, conn, params=(user_id,))
        
        if df.empty:
            logging.warning(f"No data found for user '{user_id}' in get_all_user_data_as_df.")
            return None
            
       
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values(by='timestamp')
        
        
        numeric_cols = ['steps', 'sleep_duration', 'audio_level', 'hrv', 'bp_systolic', 'bp_diastolic']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logging.info(f"Successfully fetched {len(df)} records for user '{user_id}'.")
        return df

    except Exception as e:
        logging.error(f"Pandas/DB error in get_all_user_data_as_df: {e}")
        return None
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    print("Initializing database...")
    initialize_database()
    print("Simulating one test data point...")
    write_single_data_point({'user_id': 'user1_data_baseline'})
    print("Database setup complete.")