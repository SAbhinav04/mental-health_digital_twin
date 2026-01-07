import sqlite3
import random
import time
import logging
import pandas as pd 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATABASE_NAME = 'realtime_data.db'
TABLE_NAME = 'time_series'

def create_table(conn):
    """Creates the table, ensuring the user_id and mood_log columns are added."""
    cursor = conn.cursor()
    
    # 1. Ensure the table structure is created (Core columns)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            timestamp INTEGER,
            user_id TEXT,
            steps REAL,
            sleep_duration REAL,
            audio_level REAL,
            hrv REAL,
            bp_systolic REAL,
            bp_diastolic REAL,
            mood_log TEXT DEFAULT '',
            PRIMARY KEY (timestamp, user_id)
        );
    """)
    
    conn.commit()
    logging.info(f"SQLite table '{TABLE_NAME}' schema check complete.")

def write_single_data_point(data):
    """Writes either simulated data or user-provided data."""
    
    conn = None
    
    # 1. Determine the data source and values
    if data and 'user_id' in data: 
        # Manual Input Mode (1 data point)
        logging.info(f"Writing single manual data point to SQLite for {data.get('user_id', 'unknown')}.")
        timestamp = int(time.time())
        user_id = data.get('user_id', 'user1_data_baseline')
        steps = data.get('steps', 0)
        sleep_duration = data.get('sleep_duration', 0)
        audio_level = data.get('audio_level', 0)
        hrv = data.get('hrv', 0)
        bp_systolic = data.get('bp_systolic', 0)
        bp_diastolic = data.get('bp_diastolic', 0)
        mood_log = data.get('mood_log', '') 
        data_points = [(timestamp, user_id, steps, sleep_duration, audio_level, hrv, bp_systolic, bp_diastolic, mood_log)]
        
    else:
        # Simulation Mode (10 data points for the default user)
        logging.info("Writing 10 simulated data points to SQLite for user1_data_baseline.")
        timestamp = int(time.time())
        user_id = 'user1_data_baseline' 
        data_points = []
        simulated_moods = ["Feeling fantastic and productive!", "A bit tired today.", "Extremely anxious about a deadline.", "Everything is calm.", "Low energy and feeling down."]
        
        for i in range(10):
            data_points.append((
                timestamp + (i * 60),
                user_id,
                random.randint(50, 200), random.uniform(28000, 32000), 
                random.uniform(60, 85), random.randint(30, 80), 
                random.uniform(115, 135), random.uniform(75, 85),
                random.choice(simulated_moods) 
            ))

    # 2. Write to SQLite
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        create_table(conn) 
        cursor = conn.cursor()
        
        # INSERT statement must explicitly list all 9 columns
        cursor.executemany(f"""
            INSERT INTO {TABLE_NAME} (timestamp, user_id, steps, sleep_duration, audio_level, hrv, bp_systolic, bp_diastolic, mood_log) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data_points)

        conn.commit()
        
        if data:
            logging.info("--- Manual Data Ingestion Complete ---")
        else:
            logging.info("--- Simulation Data Ingestion Complete ---")
        
        return True

    except Exception as e:
        logging.error(f"Error during SQLite operation: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_latest_data(user_id='user1_data_baseline', limit=5):
    """Retrieves the latest data points for display on the front-end, filtered by user."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} WHERE user_id = ? ORDER BY timestamp DESC LIMIT {limit};", conn, params=(user_id,))
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.to_dict('records')
    except Exception as e:
        logging.error(f"Error retrieving latest data: {e}")
        return []
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    write_single_data_point(None)