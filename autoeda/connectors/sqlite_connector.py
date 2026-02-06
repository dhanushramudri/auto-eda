"""
Connector for SQLite databases.
"""
import pandas as pd
import sqlite3

def load_sqlite(db_path, query):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
