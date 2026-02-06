"""
Connector for PostgreSQL databases.
"""
import pandas as pd
import psycopg2

def load_postgres(host, user, password, db, query):
    conn = psycopg2.connect(host=host, user=user, password=password, dbname=db)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
