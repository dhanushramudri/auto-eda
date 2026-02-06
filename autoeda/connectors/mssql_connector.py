"""
Connector for MSSQL databases.
"""
import pandas as pd
import pyodbc

def load_mssql(server, database, username, password, query):
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
