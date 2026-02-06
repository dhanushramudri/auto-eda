"""
Connector for MySQL databases.
"""
import pandas as pd
import pymysql

def load_mysql(host, user, password, db, query):
    conn = pymysql.connect(host=host, user=user, password=password, db=db)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
