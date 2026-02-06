"""
Connector for CSV and Excel files.
"""
import pandas as pd

def load_csv(filepath):
    return pd.read_csv(filepath)

def load_excel(filepath):
    return pd.read_excel(filepath)
