"""
Connector for REST APIs.
"""
import pandas as pd
import requests

def load_rest_api(url, params=None, headers=None):
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data)
