"""
Enterprise-grade REST API Connector
Supports GET, POST, PUT, DELETE methods with authentication, headers, and parameters
"""
import pandas as pd
import requests
import json
from typing import Dict, Optional, Any, Tuple
import streamlit as st


def load_rest_api(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    auth_type: Optional[str] = None,
    auth_credentials: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load data from REST API with comprehensive configuration options.
    
    Args:
        url: API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        params: Query parameters
        headers: Request headers
        body: Request body (for POST/PUT/PATCH)
        auth_type: Authentication type (None, 'basic', 'bearer', 'api_key')
        auth_credentials: Authentication credentials
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (DataFrame, response_metadata)
    """
    
    # Prepare headers
    if headers is None:
        headers = {}
    
    # Handle authentication
    auth = None
    if auth_type == 'basic' and auth_credentials:
        auth = (auth_credentials.get('username', ''), auth_credentials.get('password', ''))
    elif auth_type == 'bearer' and auth_credentials:
        headers['Authorization'] = f"Bearer {auth_credentials.get('token', '')}"
    elif auth_type == 'api_key' and auth_credentials:
        key_location = auth_credentials.get('location', 'header')
        key_name = auth_credentials.get('key_name', 'X-API-Key')
        key_value = auth_credentials.get('key_value', '')
        
        if key_location == 'header':
            headers[key_name] = key_value
        elif key_location == 'query':
            if params is None:
                params = {}
            params[key_name] = key_value
    
    # Make request based on method
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, headers=headers, auth=auth, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, params=params, headers=headers, json=body, auth=auth, timeout=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, params=params, headers=headers, json=body, auth=auth, timeout=timeout)
        elif method.upper() == "PATCH":
            response = requests.patch(url, params=params, headers=headers, json=body, auth=auth, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, params=params, headers=headers, auth=auth, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        
        # Extract response metadata
        metadata = {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'elapsed_time': response.elapsed.total_seconds(),
            'url': response.url,
            'method': method.upper()
        }
        
        # Parse response data
        try:
            data = response.json()
        except json.JSONDecodeError:
            # If not JSON, try to parse as text
            raise ValueError("Response is not valid JSON. Use a different parser for non-JSON responses.")
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Check if dict contains a list under a key
            for key, value in data.items():
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    metadata['data_key'] = key
                    break
            else:
                # Single record
                df = pd.DataFrame([data])
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        return df, metadata
        
    except requests.exceptions.Timeout:
        raise Exception(f"Request timed out after {timeout} seconds")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to {url}")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP Error {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"API request failed: {str(e)}")


def test_api_connection(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    auth_type: Optional[str] = None,
    auth_credentials: Optional[Dict[str, str]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Test API connection without loading full data.
    
    Returns:
        Tuple of (success, message, metadata)
    """
    try:
        # Make a lightweight request
        auth = None
        if auth_type == 'basic' and auth_credentials:
            auth = (auth_credentials.get('username', ''), auth_credentials.get('password', ''))
        elif auth_type == 'bearer' and auth_credentials:
            if headers is None:
                headers = {}
            headers['Authorization'] = f"Bearer {auth_credentials.get('token', '')}"
        
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, auth=auth, timeout=10)
        else:
            response = requests.request(method, url, headers=headers, auth=auth, timeout=10)
        
        response.raise_for_status()
        
        metadata = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'content_type': response.headers.get('content-type', 'unknown'),
            'success': True
        }
        
        return True, f"✅ Connection successful! Status: {response.status_code}", metadata
        
    except requests.exceptions.Timeout:
        return False, "❌ Connection timed out", {}
    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot connect to {url}", {}
    except requests.exceptions.HTTPError as e:
        return False, f"❌ HTTP Error {response.status_code}", {'status_code': response.status_code}
    except Exception as e:
        return False, f"❌ Error: {str(e)}", {}
