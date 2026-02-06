"""
Postman-style UI Component for REST API Testing
"""
import streamlit as st
import json
from typing import Dict, Any, Optional, Tuple


def render_postman_style_api_interface(key_prefix: str = "api") -> Tuple[Dict[str, Any], bool]:
    """
    Render a Postman-style API interface dialog.
    
    Args:
        key_prefix: Unique prefix for widget keys
        
    Returns:
        Tuple of (api_config, execute_clicked)
    """
    
    api_config = {}
    execute_clicked = False
    
    # Use a dialog/expander for the interface
    with st.expander("🌐 API Request Configuration", expanded=True):
        
        # Request Line
        st.markdown("### 📡 Request")
        col1, col2 = st.columns([1, 4])
        
        with col1:
            method = st.selectbox(
                "Method",
                ["GET", "POST", "PUT", "PATCH", "DELETE"],
                key=f"{key_prefix}_method"
            )
        
        with col2:
            url = st.text_input(
                "URL",
                placeholder="https://api.example.com/endpoint",
                key=f"{key_prefix}_url"
            )
        
        # Tabs for different configuration sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔐 Auth",
            "📋 Params",
            "📄 Headers",
            "📦 Body",
            "⚙️ Settings"
        ])
        
        # Authentication Tab
        with tab1:
            st.markdown("#### Authentication")
            auth_type = st.selectbox(
                "Auth Type",
                ["No Auth", "Basic Auth", "Bearer Token", "API Key"],
                key=f"{key_prefix}_auth_type"
            )
            
            auth_credentials = {}
            
            if auth_type == "Basic Auth":
                username = st.text_input("Username", key=f"{key_prefix}_auth_username")
                password = st.text_input("Password", type="password", key=f"{key_prefix}_auth_password")
                auth_credentials = {"username": username, "password": password}
                
            elif auth_type == "Bearer Token":
                token = st.text_input("Token", type="password", key=f"{key_prefix}_auth_token")
                auth_credentials = {"token": token}
                
            elif auth_type == "API Key":
                col1, col2 = st.columns(2)
                with col1:
                    key_name = st.text_input("Key Name", value="X-API-Key", key=f"{key_prefix}_auth_key_name")
                    key_value = st.text_input("Key Value", type="password", key=f"{key_prefix}_auth_key_value")
                with col2:
                    key_location = st.selectbox("Add to", ["Header", "Query Params"], key=f"{key_prefix}_auth_location")
                
                auth_credentials = {
                    "key_name": key_name,
                    "key_value": key_value,
                    "location": key_location.lower().replace(" ", "_")
                }
        
        # Query Parameters Tab
        with tab2:
            st.markdown("#### Query Parameters")
            
            if f"{key_prefix}_params" not in st.session_state:
                st.session_state[f"{key_prefix}_params"] = []
            
            # Add parameter button
            if st.button("➕ Add Parameter", key=f"{key_prefix}_add_param"):
                st.session_state[f"{key_prefix}_params"].append({"key": "", "value": "", "enabled": True})
            
            params = {}
            params_to_remove = []
            
            for idx, param in enumerate(st.session_state[f"{key_prefix}_params"]):
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
                
                with col1:
                    key = st.text_input(
                        "Key",
                        value=param.get("key", ""),
                        key=f"{key_prefix}_param_key_{idx}",
                        label_visibility="collapsed",
                        placeholder="Parameter name"
                    )
                
                with col2:
                    value = st.text_input(
                        "Value",
                        value=param.get("value", ""),
                        key=f"{key_prefix}_param_value_{idx}",
                        label_visibility="collapsed",
                        placeholder="Parameter value"
                    )
                
                with col3:
                    enabled = st.checkbox(
                        "✓",
                        value=param.get("enabled", True),
                        key=f"{key_prefix}_param_enabled_{idx}",
                        label_visibility="collapsed"
                    )
                
                with col4:
                    if st.button("🗑️", key=f"{key_prefix}_param_delete_{idx}"):
                        params_to_remove.append(idx)
                
                # Update session state
                st.session_state[f"{key_prefix}_params"][idx] = {
                    "key": key,
                    "value": value,
                    "enabled": enabled
                }
                
                # Add to params dict if enabled
                if enabled and key:
                    params[key] = value
            
            # Remove deleted params
            for idx in reversed(params_to_remove):
                st.session_state[f"{key_prefix}_params"].pop(idx)
        
        # Headers Tab
        with tab3:
            st.markdown("#### Headers")
            
            if f"{key_prefix}_headers" not in st.session_state:
                st.session_state[f"{key_prefix}_headers"] = [
                    {"key": "Content-Type", "value": "application/json", "enabled": True}
                ]
            
            # Add header button
            if st.button("➕ Add Header", key=f"{key_prefix}_add_header"):
                st.session_state[f"{key_prefix}_headers"].append({"key": "", "value": "", "enabled": True})
            
            headers = {}
            headers_to_remove = []
            
            for idx, header in enumerate(st.session_state[f"{key_prefix}_headers"]):
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
                
                with col1:
                    key = st.text_input(
                        "Header Key",
                        value=header.get("key", ""),
                        key=f"{key_prefix}_header_key_{idx}",
                        label_visibility="collapsed",
                        placeholder="Header name"
                    )
                
                with col2:
                    value = st.text_input(
                        "Header Value",
                        value=header.get("value", ""),
                        key=f"{key_prefix}_header_value_{idx}",
                        label_visibility="collapsed",
                        placeholder="Header value"
                    )
                
                with col3:
                    enabled = st.checkbox(
                        "✓",
                        value=header.get("enabled", True),
                        key=f"{key_prefix}_header_enabled_{idx}",
                        label_visibility="collapsed"
                    )
                
                with col4:
                    if st.button("🗑️", key=f"{key_prefix}_header_delete_{idx}"):
                        headers_to_remove.append(idx)
                
                # Update session state
                st.session_state[f"{key_prefix}_headers"][idx] = {
                    "key": key,
                    "value": value,
                    "enabled": enabled
                }
                
                # Add to headers dict if enabled
                if enabled and key:
                    headers[key] = value
            
            # Remove deleted headers
            for idx in reversed(headers_to_remove):
                st.session_state[f"{key_prefix}_headers"].pop(idx)
        
        # Body Tab
        with tab4:
            st.markdown("#### Request Body")
            
            if method in ["POST", "PUT", "PATCH"]:
                body_type = st.radio(
                    "Body Type",
                    ["JSON", "Raw Text", "Form Data"],
                    horizontal=True,
                    key=f"{key_prefix}_body_type"
                )
                
                body = None
                
                if body_type == "JSON":
                    body_text = st.text_area(
                        "JSON Body",
                        height=200,
                        placeholder='{\n  "key": "value"\n}',
                        key=f"{key_prefix}_body_json"
                    )
                    
                    if body_text:
                        try:
                            body = json.loads(body_text)
                            st.success("✅ Valid JSON")
                        except json.JSONDecodeError as e:
                            st.error(f"❌ Invalid JSON: {str(e)}")
                            body = None
                
                elif body_type == "Raw Text":
                    body_text = st.text_area(
                        "Raw Body",
                        height=200,
                        key=f"{key_prefix}_body_raw"
                    )
                    body = {"raw_text": body_text}
                
                else:  # Form Data
                    st.info("Use the Params tab for form data in GET requests, or add key-value pairs here")
                    if f"{key_prefix}_form_data" not in st.session_state:
                        st.session_state[f"{key_prefix}_form_data"] = []
                    
                    if st.button("➕ Add Form Field", key=f"{key_prefix}_add_form"):
                        st.session_state[f"{key_prefix}_form_data"].append({"key": "", "value": ""})
                    
                    form_data = {}
                    for idx, field in enumerate(st.session_state[f"{key_prefix}_form_data"]):
                        col1, col2, col3 = st.columns([3, 3, 1])
                        with col1:
                            key = st.text_input("Field", value=field.get("key", ""), key=f"{key_prefix}_form_key_{idx}")
                        with col2:
                            value = st.text_input("Value", value=field.get("value", ""), key=f"{key_prefix}_form_value_{idx}")
                        with col3:
                            if st.button("🗑️", key=f"{key_prefix}_form_delete_{idx}"):
                                st.session_state[f"{key_prefix}_form_data"].pop(idx)
                                st.rerun()
                        
                        if key:
                            form_data[key] = value
                    
                    body = form_data if form_data else None
            else:
                st.info(f"{method} requests typically don't have a body")
                body = None
        
        # Settings Tab
        with tab5:
            st.markdown("#### Request Settings")
            
            timeout = st.slider(
                "Timeout (seconds)",
                min_value=5,
                max_value=120,
                value=30,
                key=f"{key_prefix}_timeout"
            )
            
            follow_redirects = st.checkbox(
                "Follow Redirects",
                value=True,
                key=f"{key_prefix}_follow_redirects"
            )
            
            verify_ssl = st.checkbox(
                "Verify SSL Certificate",
                value=True,
                key=f"{key_prefix}_verify_ssl"
            )
        
        # Action Buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            execute_clicked = st.button(
                "🚀 Send Request",
                type="primary",
                use_container_width=True,
                key=f"{key_prefix}_send"
            )
        
        with col2:
            test_clicked = st.button(
                "🔍 Test Connection",
                use_container_width=True,
                key=f"{key_prefix}_test"
            )
        
        with col3:
            if st.button("💾 Save Config", use_container_width=True, key=f"{key_prefix}_save"):
                st.info("Configuration saved to session")
        
        with col4:
            if st.button("🗑️ Clear All", use_container_width=True, key=f"{key_prefix}_clear"):
                # Clear all session state for this component
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith(key_prefix)]
                for k in keys_to_clear:
                    del st.session_state[k]
                st.rerun()
    
    # Compile API configuration
    api_config = {
        "url": url,
        "method": method,
        "params": params if params else None,
        "headers": headers if headers else None,
        "body": body,
        "auth_type": None if auth_type == "No Auth" else auth_type.lower().replace(" ", "_"),
        "auth_credentials": auth_credentials if auth_type != "No Auth" else None,
        "timeout": timeout
    }
    resolved_url = url
    resolved_params = params.copy() if params else {}

    if params:
        for k, v in params.items():
            placeholder = f"{{{{{k}}}}}"
            if placeholder in resolved_url:
                resolved_url = resolved_url.replace(placeholder, str(v))
                resolved_params.pop(k, None)

    api_config["url"] = resolved_url
    api_config["params"] = resolved_params if resolved_params else None
    
    print(
        "api_config",api_config
    )

    
    return api_config, execute_clicked or test_clicked, test_clicked
