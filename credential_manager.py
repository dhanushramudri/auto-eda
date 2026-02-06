"""
Secure Credential Storage Utility
Encrypts and stores database/API credentials securely
"""
import streamlit as st
import json
import base64
from cryptography.fernet import Fernet
from typing import Dict, Any, Optional
import os


class CredentialManager:
    """Manages secure storage of credentials"""
    
    def __init__(self):
        # Initialize encryption key (in production, use environment variable or secure key management)
        if 'encryption_key' not in st.session_state:
            st.session_state.encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(st.session_state.encryption_key)
        
        # Initialize credential storage in session state
        if 'stored_credentials' not in st.session_state:
            st.session_state.stored_credentials = {}
    
    def save_credentials(self, name: str, credentials: Dict[str, Any]) -> bool:
        """
        Save credentials with encryption.
        
        Args:
            name: Credential profile name
            credentials: Dictionary of credential data
            
        Returns:
            True if saved successfully
        """
        try:
            # Convert to JSON and encrypt
            cred_json = json.dumps(credentials)
            encrypted = self.cipher.encrypt(cred_json.encode())
            
            # Store encrypted credentials
            st.session_state.stored_credentials[name] = base64.b64encode(encrypted).decode()
            return True
        except Exception as e:
            st.error(f"Error saving credentials: {str(e)}")
            return False
    
    def load_credentials(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load and decrypt credentials.
        
        Args:
            name: Credential profile name
            
        Returns:
            Dictionary of credentials or None if not found
        """
        try:
            if name not in st.session_state.stored_credentials:
                return None
            
            # Decrypt credentials
            encrypted = base64.b64decode(st.session_state.stored_credentials[name])
            decrypted = self.cipher.decrypt(encrypted)
            credentials = json.loads(decrypted.decode())
            
            return credentials
        except Exception as e:
            st.error(f"Error loading credentials: {str(e)}")
            return None
    
    def list_credentials(self) -> list:
        """List all saved credential profiles"""
        return list(st.session_state.stored_credentials.keys())
    
    def delete_credentials(self, name: str) -> bool:
        """Delete a credential profile"""
        try:
            if name in st.session_state.stored_credentials:
                del st.session_state.stored_credentials[name]
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting credentials: {str(e)}")
            return False
    
    def export_credentials(self) -> str:
        """
        Export all credentials as encrypted string (for backup).
        
        Returns:
            Base64 encoded encrypted credentials
        """
        try:
            creds_json = json.dumps(st.session_state.stored_credentials)
            encrypted = self.cipher.encrypt(creds_json.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            st.error(f"Error exporting credentials: {str(e)}")
            return ""
    
    def import_credentials(self, encrypted_data: str) -> bool:
        """
        Import credentials from encrypted string.
        
        Args:
            encrypted_data: Base64 encoded encrypted credentials
            
        Returns:
            True if imported successfully
        """
        try:
            encrypted = base64.b64decode(encrypted_data)
            decrypted = self.cipher.decrypt(encrypted)
            credentials = json.loads(decrypted.decode())
            
            st.session_state.stored_credentials.update(credentials)
            return True
        except Exception as e:
            st.error(f"Error importing credentials: {str(e)}")
            return False


def render_credential_manager_ui(credential_type: str = "database"):
    """
    Render UI for managing saved credentials.
    
    Args:
        credential_type: Type of credentials (database, api, cloud)
    """
    manager = CredentialManager()
    
    st.markdown("### 🔐 Credential Manager")
    
    tab1, tab2 = st.tabs(["💾 Saved Profiles", "➕ New Profile"])
    
    with tab1:
        saved_profiles = manager.list_credentials()
        
        if saved_profiles:
            selected_profile = st.selectbox(
                "Select saved profile",
                saved_profiles,
                key=f"{credential_type}_profile_select"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("📥 Load Profile", key=f"{credential_type}_load", use_container_width=True):
                    creds = manager.load_credentials(selected_profile)
                    if creds:
                        st.session_state[f'{credential_type}_loaded_creds'] = creds
                        st.success(f"✅ Loaded profile: {selected_profile}")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", key=f"{credential_type}_delete", use_container_width=True):
                    if manager.delete_credentials(selected_profile):
                        st.success(f"Deleted: {selected_profile}")
                        st.rerun()
            
            with col3:
                # Export button
                export_data = manager.export_credentials()
                st.download_button(
                    "📤 Export All",
                    export_data,
                    file_name="credentials_backup.enc",
                    key=f"{credential_type}_export",
                    use_container_width=True
                )
        else:
            st.info("No saved profiles. Create one in the 'New Profile' tab.")
    
    with tab2:
        profile_name = st.text_input(
            "Profile Name",
            placeholder="My Database Connection",
            key=f"{credential_type}_new_profile_name"
        )
        
        st.markdown("**Enter Credentials:**")
        
        # Dynamic credential fields based on type
        new_creds = {}
        
        if credential_type == "database":
            new_creds['host'] = st.text_input("Host", key=f"{credential_type}_new_host")
            new_creds['port'] = st.text_input("Port", key=f"{credential_type}_new_port")
            new_creds['database'] = st.text_input("Database", key=f"{credential_type}_new_db")
            new_creds['username'] = st.text_input("Username", key=f"{credential_type}_new_user")
            new_creds['password'] = st.text_input("Password", type="password", key=f"{credential_type}_new_pass")
        
        elif credential_type == "api":
            new_creds['url'] = st.text_input("API URL", key=f"{credential_type}_new_url")
            new_creds['api_key'] = st.text_input("API Key", type="password", key=f"{credential_type}_new_key")
        
        elif credential_type == "cloud":
            new_creds['access_key'] = st.text_input("Access Key", key=f"{credential_type}_new_access")
            new_creds['secret_key'] = st.text_input("Secret Key", type="password", key=f"{credential_type}_new_secret")
            new_creds['region'] = st.text_input("Region", key=f"{credential_type}_new_region")
        
        if st.button("💾 Save Profile", type="primary", key=f"{credential_type}_save"):
            if profile_name and any(new_creds.values()):
                if manager.save_credentials(profile_name, new_creds):
                    st.success(f"✅ Saved profile: {profile_name}")
                    st.rerun()
            else:
                st.warning("Please enter a profile name and at least one credential")


def get_loaded_credentials(credential_type: str) -> Optional[Dict[str, Any]]:
    """
    Get currently loaded credentials from session state.
    
    Args:
        credential_type: Type of credentials
        
    Returns:
        Credentials dictionary or None
    """
    return st.session_state.get(f'{credential_type}_loaded_creds')
