#!/usr/bin/env python3
"""
API Key Management Script for AI GPU Optimization System
Usage:
  python manage_api_keys.py list                    # List all API keys
  python manage_api_keys.py add <key>               # Add a new API key
  python manage_api_keys.py remove <key>            # Remove an API key
  python manage_api_keys.py generate               # Generate a random API key
  python manage_api_keys.py toggle-auth            # Toggle external authentication requirement
"""

import json
import sys
import os
import secrets
import string
from pathlib import Path

# Resolve API keys file path with environment override and sane defaults
# 1) API_KEYS_FILE env var (absolute or relative path)
# 2) ./api_keys.json in current working directory
# 3) <repo_dir>/api_keys.json next to this file
_env_path = os.environ.get('API_KEYS_FILE')
if _env_path:
    CONFIG_FILE = Path(_env_path)
else:
    # Try CWD first
    cwd_candidate = Path(os.getcwd()) / 'api_keys.json'
    if cwd_candidate.exists():
        CONFIG_FILE = cwd_candidate
    else:
        # Fallback to repo root (assuming src/ is under repo/)
        repo_root = Path(__file__).resolve().parent.parent  # src/ -> repo/
        CONFIG_FILE = repo_root / 'api_keys.json'

def load_config():
    """Load the API configuration"""
    if not CONFIG_FILE.exists():
        return {
            "valid_keys": [],
            "require_auth_for_external": True,
            "local_networks": ["127.0.0.1", "localhost", "::1"]
        }
    
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save the API configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def generate_api_key(length=32):
    """Generate a secure random API key"""
    alphabet = string.ascii_letters + string.digits + '-_'
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def list_keys():
    """List all API keys"""
    config = load_config()
    print(f"Authentication required for external requests: {config['require_auth_for_external']}")
    print(f"Local networks (no auth required): {', '.join(config['local_networks'])}")
    print(f"\nValid API keys ({len(config['valid_keys'])}):")
    for i, key in enumerate(config['valid_keys'], 1):
        print(f"  {i}. {key}")

def add_key(key):
    """Add a new API key"""
    config = load_config()
    if key in config['valid_keys']:
        print(f"Key '{key}' already exists!")
        return
    
    config['valid_keys'].append(key)
    save_config(config)
    print(f"Added API key: {key}")

def remove_key(key):
    """Remove an API key"""
    config = load_config()
    if key not in config['valid_keys']:
        print(f"Key '{key}' not found!")
        return
    
    config['valid_keys'].remove(key)
    save_config(config)
    print(f"Removed API key: {key}")

def generate_key():
    """Generate and add a new API key"""
    key = generate_api_key()
    add_key(key)
    return key

def toggle_auth():
    """Toggle external authentication requirement"""
    config = load_config()
    config['require_auth_for_external'] = not config['require_auth_for_external']
    save_config(config)
    status = "enabled" if config['require_auth_for_external'] else "disabled"
    print(f"External authentication {status}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_keys()
    elif command == 'add' and len(sys.argv) == 3:
        add_key(sys.argv[2])
    elif command == 'remove' and len(sys.argv) == 3:
        remove_key(sys.argv[2])
    elif command == 'generate':
        key = generate_key()
        print(f"Generated new API key: {key}")
    elif command == 'toggle-auth':
        toggle_auth()
    else:
        print(__doc__)

if __name__ == '__main__':
    main()
