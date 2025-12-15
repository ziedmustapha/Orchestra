# API Authentication Guide

> Orchestra v3 Authentication System

## Overview

Orchestra supports API key authentication for public access while keeping local access open. This allows you to:

- **Local Access**: No authentication required for requests from localhost/127.0.0.1
- **External Access**: API key required for all non-local requests
- **Flexible Configuration**: Easy management of API keys and authentication settings

## Configuration

The authentication system is configured via `api_keys.json` (or set `API_KEYS_FILE` to an absolute/relative path to override):

```json
{
  "valid_keys": [
    "demo-key-12345",
    "prod-key-67890",
    "iszcx8YNM7xqyo_lj-_iD8vMODMfW7xc"
  ],
  "require_auth_for_external": true,
  "local_networks": [
    "127.0.0.1",
    "localhost",
    "::1"
  ]
}
```

### Configuration Options

- **`valid_keys`**: Array of valid API keys
- **`require_auth_for_external`**: Enable/disable authentication for external requests
- **`local_networks`**: IP addresses/networks that don't require authentication

## API Key Management

Use the `src/manage_api_keys.py` script to manage your API keys:

### List all API keys
```bash
python src/manage_api_keys.py list
```

### Add a new API key
```bash
python src/manage_api_keys.py add "your-custom-key-here"
```

### Remove an API key
```bash
python src/manage_api_keys.py remove "key-to-remove"
```

### Generate a secure random API key
```bash
python src/manage_api_keys.py generate
```

### Toggle authentication requirement
```bash
python src/manage_api_keys.py toggle-auth
```

## Usage Examples

### Local Access (No API Key Required)
```bash
curl -X POST http://127.0.0.1:9001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "qwen3",
    "request_body": {
      "input": "Your input text here",
      "max_new_tokens": 1000
    }
  }'
```

### External Access (API Key Required)
```bash
curl -X POST http://your-server-ip:9001/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-12345" \
  -d '{
    "model_name": "qwen3",
    "request_body": {
      "input": "Your input text here",
      "max_new_tokens": 1000
    }
  }'
```

### Using JSON Files for Complex Requests
```bash
# Create your request in a JSON file
echo '{
  "model_name": "qwen3",
  "request_body": {
    "input": "Complex input with special characters...",
    "max_new_tokens": 1000
  }
}' > my_request.json

# Send the request with API key
curl -X POST http://your-server-ip:9001/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d @my_request.json
```

## Error Responses

### Missing API Key (External Request)
```json
{
  "detail": "Invalid or missing API key. Please provide a valid X-API-Key header."
}
```
**HTTP Status**: 401 Unauthorized

### Invalid API Key
```json
{
  "detail": "Invalid or missing API key. Please provide a valid X-API-Key header."
}
```
**HTTP Status**: 401 Unauthorized

## Security Best Practices

1. **Keep API Keys Secure**: Never commit API keys to version control
2. **Use Strong Keys**: Generate random, long API keys (32+ characters)
3. **Rotate Keys Regularly**: Remove old keys and generate new ones periodically
4. **Monitor Access**: Check logs for unauthorized access attempts
5. **Network Security**: Use HTTPS in production and consider firewall rules

## Server Configuration

By default, the load balancer binds to `127.0.0.1:9001` (local only). To accept external connections, set:

```bash
export LOAD_BALANCER_HOST=0.0.0.0
```

Make sure your firewall allows traffic on port 9001 if you want external access.

### Starting the Server
```bash
./scripts/run_api.sh 1 4 1 start  # 1 Gemma3, 4 Qwen, 1 Qwen3 workers
```

### Stopping the Server
```bash
./scripts/run_api.sh stop
```

## Troubleshooting

### Authentication Not Working
1. Check that `api_keys.json` exists and is valid JSON
2. Verify the API key is in the `valid_keys` array
3. Ensure `require_auth_for_external` is set to `true`
4. Check that your request IP is not in `local_networks`

### Cannot Access Externally
1. Verify the server is listening on `0.0.0.0:9001`
2. Check firewall settings
3. Ensure the load balancer process is running

### API Key Management Issues
1. Run via Python: `python src/manage_api_keys.py`
2. Verify the script can read/write `api_keys.json`

## Integration Examples

### Python Client
```python
import requests

# Local request (no API key needed)
response = requests.post(
    "http://127.0.0.1:9001/infer",
    json={
        "model_name": "qwen3",
        "request_body": {
            "input": "Your input here",
            "max_new_tokens": 1000
        }
    }
)

# External request (API key required)
response = requests.post(
    "http://your-server-ip:9001/infer",
    headers={"X-API-Key": "your-api-key-here"},
    json={
        "model_name": "qwen3",
        "request_body": {
            "input": "Your input here",
            "max_new_tokens": 1000
        }
    }
)
```

### JavaScript/Node.js Client
```javascript
const axios = require('axios');

// External request with API key
const response = await axios.post('http://your-server-ip:9001/infer', {
  model_name: 'qwen3',
  request_body: {
    input: 'Your input here',
    max_new_tokens: 1000
  }
}, {
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key-here'
  }
});
```
