# Semantic Events Security Guide

## Overview

The Semantic Events module implements 2026 best practices for secure real-time WebSocket communication.

## Security Features

### 1. Message Signing (HMAC-SHA256)

All events are signed using HMAC-SHA256 to ensure:
- **Integrity**: Messages haven't been tampered with
- **Authenticity**: Messages came from a trusted source

```python
from integradio.events import EventSigner, SemanticEvent

# Generate a secure key (do this once, store securely)
secret_key = EventSigner.generate_key()  # 32 bytes

# Sign events
signer = EventSigner(secret_key)
event = SemanticEvent(type="ui.click", source="app", data={"id": 1})
signer.sign(event)

# Verify on receipt
if not signer.verify(event):
    raise SecurityError("Invalid signature")
```

**Best practices:**
- Use at least 32-byte keys
- Store keys in environment variables or secret managers
- Rotate keys periodically

### 2. Replay Protection

Events include:
- **Nonce**: Random value unique to each event
- **Timestamp**: ISO 8601 timestamp

```python
# Events expire after 5 minutes by default
if event.is_expired(max_age_seconds=300):
    raise SecurityError("Event expired")

# Nonce tracking prevents replays
nonce_tracker = NonceTracker(window_seconds=300)
if not await nonce_tracker.check_and_add(event.nonce):
    raise SecurityError("Replay detected")
```

### 3. Rate Limiting (Token Bucket)

Prevents DoS attacks with configurable rate limiting:

```python
from integradio.events import RateLimiter

limiter = RateLimiter(
    rate=100.0,    # 100 tokens/second refill
    burst=200,     # Max 200 tokens (allows bursts)
)

result = await limiter.check(client_id)
if not result.allowed:
    # Return 429 Too Many Requests
    response.retry_after = result.retry_after
```

**Recommended settings:**
- API endpoints: 100/sec, burst 200
- WebSocket messages: 50/sec, burst 100
- Authentication attempts: 5/min, burst 10

### 4. Connection Management

```python
from integradio.events import ConnectionManager

manager = ConnectionManager(
    max_connections=10000,  # Global limit
    max_per_ip=100,         # Per-IP limit
    idle_timeout=300.0,     # Disconnect after 5 min idle
)

# Check before accepting
allowed, reason = await manager.can_connect(client_ip)
if not allowed:
    return Response(status=503, body=reason)
```

### 5. Origin Validation

```python
from integradio.events import validate_origin

allowed_origins = [
    "https://myapp.com",
    "https://*.myapp.com",  # Wildcard subdomains
]

if not validate_origin(origin_header, allowed_origins):
    return Response(status=403, body="Origin not allowed")
```

**Never use `["*"]` in production!**

### 6. Message Size Limits

```python
from integradio.events.security import validate_message_size

MAX_SIZE = 65536  # 64KB

if not validate_message_size(message, MAX_SIZE):
    return Response(status=413, body="Message too large")
```

### 7. Authentication

WebSocket authentication using the ticket pattern:

```python
# Server-side auth handler
def validate_token(token: str) -> Optional[str]:
    """Return user_id if valid, None otherwise."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]  # user_id
    except jwt.InvalidTokenError:
        return None

server = WebSocketServer(mesh, auth_handler=validate_token)
```

**Client flow:**
1. Client obtains token via HTTPS
2. Client connects to WebSocket
3. Client sends: `{"type": "auth", "token": "..."}`
4. Server validates and responds with success/failure

## Production Checklist

- [ ] Use WSS (TLS) - never WS in production
- [ ] Set specific allowed origins (no wildcards)
- [ ] Use strong secret keys (32+ bytes)
- [ ] Enable rate limiting
- [ ] Set connection limits per IP
- [ ] Implement authentication
- [ ] Validate all message contents
- [ ] Set reasonable message size limits
- [ ] Monitor for security events
- [ ] Log auth failures (but never log tokens)

## Security Event Logging

```python
import logging

# Log security events (not sensitive data)
logger.warning(f"Auth failed for IP {ip}")
logger.warning(f"Rate limit exceeded for {client_id}")
logger.warning(f"Invalid signature from {source}")

# NEVER log:
# - Tokens or secrets
# - Full message contents
# - Personal user data
```

## References

- [OWASP WebSocket Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html)
- [CloudEvents Specification](https://cloudevents.io/)
- [RFC 2104 - HMAC](https://tools.ietf.org/html/rfc2104)
