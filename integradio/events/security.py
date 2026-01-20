"""
Security utilities for Semantic Events.

Implements 2026 best practices:
- HMAC-SHA256 message signing (RFC 2104)
- Constant-time signature verification
- Token bucket rate limiting
- Connection management with limits
- Origin validation
- Nonce tracking for replay protection
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Set
from urllib.parse import urlparse

from .event import SemanticEvent


class EventSigner:
    """
    HMAC-SHA256 message signer for event integrity.

    Security features:
    - Uses SHA-256 (not SHA-1)
    - Constant-time comparison
    - Canonical JSON serialization
    - Timestamp included in signature
    """

    def __init__(self, secret_key: str | bytes):
        """
        Initialize signer with secret key.

        Args:
            secret_key: Secret key for HMAC (min 32 bytes recommended)
        """
        if isinstance(secret_key, str):
            secret_key = secret_key.encode("utf-8")

        if len(secret_key) < 32:
            import warnings
            warnings.warn(
                "Secret key should be at least 32 bytes for security",
                UserWarning,
                stacklevel=2,
            )

        self._key = secret_key

    def sign(self, event: SemanticEvent) -> SemanticEvent:
        """
        Sign an event with HMAC-SHA256.

        Args:
            event: Event to sign

        Returns:
            Event with signature field set
        """
        # Create canonical message for signing
        # Exclude signature field itself
        signing_data = {
            "type": event.type,
            "source": event.source,
            "id": event.id,
            "time": event.time,
            "nonce": event.nonce,
            "data": event.data,
        }

        # Canonical JSON: sorted keys, no whitespace
        message = json.dumps(
            signing_data, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

        # Generate HMAC-SHA256
        signature = hmac.new(self._key, message, hashlib.sha256).hexdigest()

        event.signature = signature
        return event

    def verify(self, event: SemanticEvent) -> bool:
        """
        Verify event signature.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            event: Event to verify

        Returns:
            True if signature is valid
        """
        if not event.signature:
            return False

        # Recreate signing data
        signing_data = {
            "type": event.type,
            "source": event.source,
            "id": event.id,
            "time": event.time,
            "nonce": event.nonce,
            "data": event.data,
        }

        message = json.dumps(
            signing_data, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

        expected = hmac.new(self._key, message, hashlib.sha256).hexdigest()

        # Constant-time comparison (prevents timing attacks)
        return hmac.compare_digest(event.signature, expected)

    @staticmethod
    def generate_key(length: int = 32) -> str:
        """
        Generate a cryptographically secure secret key.

        Args:
            length: Key length in bytes

        Returns:
            Hex-encoded secret key
        """
        return secrets.token_hex(length)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


class RateLimiter:
    """
    Token bucket rate limiter for WebSocket connections.

    Implements the token bucket algorithm:
    - Bucket fills at constant rate
    - Burst allowed up to bucket capacity
    - Smooth rate limiting without hard cutoffs

    Security features:
    - Per-client tracking
    - Configurable burst and sustained rates
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        rate: float = 100.0,  # tokens per second
        burst: int = 200,  # max bucket size
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Token refill rate (requests per second)
            burst: Maximum bucket capacity (allows bursts)
            cleanup_interval: How often to clean old entries
        """
        self.rate = rate
        self.burst = burst
        self.cleanup_interval = cleanup_interval

        # Per-client buckets: {client_id: (tokens, last_update)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def check(self, client_id: str, cost: int = 1) -> RateLimitResult:
        """
        Check if request is allowed under rate limit.

        Args:
            client_id: Unique client identifier
            cost: Token cost of this request (default 1)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        async with self._lock:
            now = time.monotonic()

            # Cleanup old entries periodically
            if now - self._last_cleanup > self.cleanup_interval:
                await self._cleanup(now)
                self._last_cleanup = now

            # Get or create bucket
            if client_id in self._buckets:
                tokens, last_update = self._buckets[client_id]
            else:
                tokens, last_update = self.burst, now

            # Refill tokens based on elapsed time
            elapsed = now - last_update
            tokens = min(self.burst, tokens + elapsed * self.rate)

            # Check if enough tokens
            if tokens >= cost:
                tokens -= cost
                self._buckets[client_id] = (tokens, now)
                return RateLimitResult(
                    allowed=True,
                    remaining=int(tokens),
                    reset_at=now + (self.burst - tokens) / self.rate,
                )
            else:
                # Calculate when enough tokens will be available
                wait_time = (cost - tokens) / self.rate
                self._buckets[client_id] = (tokens, now)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                )

    async def _cleanup(self, now: float) -> None:
        """Remove stale bucket entries."""
        stale_threshold = now - self.cleanup_interval * 2
        to_remove = [
            cid
            for cid, (_, last_update) in self._buckets.items()
            if last_update < stale_threshold
        ]
        for cid in to_remove:
            del self._buckets[cid]

    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        if client_id in self._buckets:
            del self._buckets[client_id]


class NonceTracker:
    """
    Track used nonces to prevent replay attacks.

    Features:
    - Time-windowed storage (only keep recent nonces)
    - Memory-efficient with automatic cleanup
    """

    def __init__(self, window_seconds: int = 300):
        """
        Initialize nonce tracker.

        Args:
            window_seconds: How long to remember nonces
        """
        self.window = window_seconds
        self._nonces: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check_and_add(self, nonce: str) -> bool:
        """
        Check if nonce is fresh and add it.

        Args:
            nonce: Nonce to check

        Returns:
            True if nonce is fresh (not seen before)
        """
        async with self._lock:
            now = time.time()

            # Cleanup old nonces
            cutoff = now - self.window
            self._nonces = {
                n: t for n, t in self._nonces.items() if t > cutoff
            }

            # Check if seen
            if nonce in self._nonces:
                return False

            # Add nonce
            self._nonces[nonce] = now
            return True


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""

    client_id: str
    connected_at: float = field(default_factory=time.time)
    origin: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    message_count: int = 0
    last_activity: float = field(default_factory=time.time)


class ConnectionManager:
    """
    Manage WebSocket connections with security controls.

    Features:
    - Per-IP connection limits
    - Maximum total connections
    - Connection timeout management
    - Activity tracking
    """

    def __init__(
        self,
        max_connections: int = 10000,
        max_per_ip: int = 100,
        idle_timeout: float = 300.0,
    ):
        """
        Initialize connection manager.

        Args:
            max_connections: Global connection limit
            max_per_ip: Per-IP connection limit
            idle_timeout: Disconnect idle connections after this time
        """
        self.max_connections = max_connections
        self.max_per_ip = max_per_ip
        self.idle_timeout = idle_timeout

        self._connections: dict[str, ConnectionInfo] = {}
        self._by_ip: dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def can_connect(self, ip_address: str) -> tuple[bool, str]:
        """
        Check if a new connection is allowed.

        Args:
            ip_address: Client IP address

        Returns:
            (allowed, reason) tuple
        """
        async with self._lock:
            # Check global limit
            if len(self._connections) >= self.max_connections:
                return False, "Server at maximum capacity"

            # Check per-IP limit
            if len(self._by_ip[ip_address]) >= self.max_per_ip:
                return False, "Too many connections from this IP"

            return True, "OK"

    async def add(
        self,
        client_id: str,
        ip_address: str,
        origin: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ConnectionInfo:
        """
        Register a new connection.

        Args:
            client_id: Unique connection identifier
            ip_address: Client IP
            origin: Origin header value
            user_id: Authenticated user ID

        Returns:
            ConnectionInfo for the new connection
        """
        async with self._lock:
            info = ConnectionInfo(
                client_id=client_id,
                origin=origin,
                user_id=user_id,
                ip_address=ip_address,
            )
            self._connections[client_id] = info
            self._by_ip[ip_address].add(client_id)
            return info

    async def remove(self, client_id: str) -> Optional[ConnectionInfo]:
        """
        Remove a connection.

        Args:
            client_id: Connection to remove

        Returns:
            ConnectionInfo if found, None otherwise
        """
        async with self._lock:
            info = self._connections.pop(client_id, None)
            if info and info.ip_address:
                self._by_ip[info.ip_address].discard(client_id)
                if not self._by_ip[info.ip_address]:
                    del self._by_ip[info.ip_address]
            return info

    async def get(self, client_id: str) -> Optional[ConnectionInfo]:
        """Get connection info."""
        return self._connections.get(client_id)

    async def update_activity(self, client_id: str) -> None:
        """Update last activity timestamp."""
        if client_id in self._connections:
            self._connections[client_id].last_activity = time.time()
            self._connections[client_id].message_count += 1

    async def get_idle_connections(self) -> list[str]:
        """Get list of idle connection IDs."""
        now = time.time()
        cutoff = now - self.idle_timeout
        return [
            cid
            for cid, info in self._connections.items()
            if info.last_activity < cutoff
        ]

    async def get_all(self) -> list[ConnectionInfo]:
        """Get all connections."""
        return list(self._connections.values())

    @property
    def connection_count(self) -> int:
        """Current number of connections."""
        return len(self._connections)


def validate_origin(
    origin: Optional[str],
    allowed_origins: list[str],
    allow_none: bool = False,
) -> bool:
    """
    Validate WebSocket Origin header.

    Args:
        origin: Origin header value
        allowed_origins: List of allowed origins (supports wildcards)
        allow_none: Whether to allow missing origin (non-browser clients)

    Returns:
        True if origin is valid
    """
    if origin is None:
        return allow_none

    # Parse origin
    try:
        parsed = urlparse(origin)
        origin_host = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return False

    for allowed in allowed_origins:
        # Exact match
        if allowed == origin_host:
            return True

        # Wildcard subdomain: *.example.com
        if allowed.startswith("*."):
            domain = allowed[2:]
            if parsed.netloc == domain or parsed.netloc.endswith("." + domain):
                return True

        # Allow all (not recommended for production)
        if allowed == "*":
            return True

    return False


def validate_message_size(data: bytes | str, max_size: int = 65536) -> bool:
    """
    Validate message size.

    Args:
        data: Message data
        max_size: Maximum size in bytes (default 64KB)

    Returns:
        True if size is within limit
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return len(data) <= max_size
