"""
Security tests for integradio.

Tests for:
- File upload sanitization (path traversal prevention)
- Secret key validation
- Security headers middleware
- Gradio version validation
"""

import pytest
from pathlib import Path


class TestFilenameSanitization:
    """Tests for upload filename sanitization (CVE prevention)."""

    def test_basic_filename(self):
        """Normal filenames should pass through."""
        from integradio.pages.upload import sanitize_filename

        assert sanitize_filename("test.jpg") == "test.jpg"
        assert sanitize_filename("my_file.png") == "my_file.png"
        assert sanitize_filename("document-v2.pdf") == "document-v2.pdf"

    def test_path_traversal_unix(self):
        """Unix path traversal attempts should be blocked."""
        from integradio.pages.upload import sanitize_filename

        # These should extract just the filename part
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("../../secret.txt") == "secret.txt"
        assert sanitize_filename("/etc/passwd") == "passwd"

    def test_path_traversal_windows(self):
        """Windows path traversal attempts should be blocked."""
        from integradio.pages.upload import sanitize_filename

        # These should extract just the filename part
        assert sanitize_filename("..\\..\\..\\windows\\system32\\config") == "config"
        # Note: cmd.exe is blocked due to .exe extension (security feature)
        with pytest.raises(ValueError, match="not allowed"):
            sanitize_filename("C:\\Windows\\System32\\cmd.exe")

    def test_null_byte_injection(self):
        """Null bytes should be stripped (prevents extension bypass)."""
        from integradio.pages.upload import sanitize_filename

        # Null bytes are stripped, then if the result has a dangerous extension, it's blocked
        # This is the correct security behavior - null byte injection to hide .exe fails
        with pytest.raises(ValueError, match="not allowed"):
            sanitize_filename("malicious.jpg\x00.exe")

        # Safe file after null byte removal passes
        result = sanitize_filename("document\x00.pdf")
        assert "\x00" not in result
        assert result == "document.pdf"

    def test_blocked_extensions(self):
        """Dangerous file extensions should be blocked."""
        from integradio.pages.upload import sanitize_filename

        blocked = [
            "script.exe", "script.bat", "script.cmd", "script.com",
            "script.ps1", "script.vbs", "script.sh", "script.php",
            "script.jsp", "script.asp", "script.aspx",
        ]

        for filename in blocked:
            with pytest.raises(ValueError, match="not allowed"):
                sanitize_filename(filename)

    def test_double_extension_attack(self):
        """Double extension attacks should be blocked."""
        from integradio.pages.upload import sanitize_filename

        # file.jpg.exe should be caught (either by primary extension or double extension check)
        with pytest.raises(ValueError, match="not allowed|blocked extension"):
            sanitize_filename("innocent.jpg.exe")

        with pytest.raises(ValueError, match="not allowed|blocked extension"):
            sanitize_filename("document.pdf.bat")

    def test_empty_filename(self):
        """Empty filenames should be rejected."""
        from integradio.pages.upload import sanitize_filename

        with pytest.raises(ValueError, match="Empty"):
            sanitize_filename("")

    def test_dot_filenames(self):
        """Dot filenames should be rejected."""
        from integradio.pages.upload import sanitize_filename

        with pytest.raises(ValueError, match="Invalid"):
            sanitize_filename(".")

        with pytest.raises(ValueError, match="Invalid"):
            sanitize_filename("..")

    def test_long_filename(self):
        """Very long filenames should be rejected."""
        from integradio.pages.upload import sanitize_filename

        long_name = "a" * 300 + ".txt"
        with pytest.raises(ValueError, match="too long"):
            sanitize_filename(long_name)

    def test_special_characters_sanitized(self):
        """Special characters should be replaced with underscores."""
        from integradio.pages.upload import sanitize_filename

        result = sanitize_filename("file<>:\"|?*.txt")
        # Should replace special chars
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result

    def test_unicode_filenames(self):
        """Unicode filenames should be handled safely."""
        from integradio.pages.upload import sanitize_filename

        # These should pass - basic unicode letters are allowed
        assert sanitize_filename("日本語.txt") == "日本語.txt"
        # Emojis are sanitized (replaced with _) since they're outside \w pattern
        # This is intentional - emojis in filenames can cause issues on some systems
        result = sanitize_filename("émoji_🎉.png")
        assert ".png" in result  # Extension preserved
        assert "émoji" in result  # Accented chars preserved


class TestSecretKeyValidation:
    """Tests for EventSigner secret key validation."""

    def test_valid_key_length(self):
        """Keys >= 32 bytes should be accepted."""
        from integradio.events.security import EventSigner

        # 32 bytes hex = 64 characters
        key = EventSigner.generate_key(32)
        signer = EventSigner(key)
        assert signer._key is not None

    def test_short_key_rejected(self):
        """Keys < 32 bytes should be rejected by default."""
        from integradio.events.security import EventSigner

        with pytest.raises(ValueError, match="at least 32 bytes"):
            EventSigner("short_key")

    def test_short_key_with_allow_weak(self):
        """Short keys should work with allow_weak_key=True (for testing)."""
        from integradio.events.security import EventSigner
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            signer = EventSigner("short", allow_weak_key=True)
            assert len(w) == 1
            assert "insecure" in str(w[0].message).lower()

    def test_generate_key_length(self):
        """Generated keys should be the correct length."""
        from integradio.events.security import EventSigner

        key = EventSigner.generate_key(32)
        # Hex encoding doubles the length
        assert len(key) == 64

    def test_key_cryptographic_quality(self):
        """Generated keys should be cryptographically random."""
        from integradio.events.security import EventSigner

        keys = [EventSigner.generate_key() for _ in range(100)]
        # All keys should be unique
        assert len(set(keys)) == 100


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_default_headers_present(self):
        """Default security headers should be defined."""
        from integradio.events.security import DEFAULT_SECURITY_HEADERS

        required = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
        ]
        for header in required:
            assert header in DEFAULT_SECURITY_HEADERS

    def test_csp_defined(self):
        """Content Security Policy should be defined."""
        from integradio.events.security import DEFAULT_CSP

        assert "default-src" in DEFAULT_CSP
        assert "script-src" in DEFAULT_CSP
        assert "frame-ancestors" in DEFAULT_CSP

    def test_middleware_creation(self):
        """Security headers middleware should be creatable."""
        from integradio.events.security import (
            create_security_headers_middleware,
            SecurityHeadersConfig,
        )

        # Default config
        middleware_class = create_security_headers_middleware()
        assert middleware_class is not None

        # Custom config
        config = SecurityHeadersConfig(
            enable_hsts=True,
            enable_csp=False,
        )
        middleware_class = create_security_headers_middleware(config)
        assert middleware_class is not None


class TestGradioConfig:
    """Tests for secure Gradio configuration."""

    def test_secure_config_no_share(self):
        """Secure config should have share=False."""
        from integradio.events.security import get_secure_gradio_config

        config = get_secure_gradio_config()
        assert config["share"] is False

    def test_secure_config_localhost(self):
        """Secure config should bind to localhost."""
        from integradio.events.security import get_secure_gradio_config

        config = get_secure_gradio_config()
        assert config["server_name"] == "127.0.0.1"

    def test_secure_config_file_size_limit(self):
        """Secure config should have file size limit."""
        from integradio.events.security import get_secure_gradio_config

        config = get_secure_gradio_config()
        assert "max_file_size" in config
        assert config["max_file_size"] <= 100 * 1024 * 1024  # <= 100MB


class TestOriginValidation:
    """Tests for WebSocket origin validation."""

    def test_exact_match(self):
        """Exact origin matches should pass."""
        from integradio.events.security import validate_origin

        assert validate_origin(
            "https://example.com",
            ["https://example.com"]
        ) is True

    def test_no_match(self):
        """Non-matching origins should fail."""
        from integradio.events.security import validate_origin

        assert validate_origin(
            "https://evil.com",
            ["https://example.com"]
        ) is False

    def test_wildcard_subdomain(self):
        """Wildcard subdomain matching should work."""
        from integradio.events.security import validate_origin

        assert validate_origin(
            "https://api.example.com",
            ["*.example.com"]
        ) is True

        assert validate_origin(
            "https://evil.com",
            ["*.example.com"]
        ) is False

    def test_none_origin_handling(self):
        """None origin should be rejected by default."""
        from integradio.events.security import validate_origin

        assert validate_origin(None, ["https://example.com"]) is False
        assert validate_origin(
            None,
            ["https://example.com"],
            allow_none=True
        ) is True


class TestMessageSizeValidation:
    """Tests for message size validation."""

    def test_within_limit(self):
        """Messages within limit should pass."""
        from integradio.events.security import validate_message_size

        assert validate_message_size("hello", max_size=100) is True
        assert validate_message_size(b"hello", max_size=100) is True

    def test_exceeds_limit(self):
        """Messages exceeding limit should fail."""
        from integradio.events.security import validate_message_size

        large_msg = "x" * 100000
        assert validate_message_size(large_msg, max_size=1000) is False


class TestRateLimiter:
    """Tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        """Requests under rate limit should be allowed."""
        from integradio.events.security import RateLimiter

        limiter = RateLimiter(rate=100, burst=10)
        result = await limiter.check("client1")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Requests over burst limit should be blocked."""
        from integradio.events.security import RateLimiter

        limiter = RateLimiter(rate=1, burst=2)

        # Use up the burst
        await limiter.check("client1", cost=2)

        # Next request should be blocked
        result = await limiter.check("client1")
        assert result.allowed is False
        assert result.retry_after is not None


class TestNonceTracker:
    """Tests for replay attack prevention."""

    @pytest.mark.asyncio
    async def test_fresh_nonce_accepted(self):
        """Fresh nonces should be accepted."""
        from integradio.events.security import NonceTracker

        tracker = NonceTracker()
        assert await tracker.check_and_add("nonce123") is True

    @pytest.mark.asyncio
    async def test_duplicate_nonce_rejected(self):
        """Duplicate nonces should be rejected."""
        from integradio.events.security import NonceTracker

        tracker = NonceTracker()
        await tracker.check_and_add("nonce123")
        assert await tracker.check_and_add("nonce123") is False
