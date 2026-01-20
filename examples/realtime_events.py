"""
Real-time Events Demo - Secure WebSocket event mesh with Integradio.

This example demonstrates:
- CloudEvents-compliant event format
- HMAC-SHA256 message signing
- Pattern-based subscriptions
- Rate limiting and security controls
- Real-time UI updates

Run with: python examples/realtime_events.py
"""

import asyncio
import sys
sys.path.insert(0, ".")

from integradio.events import (
    EventMesh,
    SemanticEvent,
    EventType,
    EventSigner,
    RateLimiter,
    on_event,
)


async def demo_event_mesh():
    """Demonstrate the EventMesh pub/sub system."""
    print("\n" + "=" * 70)
    print("SEMANTIC EVENTS - Secure Event Mesh Demo")
    print("=" * 70)

    # Create event mesh with secure signing
    secret_key = EventSigner.generate_key()
    print(f"\nGenerated secret key: {secret_key[:16]}...")

    mesh = EventMesh(
        secret_key=secret_key,
        sign_events=True,
        verify_events=True,
        rate_limit=100.0,  # 100 events/second
        rate_burst=200,
    )

    # Track received events
    received_events = []

    # Subscribe to UI component events
    @mesh.on("ui.component.*")
    async def handle_component_events(event: SemanticEvent):
        print(f"  [UI] {event.type}: {event.data}")
        received_events.append(event)

    # Subscribe to data events with priority
    @mesh.on("data.*")
    async def handle_data_events(event: SemanticEvent):
        print(f"  [DATA] {event.type}: {event.data}")
        received_events.append(event)

    # Subscribe to all system events
    @mesh.on("system.**")
    async def handle_system_events(event: SemanticEvent):
        print(f"  [SYSTEM] {event.type}")
        received_events.append(event)

    # Start the mesh
    async with mesh:
        print("\n1. Publishing signed events...")
        print("-" * 40)

        # Emit various events
        await mesh.emit(
            "ui.component.click",
            {"component_id": "btn-1", "x": 100, "y": 200},
            source="demo-app",
        )

        await mesh.emit(
            "ui.component.input",
            {"component_id": "text-1", "value": "Hello World"},
            source="demo-app",
        )

        await mesh.emit(
            "data.loaded",
            {"table": "users", "count": 42},
            source="database",
        )

        await mesh.emit(
            "system.heartbeat",
            {"uptime": 3600},
            source="monitor",
        )

        # Let events process
        await asyncio.sleep(0.5)

        print(f"\nReceived {len(received_events)} events")

        # Demonstrate signature verification
        print("\n2. Signature Verification Demo")
        print("-" * 40)

        # Create and sign an event
        event = SemanticEvent(
            type="demo.test",
            source="test",
            data={"message": "Hello, secure world!"},
        )

        signer = mesh.signer
        signer.sign(event)
        print(f"Event ID: {event.id[:8]}...")
        print(f"Nonce: {event.nonce}")
        print(f"Signature: {event.signature[:32]}...")

        # Verify
        is_valid = signer.verify(event)
        print(f"Signature valid: {is_valid}")

        # Tamper and verify again
        event.data["message"] = "Tampered!"
        is_valid_after_tamper = signer.verify(event)
        print(f"After tampering: {is_valid_after_tamper}")

        # Demonstrate rate limiting
        print("\n3. Rate Limiting Demo")
        print("-" * 40)

        limiter = RateLimiter(rate=5.0, burst=10)  # 5/sec, burst of 10

        for i in range(15):
            result = await limiter.check("test-client")
            status = "ALLOWED" if result.allowed else "BLOCKED"
            print(f"  Request {i+1:2d}: {status} (remaining: {result.remaining})")
            if not result.allowed:
                print(f"         Retry after: {result.retry_after:.2f}s")

        # Pattern matching demo
        print("\n4. Pattern Matching Demo")
        print("-" * 40)

        patterns = [
            ("ui.component.click", "ui.component.*"),
            ("ui.component.click", "ui.*"),
            ("ui.component.click", "ui.**"),
            ("data.users.loaded", "data.*"),
            ("data.users.loaded", "data.**"),
            ("system.heartbeat", "system.*"),
        ]

        for event_type, pattern in patterns:
            test_event = SemanticEvent(type=event_type, source="test")
            matches = test_event.matches_pattern(pattern)
            symbol = "OK" if matches else "--"
            print(f"  {event_type:25s} ~ {pattern:15s} : {symbol}")

        # Show stats
        print("\n5. Mesh Statistics")
        print("-" * 40)
        stats = mesh.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


async def demo_decorator_handlers():
    """Demonstrate the @on_event decorator system."""
    print("\n" + "=" * 70)
    print("EVENT HANDLERS - Decorator Demo")
    print("=" * 70)

    from integradio.events.handlers import (
        get_global_registry,
        clear_global_registry,
    )

    # Clear any existing handlers
    clear_global_registry()

    # Register handlers with decorators
    @on_event("ui.button.click", priority=10, timeout=5.0)
    async def high_priority_handler(event: SemanticEvent):
        print(f"  [HIGH] Processing: {event.data}")

    @on_event("ui.button.*", "ui.input.*", priority=5)
    async def medium_priority_handler(event: SemanticEvent):
        print(f"  [MED] Processing: {event.data}")

    @on_event("ui.**", priority=1)
    async def catch_all_handler(event: SemanticEvent):
        print(f"  [LOW] Catch-all: {event.type}")

    # Get registry and show handlers
    registry = get_global_registry()

    print("\nRegistered handlers:")
    for stat in registry.get_all_stats():
        print(f"  - {stat['name']}: patterns={stat['patterns']}, priority={stat['priority']}")

    # Dispatch a test event
    print("\nDispatching ui.button.click event...")
    test_event = SemanticEvent(
        type="ui.button.click",
        source="demo",
        data={"button_id": "submit"},
    )

    count = await registry.dispatch(test_event)
    print(f"Called {count} handlers")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nIntegradio Events - Security Demo")
    print("=====================================\n")

    # Run demos
    asyncio.run(demo_event_mesh())
    asyncio.run(demo_decorator_handlers())
