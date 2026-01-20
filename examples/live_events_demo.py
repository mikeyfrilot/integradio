"""
Live Events Demo - Real-time WebSocket with Gradio UI.

This demo shows:
1. Event mesh running in background
2. Gradio UI emitting events on button clicks
3. Real-time event log updating
4. Security features (signing, rate limiting)

Run with: python examples/live_events_demo.py
Then open http://localhost:7865
"""

import asyncio
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

import gradio as gr
from integradio import SemanticBlocks, semantic
from integradio.events import (
    EventMesh,
    SemanticEvent,
    EventSigner,
    RateLimiter,
)

# Global state
event_log = []
mesh = None
signer = None
rate_limiter = None
stats = {"published": 0, "signed": 0, "rate_limited": 0}


def init_event_system():
    """Initialize the event mesh and security components."""
    global mesh, signer, rate_limiter

    # Generate secure key
    secret_key = EventSigner.generate_key()
    signer = EventSigner(secret_key)

    # Create rate limiter (10/sec for demo)
    rate_limiter = RateLimiter(rate=10.0, burst=20)

    # Create event mesh
    mesh = EventMesh(
        secret_key=secret_key,
        sign_events=True,
        verify_events=True,
        rate_limit=100.0,
    )

    return f"Initialized with key: {secret_key[:16]}..."


def format_event(event: SemanticEvent) -> str:
    """Format event for display."""
    time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    signed = "SIGNED" if event.signature else "unsigned"
    return f"[{time_str}] {event.type} | {signed} | {json.dumps(event.data)}"


def add_to_log(message: str):
    """Add message to event log."""
    global event_log
    event_log.append(message)
    # Keep last 50 entries
    if len(event_log) > 50:
        event_log = event_log[-50:]


async def emit_event_async(event_type: str, data: dict, source: str):
    """Emit event through the mesh."""
    global stats

    # Check rate limit
    result = await rate_limiter.check("demo-user")
    if not result.allowed:
        stats["rate_limited"] += 1
        add_to_log(f"[RATE LIMITED] Retry after {result.retry_after:.2f}s")
        return False

    # Create and sign event
    event = SemanticEvent(type=event_type, source=source, data=data)
    signer.sign(event)
    stats["signed"] += 1

    # Publish
    success = await mesh.publish(event)
    if success:
        stats["published"] += 1
        add_to_log(format_event(event))

    return success


def emit_event(event_type: str, data: dict, source: str = "gradio-ui"):
    """Sync wrapper for emitting events."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(emit_event_async(event_type, data, source))
    finally:
        loop.close()


def on_button_click(button_name: str):
    """Handle button click - emit event."""
    emit_event(
        "ui.button.click",
        {"button": button_name, "timestamp": time.time()},
    )
    return get_log_display(), get_stats_display()


def on_text_submit(text: str):
    """Handle text submission - emit event."""
    if text.strip():
        emit_event(
            "ui.input.submit",
            {"text": text, "length": len(text)},
        )
    return get_log_display(), get_stats_display(), ""


def on_slider_change(value: float):
    """Handle slider change - emit event."""
    emit_event(
        "ui.slider.change",
        {"value": value},
    )
    return get_log_display(), get_stats_display()


def spam_events():
    """Spam events to test rate limiting."""
    for i in range(25):
        emit_event(
            "ui.spam.test",
            {"index": i},
            source="spam-test",
        )
    return get_log_display(), get_stats_display()


def test_signature():
    """Test signature verification."""
    # Create event
    event = SemanticEvent(
        type="test.signature",
        source="demo",
        data={"message": "Hello, secure world!"},
    )

    # Sign it
    signer.sign(event)
    original_sig = event.signature

    # Verify (should pass)
    valid_before = signer.verify(event)

    # Tamper with data
    event.data["message"] = "Tampered!"
    valid_after = signer.verify(event)

    result = f"""
Signature Test Results:
-----------------------
Original signature: {original_sig[:32]}...
Valid before tamper: {valid_before}
Valid after tamper: {valid_after}

Conclusion: {"PASS - Tampering detected!" if not valid_after else "FAIL"}
"""
    add_to_log("[TEST] Signature verification test completed")
    return result, get_log_display(), get_stats_display()


def get_log_display():
    """Get formatted log for display."""
    if not event_log:
        return "No events yet. Click buttons to emit events!"
    return "\n".join(reversed(event_log[-20:]))


def get_stats_display():
    """Get stats for display."""
    return f"Published: {stats['published']} | Signed: {stats['signed']} | Rate Limited: {stats['rate_limited']}"


def create_demo():
    """Create the Gradio demo."""
    # Initialize on startup
    init_message = init_event_system()

    with SemanticBlocks(title="Semantic Events Live Demo") as demo:
        gr.Markdown("# Real-Time Event Mesh Demo")
        gr.Markdown("Click buttons to emit CloudEvents with HMAC signatures")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")

                # Buttons
                btn1 = semantic(
                    gr.Button("Click Me!", variant="primary"),
                    intent="primary action button",
                )
                btn2 = semantic(
                    gr.Button("Secondary Action"),
                    intent="secondary action button",
                )
                btn3 = semantic(
                    gr.Button("Danger!", variant="stop"),
                    intent="dangerous action button",
                )

                gr.Markdown("---")

                # Text input
                text_input = semantic(
                    gr.Textbox(label="Send a message", placeholder="Type and press Enter"),
                    intent="text message input",
                )

                # Slider
                slider = semantic(
                    gr.Slider(0, 100, value=50, label="Adjust Value"),
                    intent="value adjustment slider",
                )

                gr.Markdown("---")
                gr.Markdown("### Security Tests")

                spam_btn = gr.Button("Spam Events (Test Rate Limiting)")
                sig_btn = gr.Button("Test Signature Verification")

                sig_result = gr.Textbox(label="Signature Test Result", lines=8)

            with gr.Column(scale=2):
                gr.Markdown("### Event Log")
                gr.Markdown(f"*{init_message}*")

                stats_display = gr.Textbox(
                    label="Statistics",
                    value=get_stats_display(),
                    interactive=False,
                )

                log_display = gr.Textbox(
                    label="Recent Events (newest first)",
                    value=get_log_display(),
                    lines=20,
                    interactive=False,
                )

        # Wire up events
        btn1.click(
            fn=lambda: on_button_click("primary"),
            outputs=[log_display, stats_display],
        )
        btn2.click(
            fn=lambda: on_button_click("secondary"),
            outputs=[log_display, stats_display],
        )
        btn3.click(
            fn=lambda: on_button_click("danger"),
            outputs=[log_display, stats_display],
        )

        text_input.submit(
            fn=on_text_submit,
            inputs=[text_input],
            outputs=[log_display, stats_display, text_input],
        )

        slider.change(
            fn=on_slider_change,
            inputs=[slider],
            outputs=[log_display, stats_display],
        )

        spam_btn.click(
            fn=spam_events,
            outputs=[log_display, stats_display],
        )

        sig_btn.click(
            fn=test_signature,
            outputs=[sig_result, log_display, stats_display],
        )

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SEMANTIC EVENTS - Live Demo")
    print("=" * 60)
    print("\nFeatures demonstrated:")
    print("  - CloudEvents format")
    print("  - HMAC-SHA256 signing")
    print("  - Rate limiting (10/sec)")
    print("  - Real-time event log")
    print("\nOpen http://localhost:7865")
    print("=" * 60 + "\n")

    demo = create_demo()
    demo.launch(server_port=7865)
