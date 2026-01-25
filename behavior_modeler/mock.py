"""
Mock flow generator for testing and development.

Generates synthetic user flows based on common UI patterns.
"""

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import Session, FlowEvent


# Common UI flow templates
FLOW_TEMPLATES = {
    "search_to_view": {
        "description": "User searches and views results",
        "steps": [
            ("SearchBox", "input", "code_search"),
            ("SearchResults", "select", "result_selection"),
            ("CodePanel", "view", "code_view"),
        ],
        "completion_rate": 0.85,
    },
    "search_to_generate": {
        "description": "User searches, views, then generates tests",
        "steps": [
            ("SearchBox", "input", "code_search"),
            ("SearchResults", "select", "result_selection"),
            ("CodePanel", "view", "code_view"),
            ("GenerateTestsButton", "click", "test_generation"),
            ("TestOutput", "view", "test_review"),
        ],
        "completion_rate": 0.70,
    },
    "upload_file": {
        "description": "User uploads a file",
        "steps": [
            ("UploadBox", "click", "file_select"),
            ("FileDialog", "select", "file_selection"),
            ("ProgressBar", "view", "upload_progress"),
            ("SuccessMessage", "view", None),
        ],
        "completion_rate": 0.75,
    },
    "upload_error_retry": {
        "description": "Upload fails and user retries",
        "steps": [
            ("UploadBox", "click", "file_select"),
            ("FileDialog", "select", "file_selection"),
            ("ProgressBar", "view", "upload_progress"),
            ("ErrorDialog", "view", "error_handling"),
            ("RetryButton", "click", "retry"),
            ("ProgressBar", "view", "upload_progress"),
            ("SuccessMessage", "view", None),
        ],
        "completion_rate": 0.60,
    },
    "settings_change": {
        "description": "User changes settings",
        "steps": [
            ("SettingsButton", "click", "settings_access"),
            ("SettingsPanel", "view", None),
            ("ThemeToggle", "click", "theme_change"),
            ("SaveButton", "click", "save_settings"),
        ],
        "completion_rate": 0.90,
    },
    "chat_conversation": {
        "description": "User has a chat conversation",
        "steps": [
            ("ChatInput", "input", "message_compose"),
            ("SendButton", "click", "message_send"),
            ("ChatHistory", "view", "response_view"),
            ("ChatInput", "input", "message_compose"),
            ("SendButton", "click", "message_send"),
            ("ChatHistory", "view", "response_view"),
        ],
        "completion_rate": 0.95,
    },
    "abandoned_search": {
        "description": "User starts search but abandons",
        "steps": [
            ("SearchBox", "input", "code_search"),
            ("SearchBox", "clear", None),
        ],
        "completion_rate": 0.0,  # Always incomplete
    },
    "navigation_only": {
        "description": "User just navigates around",
        "steps": [
            ("NavMenu", "click", "navigation"),
            ("DashboardPage", "view", None),
            ("NavMenu", "click", "navigation"),
            ("AnalyticsPage", "view", None),
        ],
        "completion_rate": 0.5,
    },
}


class MockFlowGenerator:
    """Generates synthetic user flows for testing."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self._event_counter = 0

    def generate_session(
        self,
        template_name: Optional[str] = None,
        session_id: Optional[str] = None,
        base_time: Optional[datetime] = None,
    ) -> Session:
        """
        Generate a single session.

        Args:
            template_name: Flow template to use (random if None)
            session_id: Session ID (generated if None)
            base_time: Starting timestamp (now if None)

        Returns:
            Generated Session
        """
        if template_name is None:
            template_name = self.rng.choice(list(FLOW_TEMPLATES.keys()))

        template = FLOW_TEMPLATES[template_name]
        session_id = session_id or str(uuid.uuid4())
        base_time = base_time or datetime.now(timezone.utc)

        # Decide if session completes based on template rate
        will_complete = self.rng.random() < template["completion_rate"]
        steps = template["steps"]
        if not will_complete:
            # Truncate at random point
            cutoff = self.rng.randint(1, max(1, len(steps) - 1))
            steps = steps[:cutoff]

        # Generate events
        events = []
        current_time = base_time

        for i, (component, event_type, intent) in enumerate(steps):
            # Random delay between events (100ms to 5s)
            delay_ms = self.rng.randint(100, 5000)
            current_time = current_time + timedelta(milliseconds=delay_ms)

            self._event_counter += 1
            event = FlowEvent(
                event_id=f"mock_evt_{self._event_counter:08d}",
                timestamp=current_time,
                component_id=hash(component) % 10000,
                component_type=component,
                component_intent=f"Mock {component.lower().replace('_', ' ')}",
                event_type=event_type,
                event_data=self._generate_event_data(component, event_type),
                intent=intent,
                tags=[template_name, f"step_{i}"],
            )
            events.append(event)

        # Add terminal event if complete
        if will_complete:
            current_time = current_time + timedelta(milliseconds=self.rng.randint(50, 500))
            self._event_counter += 1
            terminal = FlowEvent(
                event_id=f"mock_evt_{self._event_counter:08d}",
                timestamp=current_time,
                component_type="System",
                event_type="complete",
                tags=[template_name, "terminal"],
            )
            events.append(terminal)

        return Session(
            session_id=session_id,
            started_at=base_time,
            ended_at=current_time if will_complete else None,
            user_agent=f"MockBrowser/1.0 (Template: {template_name})",
            events=events,
        )

    def generate_batch(
        self,
        count: int,
        template_weights: Optional[dict[str, float]] = None,
        time_spread_hours: float = 24.0,
    ) -> list[Session]:
        """
        Generate multiple sessions.

        Args:
            count: Number of sessions to generate
            template_weights: Probability weights for each template
            time_spread_hours: Spread sessions over this many hours

        Returns:
            List of generated Sessions
        """
        # Default weights (realistic distribution)
        if template_weights is None:
            template_weights = {
                "search_to_view": 0.25,
                "search_to_generate": 0.15,
                "upload_file": 0.10,
                "upload_error_retry": 0.05,
                "settings_change": 0.10,
                "chat_conversation": 0.20,
                "abandoned_search": 0.10,
                "navigation_only": 0.05,
            }

        templates = list(template_weights.keys())
        weights = [template_weights.get(t, 0.1) for t in templates]

        # Generate sessions spread over time
        base_time = datetime.now(timezone.utc) - timedelta(hours=time_spread_hours)
        sessions = []

        for i in range(count):
            # Random time within spread
            offset_hours = self.rng.random() * time_spread_hours
            session_time = base_time + timedelta(hours=offset_hours)

            # Weighted random template
            template = self.rng.choices(templates, weights=weights)[0]

            session = self.generate_session(
                template_name=template,
                base_time=session_time,
            )
            sessions.append(session)

        # Sort by start time
        sessions.sort(key=lambda s: s.started_at)
        return sessions

    def _generate_event_data(self, component: str, event_type: str) -> dict:
        """Generate mock event data based on component and event type."""
        data = {}

        if event_type == "input":
            data["value"] = self.rng.choice([
                "retry decorator",
                "async function",
                "error handling",
                "test coverage",
                "api endpoint",
            ])

        if event_type == "select":
            data["item_index"] = self.rng.randint(0, 9)
            data["item_id"] = f"item_{self.rng.randint(1000, 9999)}"

        if event_type == "click":
            data["x"] = self.rng.randint(0, 1920)
            data["y"] = self.rng.randint(0, 1080)

        if component == "ProgressBar":
            data["progress"] = self.rng.randint(0, 100)

        return data


def generate_sample_flows(count: int = 100, seed: int = 42) -> list[Session]:
    """
    Convenience function to generate sample flows.

    Args:
        count: Number of sessions
        seed: Random seed

    Returns:
        List of Sessions
    """
    generator = MockFlowGenerator(seed=seed)
    return generator.generate_batch(count)


if __name__ == "__main__":
    # Demo: generate and print some flows
    sessions = generate_sample_flows(10)
    for session in sessions:
        print(f"\n{'='*60}")
        print(f"Session: {session.session_id[:8]}...")
        print(f"Duration: {session.duration_ms}ms | Events: {len(session.events)} | Complete: {session.is_complete}")
        for event in session.events:
            print(f"  [{event.event_type:10}] {event.component_type:20} {event.intent or ''}")
