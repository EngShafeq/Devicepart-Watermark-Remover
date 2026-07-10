"""Transactional notification pipeline with templating and delivery tracking.

Order lifecycle events (confirmation, shipping, delivery, review request) are
rendered from named templates and enqueued for delivery. Each message is
tracked from ``queued`` -> ``sent`` (or ``failed``), so operations can see
which confirmations actually went out. Delivery is abstracted behind a
transport; a :class:`RecordingTransport` captures messages for tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from string import Template
from typing import Dict, List, Optional, Protocol


class DeliveryStatus(str, Enum):
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"


@dataclass
class Message:
    id: str
    to: str
    template: str
    subject: str
    body: str
    status: DeliveryStatus = DeliveryStatus.QUEUED
    error: Optional[str] = None


class Transport(Protocol):
    def send(self, message: Message) -> None: ...


class RecordingTransport:
    """Test transport that records every message instead of sending it.

    Set ``fail_templates`` to force a template's delivery to fail, exercising
    the retry/failed-status path.
    """

    def __init__(self, fail_templates: tuple = ()):
        self.sent: List[Message] = []
        self.fail_templates = set(fail_templates)

    def send(self, message: Message) -> None:
        if message.template in self.fail_templates:
            raise RuntimeError(f"transport rejected {message.template}")
        self.sent.append(message)


# Default templates. ``$name`` placeholders are filled from the event context.
DEFAULT_TEMPLATES: Dict[str, tuple] = {
    "order_confirmation": (
        "Your order $order_id is confirmed",
        "Hi $name, thanks for your order $order_id totalling $total. "
        "We'll email you when it ships.",
    ),
    "shipping_update": (
        "Your order $order_id has shipped",
        "Hi $name, order $order_id is on its way. Tracking: $tracking.",
    ),
    "delivery_confirmation": (
        "Your order $order_id was delivered",
        "Hi $name, order $order_id was delivered. We hope you love it!",
    ),
    "review_request": (
        "How was your order $order_id?",
        "Hi $name, please take a moment to review your recent purchase.",
    ),
}


class NotificationService:
    def __init__(self, transport: Transport, *, templates: Optional[Dict[str, tuple]] = None, max_retries: int = 2):
        self.transport = transport
        self.templates = dict(DEFAULT_TEMPLATES)
        if templates:
            self.templates.update(templates)
        self.max_retries = max_retries
        self.log: List[Message] = []

    def render(self, template: str, context: Dict[str, str]) -> tuple:
        if template not in self.templates:
            raise KeyError(f"unknown template {template}")
        subject_tpl, body_tpl = self.templates[template]
        subject = Template(subject_tpl).safe_substitute(context)
        body = Template(body_tpl).safe_substitute(context)
        return subject, body

    def notify(self, to: str, template: str, context: Dict[str, str]) -> Message:
        """Render, enqueue, and attempt delivery with bounded retries."""
        subject, body = self.render(template, context)
        message = Message(uuid.uuid4().hex, to, template, subject, body)
        self.log.append(message)
        last_error: Optional[str] = None
        for _ in range(self.max_retries + 1):
            try:
                self.transport.send(message)
                message.status = DeliveryStatus.SENT
                message.error = None
                return message
            except Exception as exc:  # transport failure -> retry then fail
                last_error = str(exc)
        message.status = DeliveryStatus.FAILED
        message.error = last_error
        return message

    # ---- convenience hooks for the order lifecycle ----------------------
    def order_confirmed(self, to: str, *, name: str, order_id: str, total: str) -> Message:
        return self.notify(to, "order_confirmation", {"name": name, "order_id": order_id, "total": total})

    def order_shipped(self, to: str, *, name: str, order_id: str, tracking: str) -> Message:
        return self.notify(to, "shipping_update", {"name": name, "order_id": order_id, "tracking": tracking})

    def order_delivered(self, to: str, *, name: str, order_id: str) -> Message:
        return self.notify(to, "delivery_confirmation", {"name": name, "order_id": order_id})

    def request_review(self, to: str, *, name: str, order_id: str) -> Message:
        return self.notify(to, "review_request", {"name": name, "order_id": order_id})

    def delivery_stats(self) -> Dict[str, int]:
        stats = {s.value: 0 for s in DeliveryStatus}
        for m in self.log:
            stats[m.status.value] += 1
        return stats
