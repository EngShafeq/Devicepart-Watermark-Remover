"""Analytics event tracking for conversion-funnel analysis.

Events are persisted to SQLite so the funnel can be computed across sessions
and after a restart. The canonical funnel — product view -> add to cart ->
checkout step -> purchase — is expressed as an ordered list of stages, and
:meth:`AnalyticsService.funnel` reports how many *sessions* reached each stage
plus the step-to-step conversion rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .db import Database


class Event:
    PRODUCT_VIEW = "product_view"
    ADD_TO_CART = "add_to_cart"
    CHECKOUT_STEP = "checkout_step"
    PURCHASE = "purchase"


# Ordered funnel stages. A checkout-step event carries its step name in props;
# the funnel treats reaching checkout at all as one stage.
FUNNEL: List[str] = [
    Event.PRODUCT_VIEW,
    Event.ADD_TO_CART,
    Event.CHECKOUT_STEP,
    Event.PURCHASE,
]


@dataclass
class FunnelStage:
    name: str
    sessions: int
    conversion_from_previous: float  # 0..1, 1.0 for the first stage


class AnalyticsService:
    def __init__(self, db: Database, *, clock=None):
        self.db = db
        if clock is None:
            import time

            clock = time.time
        self._clock = clock

    def track(self, session_id: str, name: str, **props) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO analytics_events (session_id, name, props, at) VALUES (?, ?, ?, ?)",
                (session_id, name, json.dumps(props), self._clock()),
            )

    # ---- convenience emitters -------------------------------------------
    def product_view(self, session_id: str, product_id: str) -> None:
        self.track(session_id, Event.PRODUCT_VIEW, product_id=product_id)

    def add_to_cart(self, session_id: str, sku: str, quantity: int) -> None:
        self.track(session_id, Event.ADD_TO_CART, sku=sku, quantity=quantity)

    def checkout_step(self, session_id: str, step: str) -> None:
        self.track(session_id, Event.CHECKOUT_STEP, step=step)

    def purchase(self, session_id: str, order_id: str, total_minor: int) -> None:
        self.track(session_id, Event.PURCHASE, order_id=order_id, total=total_minor)

    # ---- reporting -------------------------------------------------------
    def sessions_reaching(self, event_name: str) -> set:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM analytics_events WHERE name = ?",
                (event_name,),
            ).fetchall()
        return {r["session_id"] for r in rows}

    def funnel(self) -> List[FunnelStage]:
        """Session counts and step conversion for the canonical funnel."""
        stages: List[FunnelStage] = []
        prev_count: Optional[int] = None
        for name in FUNNEL:
            count = len(self.sessions_reaching(name))
            if prev_count is None:
                rate = 1.0
            elif prev_count == 0:
                rate = 0.0
            else:
                rate = round(count / prev_count, 4)
            stages.append(FunnelStage(name, count, rate))
            prev_count = count
        return stages

    def event_counts(self) -> Dict[str, int]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT name, COUNT(*) AS n FROM analytics_events GROUP BY name"
            ).fetchall()
        return {r["name"]: r["n"] for r in rows}
