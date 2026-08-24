from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ORDER_ID_RE = re.compile(r"\bORD[\s_-]*\d{4}\b", re.IGNORECASE)
STRICT_ORDER_ID_RE = re.compile(r"^ORD-\d{4}$", re.IGNORECASE)

PRIVATE_FIELDS = {
    "customer_email",
    "customer_address",
    "internal_note",
    "risk_score",
}


class OrderResult:
    def __init__(self, found: bool, data: dict[str, Any] | None = None, error: str | None = None, handoff: bool = False):
        self.found = found
        self.data = data
        self.error = error
        self.requires_handoff = handoff
        self.handoff_reason = error or ""


class OrderStore:
    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        orders = raw.get("orders", raw)
        self.orders = {
            str(order["order_id"]).upper(): order
            for order in orders
        }

    @staticmethod
    def normalize(order_id: str) -> str:
        value = str(order_id or "").strip().upper()
        value = re.sub(r"^ORD[\s_-]*", "ORD-", value)
        return value

    def lookup(self, order_id: str) -> OrderResult:
        normalized = self.normalize(order_id)

        if not STRICT_ORDER_ID_RE.fullmatch(normalized):
            return OrderResult(
                False,
                error="Please provide a valid order ID such as ORD-1007.",
                handoff=False,
            )

        order = self.orders.get(normalized)
        if order is None:
            return OrderResult(
                False,
                error="Order was not found.",
                handoff=True,
            )

        safe: dict[str, Any] = {}
        for key in ("order_id", "status", "carrier", "estimated_delivery", "delivered_at"):
            if key in order:
                safe[key] = order[key]

        # Cancelled/returned orders must not leak stale carrier/ETA fields.
        if str(order.get("status", "")).lower() in {"cancelled", "returned"}:
            safe.pop("carrier", None)
            safe.pop("estimated_delivery", None)

        return OrderResult(True, safe, handoff=False)


def normalize_order_id(value: str) -> str:
    value = str(value or "").strip().upper()
    value = re.sub(r"^ORD[\s_-]*", "ORD-", value)
    return value

class OrderLookup:
    """Backward-compatible dictionary-returning order lookup facade."""

    def __init__(self, path: str | Path):
        self.store = OrderStore(path)

    def lookup(self, order_id: str) -> dict[str, Any]:
        normalized = normalize_order_id(order_id)
        if not STRICT_ORDER_ID_RE.fullmatch(normalized):
            return {"found": False, "error": "malformed_order_id"}

        result = self.store.lookup(normalized)
        if not result.found:
            return {
                "found": False,
                "error": "not_found",
                "order_id": normalized,
            }

        data = {"found": True, **(result.data or {})}
        return data
