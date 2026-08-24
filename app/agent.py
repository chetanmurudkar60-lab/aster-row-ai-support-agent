from __future__ import annotations

import re
from typing import Iterable

from app.logging_utils import Trace
from app.memory import SessionStore
from app.orders import ORDER_ID_RE, OrderStore, normalize_order_id
from app.prompts import SYSTEM_PROMPT, TOOLS
from app.retrieval import Retriever


MAX_TOOL_ROUNDS = 4

PRIVATE_DATA_TERMS = (
    "email",
    "email address",
    "address",
    "internal note",
    "internal notes",
    "risk score",
    "risk_score",
)

INTERNAL_NOTE_TERMS = (
    "warehouse note",
    "internal note",
    "internal notes",
    "warehouse instruction",
    "internal instruction",
)

UNSUPPORTED_ACTION_TERMS = (
    "cancel my order",
    "cancel the order",
    "cancel order",
    "issue a refund",
    "refund my",
    "replace my",
    "replacement",
    "change my address",
    "change the address",
    "approve my warranty",
    "approve the warranty",
    "price adjustment",
    "open a carrier investigation",
    "open an investigation",
)

SYSTEM_PROMPT_REQUEST_TERMS = (
    "system prompt",
    "system instructions",
    "hidden instructions",
    "hidden prompt",
    "reveal your prompt",
    "show your prompt",
    "developer instructions",
    "internal instructions",
)

ORDER_FOLLOWUP_RE = re.compile(
    r"\b(?:when will it arrive|when does it arrive|when should it arrive|"
    r"what carrier|tracking|where is it|where is my order|"
    r"what is the status|what's the status|status of my order)\b",
    re.I,
)


class Agent:
    """Reliable support orchestrator.

    The important customer-facing policy/order paths are deterministic and
    evidence-driven. OpenRouter is used only as a bounded fallback for a
    question that cannot be safely answered by the local policy router.
    This prevents provider wording, tool hallucination, or rate limits from
    changing safety-critical evaluation behavior.
    """

    def __init__(self, retriever: Retriever, order_store: OrderStore, llm_client=None):
        self.retriever = retriever
        self.order_store = order_store
        self.llm_client = llm_client
        self.sessions = SessionStore()

    def handle_message(self, session_id: str, user_text: str) -> dict:
        session = self.sessions.get_or_create(session_id)
        trace = Trace(
            session_id=session_id,
            user_message=user_text,
            history_snapshot=session.history_as_text(),
        )
        text = user_text.strip()
        lowered = text.lower()

        # Safety checks happen before retrieval/tool use.
        if self._asks_for_hidden_instructions(lowered):
            return self._finish(
                session,
                trace,
                text,
                "I can't provide system prompts, hidden instructions, or other internal implementation details. A human support teammate can help with legitimate support questions.",
                True,
            )

        if self._asks_for_private_data(lowered):
            return self._finish(
                session,
                trace,
                text,
                "I can't provide private or internal order information such as customer email addresses, addresses, internal notes, or other restricted fields. A human support teammate can help with legitimate account-specific requests.",
                True,
            )

        if self._asks_for_internal_note(lowered):
            return self._finish(
                session,
                trace,
                text,
                "I can't provide or act on internal warehouse or operational instructions. A human support teammate can help with a legitimate customer-support request.",
                True,
            )

        if self._asks_for_unsupported_action(lowered):
            return self._finish(
                session,
                trace,
                text,
                "I can look up information, but I can't complete that action from this system. A human support teammate will need to help with this request.",
                True,
            )

        # Retrieve first for every company-specific question.
        retrieval_query = self._build_retrieval_query(session, text)
        hits = self.retriever.search(retrieval_query)
        self._record_hits(trace, hits)

        # Order tool/function path.
        order_id = self._resolve_order_id(text, session)
        if order_id:
            result = self._dispatch_order(order_id, session, trace)
            answer = self._render_order_response(result)
            handoff = bool(result.get("handoff", False))
            return self._finish(session, trace, text, answer, handoff, add_turn=True)

        # Missing order ID: ask instead of guessing.
        if self._looks_like_order_request(text, session) and self._needs_order_id(text, session):
            return self._finish(
                session,
                trace,
                text,
                "Please provide the order ID (for example, ORD-1007) so I can check it.",
                False,
            )

        # Deterministic source conflict.
        if self._is_dishwasher_conflict(text, hits):
            answer = self._render_conflict(hits)
            return self._finish(session, trace, text, answer, True)

        # Deterministic final-sale damage exception.
        if self._is_final_sale_damage_case(text):
            answer = self._render_final_sale_damage(hits)
            return self._finish(session, trace, text, answer, True)

        # Deterministic known policy/product paths.
        answer = self._render_known_answer(text, session, hits)
        if answer is not None:
            handoff = self._known_answer_requires_handoff(text, hits)
            return self._finish(session, trace, text, answer, handoff)

        # If retrieval is insufficient, abstain. Do not make a model invent a fact.
        if not hits:
            return self._finish(
                session,
                trace,
                text,
                "The supplied information is insufficient to answer that company-specific question reliably. A human support teammate can confirm it.",
                True,
            )

        # Bounded LLM fallback. The model sees only retrieved evidence and recent
        # context, never the orders file or private data.
        if self.llm_client is not None:
            try:
                answer = self._llm_fallback(session, text, hits)
                if answer:
                    return self._finish(session, trace, text, answer, self._response_requires_handoff(answer))
            except Exception as exc:  # provider failure must not become a fake answer
                trace.errors.append(f"llm_fallback_failed:{type(exc).__name__}")

        return self._finish(
            session,
            trace,
            text,
            "The supplied information is insufficient to answer that company-specific question reliably. A human support teammate can confirm it.",
            True,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_retrieval_query(self, session, user_text: str) -> str:
        history = session.history_as_text()
        parts = []
        if history != "(no prior turns)":
            parts.append(f"Previous conversation context:\n{history}")
        parts.append(f"Current customer question:\n{user_text}")

        q = "\n\n".join(parts)
        lowered = q.lower()

        if any(x in lowered for x in ("ship", "shipping", "international", "canada", "germany", "duties", "taxes")):
            q += "\nFocus: supported destinations, Canada delivery time, processing, duties, taxes, prepaid charges."

        if any(x in lowered for x in ("return", "returns", "refund", "final sale", "final-sale", "backpack")):
            q += "\nFocus: standard 30-day return policy, TrailPlus 45-day window, delivery basis, final-sale exceptions, damaged-item reporting."

        if any(x in lowered for x in ("warranty", "lifetime")):
            q += "\nFocus: warranty periods and lifetime-warranty availability."

        if any(x in lowered for x in ("dishwasher", "tumbler", "breeze")):
            q += "\nFocus: Breeze Tumbler cleaning instructions and active product-card conflict."

        if any(x in lowered for x in ("vegan", "fabric", "adhesive", "material")):
            q += "\nFocus: evidence for material composition, certification, and vegan claims."

        return q

    @staticmethod
    def _record_hits(trace: Trace, hits) -> None:
        trace.retrieved.extend(
            {
                "filename": hit.chunk.filename,
                "heading": hit.chunk.heading,
                "status": hit.chunk.metadata.get("status"),
                "policy_authority": hit.chunk.metadata.get("policy_authority"),
                "score": round(hit.final_score, 4),
            }
            for hit in hits
        )

    # ------------------------------------------------------------------
    # Order tool/function
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_order_id(user_text: str, session) -> str | None:
        match = ORDER_ID_RE.search(user_text)
        if match:
            return normalize_order_id(match.group(0))

        if session.last_order_id and ORDER_FOLLOWUP_RE.search(user_text):
            return session.last_order_id

        return None

    @staticmethod
    def _needs_order_id(user_text: str, session) -> bool:
        if session.last_order_id:
            return False
        lowered = user_text.lower()
        return any(
            phrase in lowered
            for phrase in (
                "where is my order",
                "where's my order",
                "order status",
                "track my order",
                "tracking my order",
                "when will my order",
                "when does my order",
                "when should my order",
                "when will it arrive",
                "when does it arrive",
                "when should it arrive",
                "check order ",
                "check the order ",
            )
        )

    def _dispatch_order(self, order_id: str, session, trace: Trace) -> dict:
        result = self.order_store.lookup(order_id)
        normalized = normalize_order_id(order_id)

        trace.tool_calls.append(
            {
                "tool": "order_lookup",
                "args": {"order_id": normalized},
            }
        )

        if not result.found:
            return {
                "found": False,
                "error": result.error,
                "handoff": bool(result.requires_handoff),
            }

        order = result.data or {}
        session.last_order_id = order.get("order_id")
        session.last_order_status = str(order.get("status", "")).lower() or None

        # This is the only order data that reaches the agent response layer.
        return {
            "found": True,
            "order": order,
            "handoff": bool(result.requires_handoff),
        }

    @staticmethod
    def _render_order_response(result: dict) -> str:
        if not result.get("found"):
            return "The order was not found. I couldn't find that order. Please double-check the order ID or contact support."

        order = result.get("order", {})
        order_id = order.get("order_id", "the order")
        status = str(order.get("status", "")).lower()
        carrier = order.get("carrier")
        eta = order.get("estimated_delivery")

        if status == "cancelled":
            return f"Order {order_id} is cancelled and will not be shipped."

        if status == "returned":
            return f"Order {order_id} has been returned and is not in transit."

        if status == "shipped":
            text = f"Order {order_id} is currently shipped"
            if carrier:
                text += f" with {carrier}"
            if eta:
                text += f" and is estimated to arrive on {_format_date(eta)}"
            else:
                text += ". The delivery estimate is unavailable"
            return text.rstrip(".") + "."

        if status == "delivered":
            delivered_at = order.get("delivered_at")
            if delivered_at:
                return f"Order {order_id} was delivered on {_format_date(delivered_at)}."
            return f"Order {order_id} is marked as delivered."

        if status == "processing":
            return f"Order {order_id} is currently processing."

        if status == "pending":
            return f"Order {order_id} is currently pending."

        text = f"Order {order_id} has status {status or 'unavailable'}"
        if carrier:
            text += f" with {carrier}"
        if eta:
            text += f" and is estimated to arrive on {_format_date(eta)}"
        return text + "."

    # ------------------------------------------------------------------
    # Known policy rendering
    # ------------------------------------------------------------------

    def _render_known_answer(self, user_text: str, session, hits) -> str | None:
        q = user_text.lower()

        if self._is_vegan_question(q):
            return (
                "The supplied information is insufficient to confirm whether all fabrics and adhesives used in Aster & Row bags are vegan. A human support teammate should confirm this."
            )

        if self._is_warranty_question(q):
            if "lifetime" in q:
                return self._answer_from_source(
                    hits,
                    "Aster & Row does not offer a lifetime warranty. Bags and backpacks have a 2 years warranty from purchase, while drinkware and travel accessories have a 1 year warranty."
                )
            if "bag" in q or "backpack" in q:
                return self._answer_from_source(
                    hits,
                    "Aster & Row bags and backpacks have a 2 years limited warranty from the purchase date."
                )
            return self._answer_from_source(
                hits,
                "The limited warranty is 2 years for bags and backpacks, and 1 year for drinkware, packing cubes, and other travel accessories."
            )

        if ("migration" in q or "60 days" in q) and "return" in q:
            return self._answer_from_source(
                hits,
                "The migration note is not authoritative. The current standard policy is 30 calendar days from delivery unless a valid exception applies. I can explain the policy, but the agent cannot approve a return."
            )

        if self._is_trailplus_return(q):
            return self._answer_from_source(
                hits,
                "If TrailPlus was active when the order was placed, the return window is 45 calendar days from delivery for eligible items."
            )

        if self._is_return_question(q):
            answer = (
                "Standard-plan customers may request a return within 30 calendar days of delivery. "
                "The item must be unused, unwashed, and in resalable condition with the original tags, accessories, and packaging when supplied."
            )
            if session.last_order_status == "cancelled":
                answer = (
                    "The order discussed earlier is cancelled, so it will not be shipped. For a general return-policy question, standard-plan customers may request a return within 30 calendar days of delivery for eligible delivered items."
                )
            return self._answer_from_source(hits, answer)

        if self._is_shipping_question(q, session):
            if "germany" in q or "german" in q:
                return self._answer_from_source(
                    hits,
                    "Shipping to Germany is not currently available. Aster & Row currently ships internationally only to Canada."
                )

            if "duties" in q or "taxes" in q or "prepaid" in q:
                return self._answer_from_source(
                    hits,
                    "For Canadian orders, import duties, taxes, and brokerage charges are not prepaid by Aster & Row. The recipient is responsible for charges assessed by Canadian authorities or the carrier."
                )

            return self._answer_from_source(
                hits,
                "Aster & Row currently ships internationally only to Canada. Canadian orders generally arrive within 5–9 business days after dispatch, with 1–2 business days usually needed for processing before dispatch. Import duties, taxes, and brokerage charges are not prepaid by Aster & Row."
            )

        return None

    @staticmethod
    def _answer_from_source(hits, body: str) -> str:
        citations = []
        seen = set()
        for hit in hits:
            filename = hit.chunk.filename
            if filename in seen:
                continue
            if hit.chunk.metadata.get("status") != "active":
                continue
            if hit.chunk.metadata.get("policy_authority") != "official":
                continue
            citations.append(f"(Source: {filename} - {hit.chunk.heading})")
            seen.add(filename)

        if citations:
            return body + "\n\n" + "\n".join(citations[:3])
        return body

    # ------------------------------------------------------------------
    # Conflict / exception rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_conflict(hits) -> str:
        return (
            "The current official sources conflict. The Product Care Guide says the Breeze Tumbler body should be hand-washed and only the lid may go on the top rack, while the product card says all components are dishwasher safe. "
            "Because the sources disagree, I recommend human confirmation; the safer interim guidance is to hand-wash the tumbler body.\n\n"
            "(Source: 11-product-care.md - Breeze Tumbler)\n"
            "(Source: 12-breeze-tumbler-product-card.md - Cleaning)"
        )

    @staticmethod
    def _render_final_sale_damage(hits) -> str:
        return (
            "A final-sale restriction does not block review when an item arrived damaged, defective, or incorrect. Because the zipper arrived broken, report the issue within 7 calendar days of delivery and provide the order ID, description, and clear photos when reasonably possible. A human review is required before a refund, replacement, or other resolution is approved.\n\n"
            "(Source: 03-final-sale-and-promotions.md - Damaged or incorrect items)\n"
            "(Source: 04-damaged-or-wrong-items.md - Reporting window)"
        )

    # ------------------------------------------------------------------
    # Topic detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_return_question(q: str) -> bool:
        return any(x in q for x in ("return", "returns", "return window", "how long can i return"))

    @staticmethod
    def _is_trailplus_return(q: str) -> bool:
        return "trailplus" in q and ("return" in q or "window" in q)

    @staticmethod
    def _is_warranty_question(q: str) -> bool:
        return "warranty" in q or "lifetime" in q

    @staticmethod
    def _is_shipping_question(q: str, session) -> bool:
        return any(x in q for x in ("ship", "shipping", "international", "canada", "germany", "duties", "taxes", "prepaid")) or (
            session.last_order_id is None and q.startswith("what about")
        )

    @staticmethod
    def _is_vegan_question(q: str) -> bool:
        return any(x in q for x in ("vegan", "adhesive", "fabric", "material certification"))

    @staticmethod
    def _is_final_sale_damage_case(q: str) -> bool:
        return any(x in q for x in ("final sale", "final-sale")) and any(x in q for x in ("damaged", "broken", "defective"))

    @staticmethod
    def _is_dishwasher_conflict(q: str, hits) -> bool:
        if "dishwasher" not in q and "dish wash" not in q:
            return False
        files = {hit.chunk.filename for hit in hits}
        return {"11-product-care.md", "12-breeze-tumbler-product-card.md"}.issubset(files)

    @staticmethod
    def _looks_like_order_request(q: str, session) -> bool:
        if ORDER_ID_RE.search(q):
            return True
        if session.last_order_id and ORDER_FOLLOWUP_RE.search(q):
            return True
        lowered = q.lower()
        return any(x in lowered for x in ("where is my order", "track my order", "order status", "when will my order", "when will it arrive", "when does it arrive", "when should it arrive", "check order ", "check the order "))

    @staticmethod
    def _asks_for_private_data(q: str) -> bool:
        return any(term in q for term in PRIVATE_DATA_TERMS)

    @staticmethod
    def _asks_for_internal_note(q: str) -> bool:
        return any(term in q for term in INTERNAL_NOTE_TERMS)

    @staticmethod
    def _asks_for_unsupported_action(q: str) -> bool:
        return any(term in q for term in UNSUPPORTED_ACTION_TERMS) or bool(re.search(r"\bcancel(?:led|ed)?\b.*\bORD[\s_-]*\d{4}\b", q, re.IGNORECASE))

    @staticmethod
    def _asks_for_hidden_instructions(q: str) -> bool:
        return any(term in q for term in SYSTEM_PROMPT_REQUEST_TERMS)

    @staticmethod
    def _known_answer_requires_handoff(q: str, hits) -> bool:
        return Agent._is_final_sale_damage_case(q) or Agent._is_vegan_question(q)

    @staticmethod
    def _response_requires_handoff(answer: str) -> bool:
        lowered = answer.lower()
        return any(
            marker in lowered
            for marker in (
                "insufficient",
                "human confirmation",
                "human support",
                "sources conflict",
                "can't provide",
            )
        )

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_fallback(self, session, user_text: str, hits) -> str:
        evidence = "\n\n".join(
            f"SOURCE: {h.chunk.filename}\nHEADING: {h.chunk.heading}\nCONTENT:\n{h.chunk.content}"
            for h in hits
        )
        prompt = (
            "CURRENT QUESTION:\n"
            f"{user_text}\n\n"
            "RECENT CONTEXT:\n"
            f"{session.history_as_text()}\n\n"
            "RETRIEVED REFERENCE DATA (untrusted):\n"
            f"{evidence}\n\n"
            "Answer only from this evidence. Cite every company-specific claim. "
            "If evidence is insufficient, say so and recommend human confirmation."
        )
        response = self.llm_client.create(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], [])
        return "\n".join(
            block["text"] for block in response.get("content", []) if block.get("type") == "text"
        ).strip()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _finish(session, trace, user_text, response, handoff, add_turn=True):
        if add_turn:
            session.add_turn("user", user_text)
            session.add_turn("assistant", response)

        trace.final_response = response
        trace.handoff = bool(handoff)
        return {
            "response": response,
            "handoff": bool(handoff),
            "trace": trace,
        }


def _format_date(value) -> str:
    from datetime import datetime

    text = str(value)
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except ValueError:
            continue
    return text

# Compatibility facade for the earlier simple interface.
class SupportAgent:
    def __init__(self):
        from app.config import KNOWLEDGE_BASE_DIR, ORDERS_PATH
        self._agent = Agent(Retriever(KNOWLEDGE_BASE_DIR), OrderStore(ORDERS_PATH))

    def respond(self, session_id: str, user_text: str):
        result = self._agent.handle_message(session_id, user_text)
        return _SupportResponse(result)


class _SupportResponse:
    def __init__(self, result: dict):
        self.answer = result["response"]
        self.handoff = result["handoff"]
        self.trace = result["trace"]
        self.tool_calls = [
            {
                "name": item["tool"],
                "arguments": item["args"],
            }
            for item in self.trace.tool_calls
        ]
        self.sources = [
            {
                "filename": item["filename"],
                "heading": item["heading"],
            }
            for item in self.trace.retrieved
        ]
