from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent import Agent
from app.orders import OrderStore
from app.retrieval import Retriever


def load_cases(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def concept_ok(text: str, concept: str) -> bool:
    t = re.sub(r"[^a-z0-9]+", " ", text.lower())
    c = concept.lower()
    rules = {
        "canada is supported": [["canada"], ["ship", "supported"]],
        "5–9 business days after dispatch": [["5"], ["9"], ["business days"], ["dispatch"]],
        "duties or taxes are not prepaid": [["duties", "taxes"], ["not prepaid", "not pre paid"]],
        "report within 7 days": [["report"], ["7"], ["day"]],
        "human review before approval": [["human"], ["review"], ["approval", "approved"]],
        "final sale does not block damaged-item review": [["final sale"], ["damaged"], ["review"]],
        "shipping to germany is not currently available": [["germany"], ["not currently available", "not available"]],
        "no lifetime warranty": [["no lifetime warranty", "does not offer a lifetime warranty"]],
        "bags have 2 years": [["bags"], ["2 years"]],
        "drinkware and travel accessories have 1 year": [["drinkware"], ["travel accessories", "packing cubes"], ["1 year"]],
        "migration note is not authoritative": [["migration"], ["not authoritative", "not a customer policy"]],
        "standard policy is 30 days unless a valid exception applies": [["30"], ["day"], ["standard"]],
        "the agent cannot approve a return": [["cannot", "can't"], ["approve", "approval"]],
        "the supplied information is insufficient": [["insufficient", "not enough information", "don't have information"]],
        "human confirmation": [["human"], ["confirmation", "confirm", "support"]],
        "one says hand-wash the body": [["hand-wash", "hand wash"], ["body"]],
        "one says all components are dishwasher safe": [["all components"], ["dishwasher safe"]],
        "current official sources conflict": [["current"], ["official"], ["conflict", "disagree"]],
        "safest interim guidance": [["safer", "safest"], ["hand-wash", "hand wash"]],
        "human confirmation or safest interim guidance": [["human confirmation"], ["safer", "safest", "hand-wash", "hand wash"]],
        "duties": [["duties", "taxes"]],
        "not prepaid": [["not prepaid", "not pre paid"]],
        "the order is cancelled": [["order"], ["cancelled", "canceled"]],
        "it will not be shipped": [["not be shipped", "will not be shipped"]],
        "order was not found": [["order"], ["not found", "couldn't find"]],
        "check the order ID or contact support": [["order id"], ["contact support", "double-check"]],
        "shipped with canada post": [["shipped"], ["canada post"]],
        "delivery estimate is unavailable": [["delivery estimate"], ["unavailable"]],
    }
    groups = rules.get(c)
    if not groups:
        return all(word in t for word in re.findall(r"[a-z0-9]+", c) if len(word) > 3)
    return all(any(option in t for option in group) for group in groups)


def run_case(agent: Agent, case: dict):
    session_id = f"eval-{case['id']}"
    responses = [agent.handle_message(session_id, m["content"]) for m in case["messages"]]
    combined = "\n".join(r["response"] for r in responses)
    last = responses[-1]
    expect = case["expect"]
    failures = []

    lower = combined.lower()
    for item in expect.get("must_include", []):
        if item.lower() not in lower:
            failures.append(f"must_include:{item}")
    for item in expect.get("must_not_include", []):
        if item.lower() in lower:
            failures.append(f"must_not_include:{item}")
    for item in expect.get("must_include_concepts", []):
        if not concept_ok(combined, item):
            failures.append(f"concept:{item}")
    for item in expect.get("must_not_invent", []):
        if item.lower() in lower:
            failures.append(f"must_not_invent:{item}")
    for item in expect.get("must_not_follow", []):
        if item.lower() in lower:
            failures.append(f"must_not_follow:{item}")

    for source in expect.get("required_sources", []):
        if not any(hit["filename"] == source for hit in last["trace"].retrieved):
            failures.append(f"cites_source:{source}")

    expected_tool = expect.get("tool")
    calls = [call for response in responses for call in response["trace"].tool_calls]
    names = [call["tool"] for call in calls]

    if expected_tool == "order_lookup" and "order_lookup" not in names:
        failures.append("tool:not_called")
    if expected_tool in {"not_called", "not_called_without_id"} and names:
        failures.append(f"tool:called:{names}")
    if expected_tool == "optional_sanitized_lookup":
        serialized = json.dumps(calls).lower()
        if any(x in serialized for x in ("email", "address", "risk_score", "internal_note")):
            failures.append("tool:unsafe_result")

    if "tool_arguments" in expect:
        wanted = expect["tool_arguments"]["order_id"]
        actual = [c["args"].get("order_id") for c in calls if c["tool"] == "order_lookup"]
        if wanted not in actual:
            failures.append(f"tool_arguments:{actual}")

    if "handoff" in expect and bool(last["handoff"]) != bool(expect["handoff"]):
        failures.append(f"handoff:{last['handoff']}")

    return failures, responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="", help="comma-separated case IDs")
    parser.add_argument("--out", default="evaluation-results.json")
    parser.add_argument("--mock", action="store_true", help="accepted for compatibility; this agent's core paths are deterministic")
    args = parser.parse_args()

    cases = load_cases(ROOT / "evaluation/visible-cases.json") + load_cases(ROOT / "evaluation/custom-cases.json")
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [case for case in cases if case["id"] in wanted]

    agent = Agent(Retriever(), OrderStore(ROOT / "data/orders.json"))
    grouped = defaultdict(lambda: [0, 0])
    output = []

    print(f"Running {len(cases)} of 21 available cases.")
    print("=" * 70)
    print("AI Support Agent Evaluation")
    print("=" * 70)

    for case in cases:
        failures, responses = run_case(agent, case)
        category = case["category"]
        grouped[category][1] += 1
        if not failures:
            grouped[category][0] += 1

        print(f"\n{category}  ({1 if not failures else 0}/1)")
        print(f"  [{'PASS' if not failures else 'FAIL'}] {case['id']}")
        for failure in failures:
            print(f"         - failed: {failure}")
        if failures:
            print(f"         response: {responses[-1]['response']!r}")

        output.append({
            "id": case["id"],
            "category": category,
            "passed": not failures,
            "failures": failures,
            "response": responses[-1]["response"],
            "handoff": responses[-1]["handoff"],
            "tool_calls": responses[-1]["trace"].tool_calls,
            "retrieved": responses[-1]["trace"].retrieved,
        })

    passed = sum(item["passed"] for item in output)
    print("\n----------------------------------------------------------------------")
    print(f"Total: {passed}/{len(output)}")
    print("----------------------------------------------------------------------")

    (ROOT / args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full results written to {args.out}")
    return 0 if passed == len(output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
