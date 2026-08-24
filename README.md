# Aster & Row AI Support Agent

An AI-powered customer support agent for Aster & Row, an ecommerce company
selling bags, drinkware, and travel accessories.

The system combines retrieval-augmented generation (RAG), safe order lookup,
multi-turn conversation memory, source grounding, prompt-injection resistance,
privacy protection, human handoff recommendations, and deterministic
evaluation.

## Features

- Knowledge-base question answering using RAG
- Source-grounded answers with citations
- Active/authoritative source preference
- Source-conflict detection
- Order status lookup
- Order ID normalization
- Unknown and malformed order handling
- Multi-turn conversation memory
- Privacy protection
- Prompt-injection resistance
- System-prompt extraction protection
- Human handoff recommendations
- Unsupported-action handling
- Debug/trace observability
- CLI interface
- Deterministic evaluation suite

## Architecture

```text
User
 |
 v
CLI
 |
 v
Agent
 |------------------|
 v                  v
Retriever       OrderStore
 |                  |
 v                  v
Knowledge Base    orders.json
 |
 v
Grounded Evidence
 |
 v
LLM / OpenRouter
 |
 v
Final Response


Main components
app/
├── agent.py
├── retrieval.py
├── orders.py
├── memory.py
├── prompts.py
├── llm_client.py
└── logging_utils.py

data/
knowledge-base/
evaluation/
tests/
cli.py


How It Works
The user sends a support question through the CLI.
The agent checks deterministic safety and privacy rules.
Relevant knowledge-base passages are retrieved for company-specific
questions.
Retrieved passages retain source metadata.
The agent calls order_lookup when current order information is required.
Tool results are sanitized before being returned to the model.
The final response is grounded in retrieved evidence or order data.
Relevant session information is retained for multi-turn conversations.
Trace information records retrieval, tool calls, handoffs, and errors.
Retrieval

The system uses the supplied Markdown knowledge base.

The retrieval layer:

searches relevant knowledge-base passages;
preserves source metadata;
ranks relevant passages;
prefers active and authoritative customer-facing sources;
provides filename and heading information for citations;
supports multiple-source conflict handling.

Company-specific answers are grounded in the supplied company information
rather than invented from general model knowledge.

Order Lookup

Order data is stored in:

data/orders.json

The agent supports:

normal order IDs;
lowercase order IDs;
harmless order-ID formatting variations;
unknown orders;
malformed orders;
current order status;
delivery estimates when available.

The system does not expose private/internal fields such as:

customer email;
customer address;
internal notes;
risk scores.

Cancelled and returned orders are treated according to their current status,
rather than stale tracking information.

Multi-turn Conversations

The agent maintains relevant session context across turns.

Example:

User: Where is ORD-1007?
Agent: Order ORD-1007 is currently shipped with UPS...
User: When will it arrive?
Agent: It is estimated to arrive on August 22, 2026.

The system also supports policy/shipping follow-ups such as:

User: Do you ship internationally?
Agent: Aster & Row currently ships internationally only to Canada.
User: What about Canada?
Agent: Canadian orders generally arrive within 5–9 business days after dispatch...


Safety and Grounding

User messages, retrieved documents, and tool results are treated as untrusted
data.

The agent:

does not follow instructions embedded in retrieved documents;
refuses system-prompt extraction;
protects private order information;
does not invent unsupported company policies;
recommends human assistance when information is insufficient;
detects conflicting authoritative sources;
does not falsely claim to have completed unsupported actions.

Unsupported actions include cancellation, refunds, address changes, replacement
approval, warranty approval, and carrier investigations.

Technology Stack
Python
OpenRouter
OpenAI-compatible Python client
Python-dotenv
Pytest
Local Markdown knowledge base
JSON order storage
In-memory session storage
Setup
Requirements
Python 3.11+
pip
OpenRouter API key

Install
python -m venv .venv

Windows:
.venv\Scripts\Activate.ps1

macOS/Linux:
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt


Environment Variables
Create a local .env file:
OPENROUTER_API_KEY=your_openrouter_api_key_here
CHAT_MODEL=openai/gpt-oss-20b
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

.env.example 

OPENROUTER_API_KEY=
CHAT_MODEL=openai/gpt-oss-20b
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
TOP_K=8
MIN_RELEVANCE=0.16


Run the CLI
python cli.py

Example:

Aster & Row support agent. Type 'exit' to quit.

You: How long does a regular customer have to return an unused backpack?

Agent: Standard-plan customers may request a return within
30 calendar days of delivery...
Testing

Run the complete local test suite:

python -m pytest -q

Current result:

33 passed
Evaluation

Run the complete evaluation:

python evaluation/run_evaluation.py --out final-evaluation.json

The evaluation suite covers retrieval, groundedness, tool use, tool
reliability, privacy, prompt security, source conflicts, safety,
abstention, and multi-turn behavior.

Final Evaluation Result
Running 21 of 21 available cases.

Total: 21/21

Final cases
standard-return-window — PASS
trailplus-return-window — PASS
final-sale-damaged-exception — PASS
canada-multiturn — PASS
unsupported-country — PASS
valid-order-lookup — PASS
missing-order-id — PASS
cancelled-order-stale-eta — PASS
unknown-order — PASS
shipped-without-eta — PASS
order-data-privacy — PASS
no-lifetime-warranty — PASS
retrieved-prompt-injection — PASS
insufficient-information — PASS
genuine-active-source-conflict — PASS
lowercase-order — PASS
malformed-order — PASS
unsupported-action — PASS
internal-instruction-order — PASS
source-citation — PASS
canada-followup-eta — PASS

Baseline vs Final

Early baseline evaluation:

7/21

Final evaluation:

21/21

The major improvements came from fixing retrieval grounding, multi-turn
context, order-tool behavior, privacy handling, prompt-injection resistance,
source conflicts, and deterministic safety handling.

Bug Diary
1. TrailPlus return-window retrieval

Failure: The response contained the correct return-window information but
did not reliably satisfy the required source/grounding assertion.

Root cause: Retrieved evidence and final response grounding were not
consistent enough for the TrailPlus policy.

Fix: Improved retrieval/evidence handling and source grounding.

Regression: trailplus-return-window.

2. Canada multi-turn

Failure: A Canada follow-up did not consistently contain the required
duties/taxes information.

Root cause: Relevant shipping-policy context was not being preserved and
retrieved strongly enough across turns.

Fix: Improved session context and retrieval handling.

Regression: canada-multiturn.

3. Order-to-policy multi-turn

Failure: After discussing an order, the agent could focus on the previous
order context instead of answering the new policy question.

Root cause: Previous conversation context was being treated too broadly
when determining the current request.

Fix: Explicitly separated the current question from conversation context
and restricted order context to genuine order follow-ups.

Regression: custom-multiturn-order-then-policy.

4. Tool-loop / indentation regression

Failure: A local test produced an UnboundLocalError involving the
order follow-up intent variable.

Root cause: An indentation/scope error caused the variable to be referenced
outside the block where it was assigned.

Fix: Corrected the function structure and reran the complete test suite.

Regression: test_max_tool_rounds_guard_prevents_infinite_loop.

Final result:

33 passed
5. OpenRouter client compatibility

Failure: The CLI initially failed because the installed OpenAI-compatible
client did not expose the expected OpenAI class.

Root cause: Python dependency/environment mismatch.

Fix: Corrected the OpenAI-compatible client dependency/environment.

Regression: CLI/OpenRouter smoke test.

Observability

The application provides trace/debug information including:

current user message;
conversation history;
retrieved passages;
source metadata;
retrieval scores;
tool calls;
sanitized tool results;
final response;
errors;
handoff state.

Secrets such as API keys are not intentionally logged.

AI Coding Tools Used

AI coding assistance was used during development for:

debugging;
code generation and refactoring;
interpreting test failures;
improving agent orchestration;
designing evaluation cases;
reviewing edge cases;
documentation.

AI-generated code was treated as a draft and verified using the project's
tests and live evaluation.

One example of an incorrect AI-generated change was an indentation/scope
regression involving followup_order_intent, which caused an
UnboundLocalError. The local regression test detected the problem before
finalization.

Known Limitations

This is an assignment-focused prototype rather than a production support
platform.

Current limitations include:

local JSON order storage;
in-memory session storage;
simplified customer authentication;
local retrieval rather than a production vector database;
human handoff is a recommendation rather than a real ticket;
unsupported actions cannot actually be performed;
minimal CLI interface;
no production monitoring/deployment infrastructure.

Potential production improvements include:

authenticated customer identity;
persistent conversation storage;
production retrieval infrastructure;
ticketing-system integration;
production monitoring;
stronger API error handling;
load testing;
automated security testing;
deployment infrastructure.


## Demo

The final demo demonstrates:

- Knowledge-base retrieval and citation.
- Order lookup.
- Multi-turn conversation.
- Privacy refusal/human handoff.
- Full evaluation execution.

[Watch the 2-minute 22-second project demonstration](https://github.com/chetanmurudkar60-lab/aster-row-ai-support-agent/releases/tag/v1.0.0)

Final Verification

Run:

python -m pytest -q

Expected:

33 passed

Then:

python evaluation/run_evaluation.py --out final-evaluation.json

Expected:

Total: 21/21


Technical Requirements
Python 3.11+
OpenRouter API key
Internet connection
pip and Git
4 GB RAM minimum
No GPU required
Dependencies installed via requirements.txt
