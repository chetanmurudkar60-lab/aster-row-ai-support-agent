from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

TOP_K = int(os.getenv("TOP_K", "8"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.16"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge-base"
ORDERS_PATH = ROOT_DIR / "data" / "orders.json"
EVAL_PATH = ROOT_DIR / "evaluation" / "visible-cases.json"

# Backward-compatible names used by the original starter-era tests.
GEMINI_API_KEY = ""
EMBEDDING_MODEL = ""
VECTOR_DIR = ROOT_DIR / "storage" / "vectors"
TRACE_DIR = ROOT_DIR / "storage" / "traces"
