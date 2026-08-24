import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from app.config import KNOWLEDGE_BASE_DIR
from app.knowledge import load_all_chunks

chunks=load_all_chunks(KNOWLEDGE_BASE_DIR)
print(f"Indexed {len(chunks)} chunks from {len(list(KNOWLEDGE_BASE_DIR.glob('*.md')))} documents.")
for c in chunks:
    print(f"- {c.filename} :: {c.heading}")
