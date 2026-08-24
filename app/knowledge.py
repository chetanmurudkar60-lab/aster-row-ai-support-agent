from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import Chunk


FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(text)
    if not match:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata, text[match.end():]


def split_into_chunks(path: Path) -> list[Chunk]:
    metadata, body = parse_markdown(path)
    filename = path.name

    sections: list[tuple[str, list[str]]] = []
    current_heading = path.stem
    current: list[str] = []

    for line in body.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            if current:
                sections.append((current_heading, current))
            current_heading = match.group(1).strip()
            current = []
        elif re.match(r"^#\s+", line):
            continue
        else:
            current.append(line)

    if current:
        sections.append((current_heading, current))

    chunks: list[Chunk] = []
    for section_index, (heading, lines) in enumerate(sections):
        content = "\n".join(lines).strip()
        if not content:
            continue

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", content)
            if p.strip()
        ]

        # Preserve normal policy sections as a single chunk. Very long sections
        # are split only at paragraph boundaries.
        if len(content) <= 1800:
            paragraphs = [content]

        for part_index, part in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}:{section_index}:{part_index}",
                    content=part,
                    filename=filename,
                    heading=heading,
                    metadata=dict(metadata),
                )
            )

    return chunks


def load_all_chunks(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(split_into_chunks(path))
    return chunks


def is_customer_authoritative(chunk: Chunk) -> bool:
    metadata = chunk.metadata
    return (
        metadata.get("status") == "active"
        and metadata.get("audience") == "customer"
        and metadata.get("policy_authority") == "official"
        and metadata.get("customer_answering", True) is not False
    )


def authority_bonus(chunk: Chunk) -> float:
    metadata = chunk.metadata
    score = 0.0

    if metadata.get("status") == "active":
        score += 0.22
    elif metadata.get("status") == "superseded":
        score -= 0.40
    elif metadata.get("status") == "draft":
        score -= 0.35

    if metadata.get("audience") == "customer":
        score += 0.12
    elif metadata.get("audience") == "internal":
        score -= 0.22

    if metadata.get("policy_authority") == "official":
        score += 0.16
    else:
        score -= 0.14

    if metadata.get("customer_answering") is False:
        score -= 0.35

    return score
