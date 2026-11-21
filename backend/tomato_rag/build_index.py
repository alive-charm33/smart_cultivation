"""Chunk the markdown knowledge base by '## ' section and index it in Chroma.
Run:  python build_index.py
"""
import re, json
from pathlib import Path
import yaml

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "tomato_kb"
MAX_CHARS = 1400  # about 300-350 tokens; longer sections are split on blank lines


def parse_md(path: Path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError(f"{path} has no YAML front matter")
    return yaml.safe_load(m.group(1)), m.group(2)


def split_sections(body: str):
    parts = re.split(r"^## +(.+)$", body, flags=re.M)
    for i in range(1, len(parts), 2):
        yield parts[i].strip(), parts[i + 1].strip()


def split_long(text: str, max_chars: int = MAX_CHARS):
    """Split on blank lines, keeping tables/bullet blocks together."""
    blocks, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > max_chars:
            blocks.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        blocks.append(cur.strip())
    return blocks


def flat(v):
    """Chroma metadata must be str/int/float/bool."""
    if isinstance(v, (list, tuple)):
        return ",".join(map(str, v))
    return "" if v is None else v if isinstance(v, (int, float, bool)) else str(v)


def build_chunks():
    chunks = []
    for path in sorted(DOCS.rglob("*.md")):
        meta, body = parse_md(path)
        title = meta.get("common_name") or meta.get("stage_name") or meta.get("doc_id") or path.stem
        base = {k: flat(meta.get(k)) for k in
                ("doc_type", "disease_id", "stage_id", "doc_id", "class_label", "risk_stages",
                 "diseases_at_risk", "sources", "bbch")}
        for k in ("day_from", "day_to", "day_reference"):
            if k in meta:
                base[k] = meta[k]
        base["source_file"] = str(path.relative_to(DOCS))
        base["title"] = title
        for section, content in split_sections(body):
            for n, piece in enumerate(split_long(content)):
                slug = re.sub(r"\W+", "_", section.lower()).strip("_")
                chunks.append({
                    "id": f"{path.stem}::{slug}::{n}",
                    "text": f"{title} - {section}\n{piece}",
                    "metadata": {**base, "section": section},
                })
    return chunks


def main():
    import chromadb
    from common import Embedder
    chunks = build_chunks()
    print(f"{len(chunks)} chunks from {len(list(DOCS.rglob('*.md')))} documents")
    emb = Embedder()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=emb.embed_docs([c["text"] for c in chunks]))
    print(f"Indexed into {DB_DIR} (embedder: {emb.name})")


if __name__ == "__main__":
    main()
