import json
import pickle
from pathlib import Path

from src.utils import concat_sparse, sparse_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "sparses_embedding.pkl"


def load_jsonl_dict(path: Path, id_field: str = "_id") -> dict[str, dict]:
    items: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = data.get(id_field)
            if key is not None:
                items[str(key)] = data
    return items


def collect_texts():
    queries = load_jsonl_dict(DATA_DIR / "queries.jsonl")
    corpus = load_jsonl_dict(DATA_DIR / "corpus.jsonl")

    ids: list[str] = []
    embeddings: list[dict[str, int]] = []
    seen_ids: set[str] = set()

    for qid, payload in queries.items():
        text = payload.get("text", "").strip()
        if not text or qid in seen_ids:
            continue
        ids.append(qid)
        embeddings.append(sparse_embedding(text))
        seen_ids.add(qid)

    for cid, payload in corpus.items():
        if cid in seen_ids:
            continue
        title = (payload.get("title") or payload.get("text", "")).strip()
        if not title:
            continue
        ids.append(cid)
        embeddings.append(sparse_embedding(title))
        seen_ids.add(cid)

    if not ids:
        raise RuntimeError("No texts available to embed.")

    return ids, embeddings


def main():
    ids, embeddings = collect_texts()
    ids, vocab, matrix = concat_sparse(ids, embeddings)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump({"ids": ids, "vocab": vocab, "matrix": matrix}, f)

    print(f"Saved {len(ids)} embeddings with vocab size {len(vocab)} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
