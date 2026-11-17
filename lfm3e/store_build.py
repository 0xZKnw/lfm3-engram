"""Builds the note store (see store.py for the format) from the notes3 databases. CPU only."""
from .store import FACTS_STORE, LANGS, LEAD_STORE, encode_note, name_hash, varint_encode

_W = {}
HEAD_STORE = 48


def fields(tok, rows):
    """rows of (title, description, lead, infobox) -> three lists of token lists, cut to the stored lengths."""
    enc = lambda texts: tok(texts, add_special_tokens=False)["input_ids"]
    heads = enc([f" {r[0]} ({r[1]})" if r[1] else f" {r[0]}" for r in rows])
    leads = enc([r[2] or "" for r in rows])
    facts = enc([("\nFacts:\n" + r[3]) if r[3] else "" for r in rows])
    return [h[:HEAD_STORE] for h in heads], [l[:LEAD_STORE] for l in leads], [f[:FACTS_STORE] for f in facts]


def _worker_init(model_key, tok_rank, zdicts, tok=None):
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # one process per core already
    from .common import load_tokenizer

    _W["tok"], _W["rank"], _W["zdicts"] = tok or load_tokenizer(model_key), [int(x) for x in tok_rank], zdicts


def _encode_range(args):
    """Encodes the articles of one id range of one source database."""
    import sqlite3

    lang, db_path, lo, hi, skip = args
    rows = sqlite3.connect(db_path).execute("SELECT title, description, lead, infobox, id, size, qid FROM articles WHERE id >= ? AND id < ?", (lo, hi)).fetchall()
    rows = [r for r in rows if r[4] not in skip]
    if not rows:
        return []
    heads, leads, facts = fields(_W["tok"], rows)
    zdict = _W["zdicts"][lang]
    return [(r[4], encode_note(h, l, f, _W["rank"], zdict), r[5], r[6], len(h), len(l), len(f)) for r, h, l, f in zip(rows, heads, leads, facts)]


def build(sources, out_path: str, model_key: str = "instruct", workers: int = 14, chunk: int = 5000, limit: int = 0, tok=None):
    """sources: list of (lang, sqlite path) in the notes3 format. Writes the safetensors store.
    limit > 0 keeps only the first `limit` article ids of each source (dry run). workers <= 1 runs in-process."""
    import math
    import multiprocessing
    import os
    import random
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    from safetensors.numpy import save_file

    given = tok  # a stand-in tokenizer (tests) is handed to the workers; the real one is loaded by each of them
    if tok is None:
        from .common import load_tokenizer

        tok = load_tokenizer(model_key)
    vocab = len(tok)

    def n_ids(path):
        n = sqlite3.connect(path).execute("SELECT MAX(id) FROM articles").fetchone()[0] + 1
        return min(n, limit) if limit else n

    # 1. token frequencies over all sources, then one zlib dictionary per language, from a sample of each
    counts = np.ones(vocab, dtype=np.int64)
    sampled = {}
    for lang, path in sources:
        db, n = sqlite3.connect(path), n_ids(path)
        ids = random.Random(0).sample(range(n), min(60_000, n))
        rows = [db.execute("SELECT title, description, lead, infobox FROM articles WHERE id=?", (i,)).fetchone() for i in ids]
        parts = fields(tok, [r for r in rows if r])
        for part in parts:
            np.add.at(counts, np.fromiter((t for seq in part for t in seq), dtype=np.int64), 1)
        sampled[lang] = list(zip(*parts))[:600]
    rank_tok = np.argsort(-counts, kind="stable").astype(np.int32)
    tok_rank = np.empty(vocab, dtype=np.int32)
    tok_rank[rank_tok] = np.arange(vocab, dtype=np.int32)
    rank = [int(x) for x in tok_rank]
    zdicts = {}
    for lang, notes in sampled.items():  # typical notes, uncompressed: zlib can point into the last 32 KB of it
        raw = b"".join(varint_encode([len(h)] + [rank[t] for t in h] + [len(l)] + [rank[t] for t in l] + [rank[t] for t in f]) for h, l, f in notes)
        zdicts[lang] = raw[-32768:]

    # 2. encode every article
    blobs, sizes, qids, langs = [], [], [], []
    n_tok = {"head": 0, "lead": 0, "facts": 0}
    full = {"lead": 0, "facts": 0, "with_facts": 0}
    surf_hash, surf_id, surf_exact = [], [], []
    duplicates = 0
    # spawn, not fork: the tokenizer has already run threads in this process, and a forked copy of them deadlocks
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=(model_key, tok_rank, zdicts, given)) if workers > 1 else None
    if pool is None:
        _worker_init(model_key, tok_rank, zdicts, tok)
    for lang, path in sources:
        db, n = sqlite3.connect(path), n_ids(path)
        best = {}  # the export holds several versions of some articles: keep the most developed one per title
        for aid, title, size in db.execute("SELECT id, title, size FROM articles WHERE id < ?", (n,)):
            if title not in best or size > best[title][0]:
                best[title] = (size, aid)
        keep = {aid for _, aid in best.values()}
        duplicates += n - len(keep)
        del best
        tasks = [(lang, path, lo, min(lo + chunk, n), frozenset(i for i in range(lo, min(lo + chunk, n)) if i not in keep)) for lo in range(0, n, chunk)]
        del keep
        local = {}
        for k, part in enumerate(pool.map(_encode_range, tasks) if pool else map(_encode_range, tasks)):
            if k % 100 == 0:
                print(f"[store] {lang}: {k}/{len(tasks)} chunks, {len(blobs):,} notes", flush=True)
            for aid, data, size, qid, nh, nl, nf in part:
                local[aid] = len(blobs)
                blobs.append(data)
                sizes.append(min(255, round(8 * math.log2(max(size or 1, 1)))))
                q = qid[1:] if qid and qid[0] == "Q" else ""
                qids.append(int(q) if q.isdigit() and int(q) < 2**32 else 0)
                langs.append(LANGS[lang])
                n_tok["head"] += nh
                n_tok["lead"] += nl
                n_tok["facts"] += nf
                full["lead"] += nl >= LEAD_STORE
                full["facts"] += nf >= FACTS_STORE
                full["with_facts"] += nf > 0
        for surface, aid, exact in db.execute("SELECT surface, id, exact FROM titles"):
            if aid in local:
                surf_hash.append(name_hash(surface))
                surf_id.append(local[aid])
                surf_exact.append(exact)
    if pool:
        pool.shutdown()

    n = len(blobs)
    off = np.zeros(n + 1, dtype=np.uint64)
    off[1:] = np.cumsum([len(b) for b in blobs], dtype=np.uint64)
    by_lang_bytes = {lang: int(sum(len(b) for b, x in zip(blobs, langs) if x == code)) for lang, code in LANGS.items()}
    blob = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    del blobs
    order = np.argsort(np.asarray(surf_hash, dtype=np.uint64), kind="stable")
    tensors = {
        "tok_rank": tok_rank, "rank_tok": rank_tok, "blob": blob, "off": off,
        "size": np.asarray(sizes, dtype=np.uint8), "qid": np.asarray(qids, dtype=np.uint32), "lang": np.asarray(langs, dtype=np.uint8),
        "surf_hash": np.asarray(surf_hash, dtype=np.uint64)[order], "surf_id": np.asarray(surf_id, dtype=np.uint32)[order], "surf_exact": np.asarray(surf_exact, dtype=np.uint8)[order],
    }
    for lang, z in zdicts.items():
        tensors[f"zdict_{lang}"] = np.frombuffer(z, dtype=np.uint8)
    save_file(tensors, out_path, metadata={"lead_store": str(LEAD_STORE), "facts_store": str(FACTS_STORE), "tokenizer": model_key, "source": "Wikipedia (CC BY-SA 4.0) via wikimedia/structured-wikipedia 20240916"})
    total, tokens = os.path.getsize(out_path), sum(n_tok.values())
    return {
        "articles": n, "duplicates_dropped": duplicates, "by_lang": {lang: int(sum(1 for x in langs if x == code)) for lang, code in LANGS.items()}, "names": len(surf_hash),
        "gb": round(total / 1e9, 3), "notes_gb": round(len(blob) / 1e9, 3), "notes_gb_by_lang": {k: round(v / 1e9, 3) for k, v in by_lang_bytes.items()},
        "bytes_per_token": round(len(blob) / max(tokens, 1), 3), "bytes_per_article": round(total / max(n, 1), 1),
        "tokens": tokens, "tokens_per_article": {k: round(v / max(n, 1), 1) for k, v in n_tok.items()},
        "share_lead_at_cap": round(full["lead"] / max(n, 1), 3), "share_facts_at_cap": round(full["facts"] / max(n, 1), 3), "share_with_facts": round(full["with_facts"] / max(n, 1), 3),
    }
