"""The note store shipped with the integrated model: one safetensors file, read by memory mapping.

Goal: as many articles and as much text as possible in about 3 GB. What makes it compact:
  - notes are stored as tokens, not text: no tokenizer call at run time, and the truncation is exact
  - each token is written as its frequency rank in a variable-length code (frequent tokens take one byte),
    then the note is compressed with zlib primed with a dictionary of typical notes
  - names are indexed by a 64-bit hash in a sorted array (13 bytes per name) instead of a text index
  - no per-row database overhead: one byte blob and an offset array

Tensors
    tok_rank [V] int32      token id -> rank          rank_tok [V] int32   rank -> token id
    zdict_en, zdict_fr [~32k] uint8   zlib preset dictionaries, one per language
    blob [...] uint8        compressed notes, concatenated
    off [N+1] uint64        note i is blob[off[i]:off[i+1]]
    size [N] uint8          round(8 * log2(article size)), the prior used by the choice head
    qid [N] uint32          Wikidata id (0 if unknown)
    lang [N] uint8          0 English, 1 French
    surf_hash [M] uint64    sorted hashes of the normalised names;  surf_id [M] uint32;  surf_exact [M] uint8

A note, before compression, is a sequence of varints:
    n_head, head ranks...   tokens of " {title} ({description})"
    n_lead, lead ranks...   start of the summary (up to LEAD_STORE tokens)
    facts ranks...          "\\nFacts:\\n" + infobox (up to FACTS_STORE tokens), possibly empty"""
import hashlib
import zlib

LEAD_STORE, FACTS_STORE = 192, 256  # stored; the model reads the first LEAD_TOKENS / INFO_TOKENS of them
LANGS = {"en": 0, "fr": 1}


def name_hash(surface: str) -> int:
    return int.from_bytes(hashlib.blake2b(surface.encode("utf-8"), digest_size=8).digest(), "little")


def varint_encode(values) -> bytes:
    out = bytearray()
    for v in values:
        v = int(v)
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def varint_decode(data: bytes):
    out, cur, shift = [], 0, 0
    for b in data:
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            out.append(cur)
            cur, shift = 0, 0
    return out


def encode_note(head, lead, facts, tok_rank, zdict) -> bytes:
    raw = varint_encode([len(head)] + [tok_rank[t] for t in head] + [len(lead)] + [tok_rank[t] for t in lead] + [tok_rank[t] for t in facts])
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_DEFAULT_STRATEGY, zdict)
    return c.compress(raw) + c.flush()


def decode_note(data: bytes, rank_tok, zdict):
    """Returns (head, lead, facts) token id lists."""
    d = zlib.decompressobj(-15, zdict)
    vals = varint_decode(d.decompress(bytes(data)) + d.flush())
    n_head = vals[0]
    head = [int(rank_tok[r]) for r in vals[1 : 1 + n_head]]
    n_lead = vals[1 + n_head]
    lead = [int(rank_tok[r]) for r in vals[2 + n_head : 2 + n_head + n_lead]]
    facts = [int(rank_tok[r]) for r in vals[2 + n_head + n_lead :]]
    return head, lead, facts


class Store:
    """Read access. Everything is memory-mapped: opening a 3 GB store costs no RAM."""

    def __init__(self, path):
        import numpy as np
        from safetensors import safe_open

        self.f = safe_open(path, framework="np")
        g = self.f.get_tensor
        self.rank_tok = g("rank_tok")
        self.zdict = {code: g(f"zdict_{lang}").tobytes() for lang, code in LANGS.items() if f"zdict_{lang}" in self.f.keys()}
        self.off, self.size, self.qid, self.lang = g("off"), g("size"), g("qid"), g("lang")
        self.surf_hash, self.surf_id, self.surf_exact = g("surf_hash"), g("surf_id"), g("surf_exact")
        self.blob = self.f.get_slice("blob")
        self._qid_order = None
        self._np = np

    def __len__(self):
        return len(self.size)

    def note(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return decode_note(self.blob[a:b].tobytes(), self.rank_tok, self.zdict[int(self.lang[i])])

    def ids_for(self, surface, lang=None):
        """(article id, exact flag) of every article answering to this normalised name, most developed first."""
        np = self._np
        h = np.uint64(name_hash(surface))
        a, b = np.searchsorted(self.surf_hash, h, "left"), np.searchsorted(self.surf_hash, h, "right")
        rows = [(int(self.surf_id[k]), int(self.surf_exact[k])) for k in range(a, b)]
        if lang is not None:
            rows = [r for r in rows if int(self.lang[r[0]]) == LANGS[lang]]
        return sorted(set(rows), key=lambda r: -int(self.size[r[0]]))

    def by_qid(self, qid: int, lang="en"):
        np = self._np
        if self._qid_order is None:
            self._qid_order = np.argsort(self.qid, kind="stable")
            self._qid_sorted = self.qid[self._qid_order]
        a, b = np.searchsorted(self._qid_sorted, qid, "left"), np.searchsorted(self._qid_sorted, qid, "right")
        ids = [int(self._qid_order[k]) for k in range(a, b) if int(self.lang[self._qid_order[k]]) == LANGS[lang]]
        return max(ids, key=lambda i: int(self.size[i])) if ids else None


# ---------------------------------------------------------------------------- build (CPU)
_W = {}
HEAD_STORE = 48


def _fields(tok, rows):
    """rows of (title, description, lead, infobox) -> three lists of token lists, cut to the stored lengths."""
    enc = lambda texts: tok(texts, add_special_tokens=False)["input_ids"]
    heads = enc([f" {r[0]} ({r[1]})" if r[1] else f" {r[0]}" for r in rows])
    leads = enc([r[2] or "" for r in rows])
    facts = enc([("\nFacts:\n" + r[3]) if r[3] else "" for r in rows])
    return [h[:HEAD_STORE] for h in heads], [l[:LEAD_STORE] for l in leads], [f[:FACTS_STORE] for f in facts]


def _worker_init(model_key, tok_rank, zdicts, tok=None):
    from .common import load_tokenizer

    _W["tok"], _W["rank"], _W["zdicts"] = tok or load_tokenizer(model_key), [int(x) for x in tok_rank], zdicts


def _encode_range(args):
    """Encodes the articles of one id range of one source database."""
    import sqlite3

    lang, db_path, lo, hi = args
    rows = sqlite3.connect(db_path).execute("SELECT title, description, lead, infobox, id, size, qid FROM articles WHERE id >= ? AND id < ?", (lo, hi)).fetchall()
    if not rows:
        return []
    heads, leads, facts = _fields(_W["tok"], rows)
    zdict = _W["zdicts"][lang]
    return [(r[4], encode_note(h, l, f, _W["rank"], zdict), r[5], r[6], len(h), len(l), len(f)) for r, h, l, f in zip(rows, heads, leads, facts)]


def build(sources, out_path: str, model_key: str = "instruct", workers: int = 14, chunk: int = 5000, limit: int = 0, tok=None):
    """sources: list of (lang, sqlite path) in the notes3 format. Writes the safetensors store.
    limit > 0 keeps only the first `limit` article ids of each source (dry run). workers <= 1 runs in-process."""
    import math
    import os
    import random
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    from safetensors.numpy import save_file

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
        fields = _fields(tok, [r for r in rows if r])
        for part in fields:
            np.add.at(counts, np.fromiter((t for seq in part for t in seq), dtype=np.int64), 1)
        sampled[lang] = [(h, l, f) for h, l, f in zip(*fields)][:600]
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
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(model_key, tok_rank, zdicts)) if workers > 1 else None
    if pool is None:
        _worker_init(model_key, tok_rank, zdicts, tok)
    for lang, path in sources:
        db, n = sqlite3.connect(path), n_ids(path)
        tasks = [(lang, path, lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
        local = {}
        for part in pool.map(_encode_range, tasks) if pool else map(_encode_range, tasks):
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
        "articles": n, "by_lang": {lang: int(sum(1 for x in langs if x == code)) for lang, code in LANGS.items()}, "names": len(surf_hash),
        "gb": round(total / 1e9, 3), "notes_gb": round(len(blob) / 1e9, 3), "notes_gb_by_lang": {k: round(v / 1e9, 3) for k, v in by_lang_bytes.items()},
        "bytes_per_token": round(len(blob) / max(tokens, 1), 3), "bytes_per_article": round(total / max(n, 1), 1),
        "tokens": tokens, "tokens_per_article": {k: round(v / max(n, 1), 1) for k, v in n_tok.items()},
        "share_lead_at_cap": round(full["lead"] / max(n, 1), 3), "share_facts_at_cap": round(full["facts"] / max(n, 1), 3), "share_with_facts": round(full["with_facts"] / max(n, 1), 3),
    }
