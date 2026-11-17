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


_DTYPES = {"U8": "u1", "I32": "<i4", "U32": "<u4", "U64": "<u8", "F32": "<f4", "F16": "<f2", "I64": "<i8"}


def open_tensors(path):
    """Memory-maps every tensor of a safetensors file (8-byte header length, JSON header, raw data).
    Returns (name -> array, metadata). Nothing is read until it is used."""
    import json

    import numpy as np

    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        shape = tuple(info["shape"])
        out[name] = np.memmap(path, dtype=_DTYPES[info["dtype"]], mode="r", offset=8 + n + a, shape=shape) if b > a else np.zeros(shape, dtype=_DTYPES[info["dtype"]])
    return out, header.get("__metadata__") or {}


class Store:
    """Read access. Everything is memory-mapped: opening the store costs no RAM, a lookup touches a few pages."""

    def __init__(self, path):
        import numpy as np

        t, self.meta = open_tensors(path)
        self.t = t
        self.rank_tok, self.blob = t["rank_tok"], t["blob"]
        self.zdict = {code: t[f"zdict_{lang}"].tobytes() for lang, code in LANGS.items() if f"zdict_{lang}" in t}
        self.off, self.size, self.qid, self.lang = t["off"], t["size"], t["qid"], t["lang"]
        self.surf_hash, self.surf_id, self.surf_exact = t["surf_hash"], t["surf_id"], t["surf_exact"]
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
        qid = self._qid_sorted.dtype.type(qid)  # same type as the array: otherwise numpy converts the whole array at each call
        a, b = np.searchsorted(self._qid_sorted, qid, "left"), np.searchsorted(self._qid_sorted, qid, "right")
        ids = [int(self._qid_order[k]) for k in range(a, b) if int(self.lang[self._qid_order[k]]) == LANGS[lang]]
        return max(ids, key=lambda i: int(self.size[i])) if ids else None
