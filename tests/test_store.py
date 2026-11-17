"""Local test of the note store: build from two small databases with a stand-in tokenizer, then read back.
Run: .venv/bin/python tests/test_store.py"""
import os
import random
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfm3e import store


class FakeTok:
    """Words -> ids. Enough to exercise the build; the real sizes are measured with the real tokenizer."""

    def __init__(self, size=20000):
        self.size, self.ids, self.words = size, {}, []

    def __len__(self):
        return self.size

    def one(self, text):
        out = []
        for w in text.replace("\n", " \n ").split(" "):
            if w and w not in self.ids:
                self.ids[w] = len(self.words)
                self.words.append(w)
            if w:
                out.append(self.ids[w])
        return out

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.one(t) for t in ([texts] if isinstance(texts, str) else texts)]}


def make_db(path, n, rng, words):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, description TEXT, lead TEXT, qid TEXT, infobox TEXT, size INTEGER)")
    db.execute("CREATE TABLE titles (surface TEXT, id INTEGER, exact INTEGER)")
    rows = []
    for i in range(n):
        title = f"{rng.choice(words).title()} {i % 7}"
        desc = "" if i % 5 == 0 else " ".join(rng.choices(words, k=4))
        lead = " ".join(rng.choices(words, k=rng.choice([30, 120, 400])))
        info = "" if i % 3 == 0 else "\n".join(f"{rng.choice(words)}: {rng.choice(words)}" for _ in range(rng.choice([4, 300])))
        qid = "" if i % 11 == 0 else f"Q{1000 + i}"
        rows.append((i, title, desc, lead, qid, info, rng.randrange(200, 400000)))
        db.execute("INSERT INTO titles VALUES (?,?,?)", (title.lower(), i, 1))
        db.execute("INSERT INTO titles VALUES (?,?,?)", (title.lower().split()[0], i, 0))
    db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", rows)
    db.commit()
    return rows


def main():
    rng = random.Random(1)
    words = [f"w{i}" for i in range(3000)]
    d = tempfile.mkdtemp()
    en, fr = make_db(f"{d}/en.db", 700, rng, words), make_db(f"{d}/fr.db", 300, rng, words)
    tok = FakeTok()
    out = store.build([("en", f"{d}/en.db"), ("fr", f"{d}/fr.db")], f"{d}/m.safetensors", workers=1, chunk=128, tok=tok)
    assert out["articles"] == 1000 and out["by_lang"] == {"en": 700, "fr": 300} and out["names"] == 2000, out
    s = store.Store(f"{d}/m.safetensors")
    assert len(s) == 1000
    for base, rows, lang in ((0, en, "en"), (700, fr, "fr")):
        for r in rng.sample(rows, 120):
            i, title, desc, lead, qid, info, size = r
            head, lead_t, facts = s.note(base + i)
            assert head == tok.one(f" {title} ({desc})" if desc else f" {title}")[:48]
            assert lead_t == tok.one(lead)[: store.LEAD_STORE]
            assert facts == (tok.one("\nFacts:\n" + info)[: store.FACTS_STORE] if info else [])
            assert (base + i, 1) in s.ids_for(title.lower(), lang) and (base + i, 1) not in s.ids_for(title.lower(), "fr" if lang == "en" else "en")
            assert (base + i, 0) in s.ids_for(title.lower().split()[0])
            if qid:
                assert s.by_qid(int(qid[1:]), lang) == base + i
            assert int(s.qid[base + i]) == (int(qid[1:]) if qid else 0)
    sizes = [int(s.size[i]) for i, _ in s.ids_for(en[0][1].lower().split()[0])]
    assert sizes == sorted(sizes, reverse=True)
    # a limited build keeps only the first ids of each source
    small = store.build([("en", f"{d}/en.db"), ("fr", f"{d}/fr.db")], f"{d}/s.safetensors", workers=1, chunk=64, limit=100, tok=tok)
    assert small["by_lang"] == {"en": 100, "fr": 100}, small
    print("magasin : construction et lecture ok |", {k: out[k] for k in ("gb", "bytes_per_token", "tokens_per_article", "share_lead_at_cap", "share_facts_at_cap", "share_with_facts")})


if __name__ == "__main__":
    main()
