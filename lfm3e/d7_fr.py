"""D7: the same memory in French.

MKQA questions (Natural Questions translated by humans) asked in French, with French notes found through French
titles. The same questions are also asked in English with English notes, as a reference.
No training, no thinking (forced), same layout as D4/D5: [note] [256 masked tokens] [prompt]."""
import gzip
import json
import random
import sqlite3
import time
import unicodedata
import urllib.request

from .d6_reason import find_triggers

GAP = 256
NOTE_TOKENS = 96
MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
TEXT = {
    "fr": {
        "ask": "Réponds à la question en quelques mots.\n\n{q}",
        "head": "Note de référence sur {title}{desc}. Utilise-la seulement si elle est pertinente, sinon ignore-la.\n",
    },
    "en": {
        "ask": "Answer the question in a few words.\n\n{q}",
        "head": "Reference note about {title}{desc}. Use it only if it is relevant, otherwise ignore it.\n",
    },
}


def fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()


def is_correct(out, golds):
    low = fold(out)
    return any(fold(g) in low for g in golds)


def run(data_dir: str, web_tokens_en: str, n=1000, seed=0):
    import os
    import shutil

    import numpy as np
    import torch

    from .common import load_model, load_tokenizer

    t_start = time.time()
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    # ---- dictionaries and word rarity, per language
    dbs = {}
    for lang, name in (("fr", "notes2_fr.db"), ("en", "notes2.db")):
        if os.path.exists(f"{data_dir}/{name}"):
            shutil.copy(f"{data_dir}/{name}", f"/tmp/{name}")
            dbs[lang] = sqlite3.connect(f"/tmp/{name}")
    fr_freq = json.load(open(f"{data_dir}/word_freq_fr.json"))
    web = np.load(web_tokens_en)
    counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
    del web
    wc = {}

    def word_tokens(word):
        if word not in wc:
            wc[word] = enc(" " + word)
        return wc[word]

    rarity = {
        # French: real word frequencies (the tokenizer splits most French words, so token counts say nothing)
        "fr": (lambda w: float(fr_freq.get(w.lower(), 0.0)), lambda w: 3 if fr_freq.get(w.lower(), 0.0) < 1.0 else 1),
        "en": (lambda w: float(counts[word_tokens(w.lower())[0]]) if len(word_tokens(w.lower())) == 1 else 0.0, lambda w: len(word_tokens(w))),
    }
    title_cache = {}

    def lookup(lang, surface):
        key = (lang, surface)
        if key not in title_cache:
            rows = dbs[lang].execute("SELECT a.title, a.description, a.lead FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,)).fetchall()
            title_cache[key] = min(rows, key=lambda r: (r[0].isupper(), -sum(c.islower() for c in r[0]), len(r[0]))) if rows else None
        return title_cache[key]

    def note(lang, row):
        title, desc, lead = row
        head = TEXT[lang]["head"].format(title=title, desc=f" ({desc})" if desc else "")
        return [bos] + enc(head) + enc(lead)[:NOTE_TOKENS]

    def chat_ids(lang, q):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": TEXT[lang]["ask"].format(q=q)}], add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    think = enc("<think>")
    no_think = {"suppress_tokens": think} if len(think) == 1 else {}

    def generate(seqs, batch=32, max_new=40):
        outs = []
        for i in range(0, len(seqs), batch):
            group = seqs[i : i + batch]
            width = max(len(x[0]) for x in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, (x, m) in enumerate(group):
                ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False, **no_think)
            outs += [o.strip() for o in tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
        return outs

    def layout(blocks, prompt):
        ids, mask = [], []
        for b in blocks:
            ids += b + [pad] * GAP
            mask += [1] * len(b) + [0] * GAP
        return ids + prompt, mask + [1] * len(prompt)

    # ---- MKQA: questions whose answer is an entity, in both languages
    with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
        raw = gzip.decompress(r.read()).decode("utf-8").splitlines()
    rows = []
    for line in raw:
        x = json.loads(line)
        golds = {}
        for lang in ("fr", "en"):
            g = []
            for a in x["answers"].get(lang, []):
                if a.get("type") == "entity" and a.get("text"):
                    g += [a["text"]] + [al for al in a.get("aliases", []) if al]
            golds[lang] = [s for s in dict.fromkeys(g) if len(s) >= 3]
        if golds["fr"] and golds["en"] and x["queries"].get("fr"):
            rows.append({"fr": x["queries"]["fr"], "en": x["query"], "golds": golds})
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    result = {"n": len(rows), "questions_with_a_capital_after_first_letter": {lang: round(sum(any(c.isupper() for c in r[lang][1:]) for r in rows) / len(rows), 4) for lang in ("fr", "en")}}

    def evaluate(lang):
        freq, ntok = rarity[lang]
        prompts = [chat_ids(lang, r[lang]) for r in rows]
        base = generate([(p, [1] * len(p)) for p in prompts])
        picks = []
        for r in rows:
            found = []
            for surface, kind in find_triggers(r[lang], lambda s: lookup(lang, s) is not None, freq, ntok):
                row = lookup(lang, surface)
                if row and row[0] not in [f[0] for f in found]:
                    found.append(row)
            picks.append(found[:3])
        mem = generate([layout([note(lang, row) for row in pick], p) if pick else (p, [1] * len(p)) for pick, p in zip(picks, prompts)])
        ok_b = [is_correct(o, r["golds"][lang]) for o, r in zip(base, rows)]
        ok_m = [is_correct(o, r["golds"][lang]) for o, r in zip(mem, rows)]
        trig = [bool(p) for p in picks]
        n_trig = max(sum(trig), 1)
        return {
            "no_memory": round(sum(ok_b) / len(rows), 4),
            "memory": round(sum(ok_m) / len(rows), 4),
            "fixed": sum(m and not b for b, m in zip(ok_b, ok_m)),
            "broken": sum(b and not m for b, m in zip(ok_b, ok_m)),
            "triggered": round(sum(trig) / len(rows), 4),
            "mean_notes": round(sum(len(p) for p in picks) / len(rows), 2),
            "when_triggered": {"no_memory": round(sum(b for b, t in zip(ok_b, trig) if t) / n_trig, 4), "memory": round(sum(m for m, t in zip(ok_m, trig) if t) / n_trig, 4)},
            "answer_in_a_note": round(sum(any(is_correct(row[2], r["golds"][lang]) for row in pick) for pick, r in zip(picks, rows)) / len(rows), 4),
            "examples": [
                {"q": r[lang], "gold": r["golds"][lang][:2], "notes": [row[0] for row in pick], "no_memory": b[:70], "memory": m[:70]}
                for r, pick, b, m in list(zip(rows, picks, base, mem))[:14]
            ],
        }

    for lang in ("fr", "en"):
        if lang in dbs:
            result[lang] = evaluate(lang)
    result["seconds_total"] = round(time.time() - t_start)
    return result
