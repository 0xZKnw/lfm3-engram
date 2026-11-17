"""D6: does the attention memory help reasoning? Thinking enabled.

Three mechanisms under test:
    1. correct premises       the memory supplies facts while the model reasons
    2. concept keys           triggers on rare technical terms, not only on capitalised names
    3. mid-generation trigger a name or term the model writes in its own reasoning brings its note in

Benchmarks: HotpotQA (two hops), GPQA Diamond (science, multiple choice), GSM8K (pure reasoning: must not degrade).
Memory = lead of the Wikipedia article, read as [note] [256 masked tokens] ... [prompt], as in D4."""
import json
import os
import random
import re
import sqlite3
import time

from .d5_trigger import SYSTEM, norm, title_bases

GAP = 256
NOTE_TOKENS = 96
CAP_STOP = {
    "the", "this", "that", "these", "those", "we", "it", "in", "if", "so", "but", "and", "then", "since", "because", "however",
    "therefore", "thus", "let", "given", "for", "as", "to", "now", "first", "second", "third", "next", "also", "they", "he", "she",
    "there", "what", "who", "which", "where", "when", "why", "how", "is", "was", "are", "were", "answer", "question", "note", "yes",
    "no", "a", "an", "i", "you", "your", "my", "our", "its", "his", "her", "their", "on", "at", "by", "from", "with", "or", "not",
    "both", "each", "all", "some", "one", "two", "three", "here", "step", "option", "options", "final", "reference", "user", "okay",
    "ok", "hmm", "wait", "actually", "maybe", "perhaps", "after", "before", "while", "during", "according", "based", "need", "check",
}
CONNECT = {"of", "the", "and", "de", "la", "von", "van", "in", "for", "on", "at", "to", "del", "di", "da", "du", "le", "des", "les", "et", "en", "au", "aux", "d'", "l'", "d’", "l’"}
FR_STOP = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou", "à", "a", "en", "dans", "sur", "pour", "par", "avec", "sans",
    "que", "qui", "quoi", "quel", "quelle", "quels", "quelles", "où", "ou", "quand", "comment", "combien", "pourquoi", "est", "sont", "était",
    "étaient", "été", "être", "avoir", "ont", "il", "elle", "ils", "elles", "on", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur",
    "leurs", "plus", "moins", "très", "ne", "pas", "se", "si", "mais", "donc", "car", "comme", "entre", "vers", "chez", "depuis", "pendant",
    "réponse", "question", "note", "oui", "non", "je", "tu", "nous", "vous", "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre",
    "premier", "première", "dernier", "dernière", "tout", "tous", "toute", "toutes", "fait", "faire", "y", "lors", "après", "avant",
    "l", "d", "j", "n", "s", "c", "m", "t", "qu",
}
FUNC = CAP_STOP | CONNECT | FR_STOP | {
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should", "may", "might",
    "must", "than", "then", "more", "most", "less", "least", "very", "such", "any", "other", "another", "same", "different", "between",
    "into", "about", "over", "under", "per", "following", "which", "whose", "whom", "use", "used", "using", "find", "get", "give",
}


def cap_runs(text):
    """Runs of capitalised words ('Kiss and Tell', 'Shirley Temple'), each with the up to three words that
    precede it ('guerre de' before 'Cent Ans'). A single capitalised word that starts a sentence is dropped: the
    capital says nothing there."""
    text = re.sub(r"\b([ldjmnstcLDJMNSTC]|qu|Qu)['’](?=\w)", r"\1' ", text)  # l'Allemagne -> l' Allemagne
    toks = re.findall(r"[^\W_][\w'’\-]*|[.!?\n:;]", text)
    runs, i, sentence_start = [], 0, True
    while i < len(toks):
        t = toks[i]
        if t in ".!?\n:;":
            sentence_start, i = True, i + 1
            continue
        if t[0].isupper():
            j, words, at_start = i + 1, [t], sentence_start
            while j < len(toks):
                if toks[j][0].isupper():
                    words.append(toks[j])
                    j += 1
                elif toks[j].lower() in CONNECT and j + 1 < len(toks) and toks[j + 1][0].isupper():
                    words += [toks[j], toks[j + 1]]
                    j += 2
                else:
                    break
            if not (len(words) == 1 and (at_start or words[0].lower() in CAP_STOP or words[0].lower() in FR_STOP)):
                left = []
                for k in range(i - 1, max(i - 4, -1), -1):
                    if toks[k] in ".!?\n:;":
                        break
                    left.insert(0, toks[k])
                runs.append((words, left))
            i = j
        else:
            i += 1
        sentence_start = False
    return runs


def find_triggers(text, exists, freq, ntok, max_concepts=4):
    """Surfaces to look up, as (surface, kind). Precision first: a wrong note costs more than a missing one.

    `exists(surface)`: an article has exactly this title. `freq(word)`: occurrences per million tokens of general
    web text (0 when the tokenizer splits the word, i.e. the word is rare). `ntok(word)`: number of tokens.

      - names: runs of at least two capitalised words that are an article title; a single word only when the
        tokenizer needs 3 tokens or more for it ('Pleiospilos' yes; 'Andy', 'Multiply', 'Switzerland' no)
      - concepts: 2-4 word terms that are an article title and contain an uncommon word ('center of mass' no,
        'pyridinium chlorochromate' yes); one word only when it is long and split into 3 tokens or more"""
    out, seen = [], set()

    def add(surface, kind):
        if surface and surface not in seen:
            seen.add(surface)
            out.append((surface, kind))

    def match(words):
        """Longest sub-span that is a title, then the same on what is left on each side."""
        for size in range(len(words), 0, -1):
            for a in range(0, len(words) - size + 1):
                sub = words[a : a + size]
                if sub[0] in CONNECT or sub[-1] in CONNECT:
                    continue
                surface = norm(" ".join(sub))
                if size == 1 and (len(surface) < 5 or " " in surface or surface in CAP_STOP or surface in FR_STOP or ntok(sub[0]) < 3):
                    continue
                if size > 1 and (len(surface.replace(" ", "")) < 5 or max(len(w) for w in surface.split()) < 3):
                    continue
                if exists(surface):
                    match(words[:a])
                    add(surface, "name")
                    match(words[a + size :])
                    return

    for words, left in cap_runs(text):
        for k in range(len(left), 0, -1):  # 'guerre de' + 'Cent Ans', 'battle of' + 'Hastings'
            ext = left[-k:] + words
            if ext[0].lower() in FUNC or len(ext) > 10:
                continue
            surface = norm(" ".join(ext))
            if exists(surface):
                add(surface, "name")
                break
        else:
            match(words[:10])

    text2 = re.sub(r"\b([ldjmnstc]|qu)['’](?=\w)", r"\1 ", text.lower())
    broken = re.sub(r"[.,;:!?()\[\]\n\"*_$\\{}=+<>/|^«»]", " zzzbreak ", re.sub(r"['’]s?(?![^\W\d_])", "", text2))
    low = re.findall(r"[^\W\d_][^\W\d_\-]+|[^\W\d_]", broken)
    found = []
    for size in (5, 4, 3, 2, 1):
        for a in range(0, len(low) - size + 1):
            sub = low[a : a + size]
            if "zzzbreak" in sub or sub[0] in FUNC or sub[-1] in FUNC or any(len(w) < 2 for w in sub):
                continue
            content = [w for w in sub if w not in FUNC]
            rarest = min(freq(w) for w in content)
            if size == 1 and (len(sub[0]) < 8 or ntok(sub[0]) < 3):
                continue
            if size > 1 and rarest > 20:
                continue
            surface = norm(" ".join(sub))
            if len(surface) < 6 or surface in seen or any(surface in f for f, _ in found) or any(surface in n for n in seen):
                continue
            if exists(surface):
                found.append((surface, rarest))
    for surface, _ in sorted(found, key=lambda x: (-len(x[0].split()), x[1]))[:max_concepts]:
        add(surface, "concept")
    return out


def norm_keep(text):
    """Lower-case with accents folded, punctuation kept (the concept scan uses it to cut at sentence marks)."""
    import unicodedata

    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).lower()


# ---------------------------------------------------------------- CPU: dictionary and datasets
def _scan(path):
    import collections

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["title", "text"])
    rows, df = [], collections.Counter()
    for title, text in zip(table["title"].to_pylist(), table["text"].to_pylist()):
        if len(text) < 1000 or "(disambiguation)" in title or title.startswith("List of"):
            continue
        rows.append((title, len(text), text[:700]))
        df.update(set(re.findall(r"[a-z][a-z\-]+", text[:600].lower())))
    return rows, df


def fetch_gpqa():
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    return [
        {"q": row["Question"], "correct": row["Correct Answer"], "wrong": [row[f"Incorrect Answer {k}"] for k in (1, 2, 3)], "domain": row.get("High-level domain", "")}
        for row in csv.DictReader(open(path, newline="", encoding="utf-8"))
    ]


def add_gpqa(out_dir: str):
    """Fetch GPQA alone and add it to the benchmark file (once access to the gated dataset is granted)."""
    bench = json.load(open(f"{out_dir}/benchmarks.json"))
    bench["gpqa"] = fetch_gpqa()
    with open(f"{out_dir}/benchmarks.json", "w") as f:
        json.dump(bench, f)
    return {"gpqa": len(bench["gpqa"])}


def build(out_dir: str):
    import collections
    import glob
    from concurrent.futures import ProcessPoolExecutor

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = rng.sample(range(len(hp)), 500)
    hotpot = [{"q": hp[i]["question"], "answer": hp[i]["answer"], "type": hp[i]["type"], "titles": sorted(set(hp[i]["supporting_facts"]["title"]))} for i in idx]
    gs = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(gs)), 300)
    gsm = [{"q": gs[i]["question"], "answer": gs[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]
    try:
        gpqa = fetch_gpqa()
    except Exception as e:  # noqa: BLE001 - gated dataset: keep going without it and say so
        gpqa, gpqa_error = [], f"{type(e).__name__}: {str(e)[:160]}"
    else:
        gpqa_error = None
    with open(f"{out_dir}/benchmarks.json", "w") as f:
        json.dump({"hotpot": hotpot, "gsm8k": gsm, "gpqa": gpqa}, f)
    root = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=["20231101.en/*.parquet"])
    files = sorted(glob.glob(f"{root}/20231101.en/*.parquet"))
    db_path = "/tmp/notes.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, len INTEGER, lead TEXT)")
    db.execute("CREATE TABLE surf (surface TEXT, id INTEGER, len INTEGER, exact INTEGER)")
    df, n_docs, next_id = collections.Counter(), 0, 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for rows, part_df in pool.map(_scan, files):
            df.update(part_df)
            n_docs += len(rows)
            arts, surfs = [], []
            for title, length, lead in rows:
                arts.append((next_id, title, length, lead))
                plain_title = norm(title)
                for b in title_bases(title):
                    surfs.append((b, next_id, length, int(b == plain_title)))
                next_id += 1
            db.executemany("INSERT INTO articles VALUES (?,?,?,?)", arts)
            db.executemany("INSERT INTO surf VALUES (?,?,?,?)", surfs)
    db.execute("CREATE INDEX surf_idx ON surf (surface)")
    db.execute("CREATE INDEX title_idx ON articles (title)")
    db.commit()
    db.close()
    import shutil

    shutil.copy(db_path, f"{out_dir}/notes.db")
    with open(f"{out_dir}/df.json", "w") as f:
        json.dump({"n_docs": n_docs, "df": {w: c for w, c in df.items() if c >= 3}}, f)

    return {"gpqa_error": gpqa_error, "articles": n_docs, "df_words": len(df), "db_gb": round(os.path.getsize(db_path) / 1e9, 2), "hotpot": len(hotpot), "gsm8k": len(gsm), "gpqa": len(gpqa)}


# ---------------------------------------------------------------- GPU
def run(data_dir: str, web_tokens: str, n_hotpot=250, n_gsm=150, n_gpqa=198, max_blocks=6, chunk=128, seed=0):
    import shutil

    import torch

    from .common import load_model, load_tokenizer

    t_start = time.time()
    shutil.copy(f"{data_dir}/notes.db", "/tmp/notes.db")
    db = sqlite3.connect("/tmp/notes.db")
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad, eos = tok.bos_token_id, tok.pad_token_id, tok.eos_token_id
    cache = {}

    def exists(surface):
        if surface not in cache:
            cache[surface] = db.execute("SELECT 1 FROM surf WHERE surface=? AND exact=1 LIMIT 1", (surface,)).fetchone() is not None
        return cache[surface]

    import numpy as np

    web = np.load(f"{web_tokens}")
    counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
    del web
    word_cache = {}

    def word_tokens(word):
        if word not in word_cache:
            word_cache[word] = tok(" " + word, add_special_tokens=False)["input_ids"]
        return word_cache[word]

    def ntok(word):  # in the word's own casing: ' Tuesday' is one token, ' tuesday' is not
        return len(word_tokens(word))

    def freq(word):
        ids = word_tokens(word.lower())
        return float(counts[ids[0]]) if len(ids) == 1 else 0.0

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def note(title, lead):
        return [bos] + enc(f"Reference note about {title}:\n{lead}")[:NOTE_TOKENS]

    def notes_for(surface, kind):
        rows = db.execute(
            "SELECT a.title, a.lead FROM surf s JOIN articles a ON a.id = s.id WHERE s.surface=? AND s.exact=1 ORDER BY s.len DESC LIMIT 1", (surface,)
        ).fetchall()
        return [(t, note(t, lead)) for t, lead in rows]

    def notes_for_title(title):
        row = db.execute("SELECT title, lead FROM articles WHERE title=? LIMIT 1", (title,)).fetchone()
        return [(row[0], note(row[0], row[1]))] if row else []

    def chat_ids(content, system):
        msgs = ([{"role": "system", "content": SYSTEM}] if system else []) + [{"role": "user", "content": content}]
        ids = enc(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    def layout(blocks, tail):
        ids, mask = [], []
        for b in blocks:
            ids += b + [pad] * GAP
            mask += [1] * len(b) + [0] * GAP
        return ids + tail, mask + [1] * len(tail)

    sampling = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}  # Liquid's evaluation setting

    close_think = enc("\n</think>\n\n")

    def solve(contents, mode, max_total, batch, oracle_titles=None, answer_budget=96):
        """mode: 'none', 'static' (notes from the question), 'dynamic' (plus notes triggered by what the model
        writes), 'oracle' (given titles). Returns per example: text, tokens generated, note titles, finished."""
        torch.manual_seed(seed)
        st = []
        for k, content in enumerate(contents):
            s = {"prompt": chat_ids(content, system=mode != "none"), "gen": [], "done": False, "blocks": [], "titles": [], "seen": set()}
            s["think_open"] = "<think>" in tok.decode(s["prompt"][-6:])
            if mode in ("static", "dynamic"):
                for surface, kind in find_triggers(content, exists, freq, ntok):
                    s["seen"].add(surface)
                    for title, block in notes_for(surface, kind):
                        if len(s["blocks"]) < max_blocks and title not in s["titles"]:
                            s["blocks"].append(block)
                            s["titles"].append(title)
            elif mode == "oracle":
                for t in oracle_titles[k]:
                    for title, block in notes_for_title(t):
                        s["blocks"].append(block)
                        s["titles"].append(title)
            s["static_titles"] = list(s["titles"])
            st.append(s)
        while True:
            active = sorted((s for s in st if not s["done"]), key=lambda s: len(s["prompt"]) + len(s["gen"]) + 400 * len(s["blocks"]))
            if not active:
                break
            for i in range(0, len(active), batch):
                group = active[i : i + batch]
                seqs = [layout(s["blocks"], s["prompt"] + s["gen"]) for s in group]
                width = max(len(x[0]) for x in seqs)
                ids = torch.full((len(group), width), pad, dtype=torch.long)
                att = torch.zeros((len(group), width), dtype=torch.long)
                for r, (x, m) in enumerate(seqs):
                    ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
                budget = min(chunk if mode == "dynamic" else max_total, max_total - min(len(s["gen"]) for s in group))
                with torch.inference_mode():
                    out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=budget, **sampling)
                for r, s in enumerate(group):
                    new = out[r, width:].tolist()
                    if eos in new:
                        new, s["done"] = new[: new.index(eos)], True
                    new = new[: max_total - len(s["gen"])]
                    s["gen"] += new
                    if len(s["gen"]) >= max_total:
                        s["done"] = True
                    if mode == "dynamic" and not s["done"] and len(s["blocks"]) < max_blocks:
                        recent = tok.decode(s["gen"][-(len(new) + 16) :], skip_special_tokens=True)
                        for surface, kind in find_triggers(recent, exists, freq, ntok, max_concepts=2):
                            if surface in s["seen"]:
                                continue
                            s["seen"].add(surface)
                            for title, block in notes_for(surface, kind):
                                if len(s["blocks"]) < max_blocks and title not in s["titles"]:
                                    s["blocks"].append(block)
                                    s["titles"].append(title)
        # thinking budget exhausted: close the block and ask for the answer (what a serving stack does)
        forced = [s for s in st if "</think>" not in tok.decode(s["gen"], skip_special_tokens=True) and (s["think_open"] or "<think>" in tok.decode(s["gen"][:8], skip_special_tokens=True)) and len(s["gen"]) >= max_total]
        for s in forced:
            s["gen"] += close_think
            s["forced"] = True
        for i in range(0, len(forced), batch):
            group = forced[i : i + batch]
            seqs = [layout(s["blocks"], s["prompt"] + s["gen"]) for s in group]
            width = max(len(x[0]) for x in seqs)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, (x, m) in enumerate(seqs):
                ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
            with torch.inference_mode():
                out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=answer_budget, **sampling)
            for r, s in enumerate(group):
                new = out[r, width:].tolist()
                s["gen"] += new[: new.index(eos)] if eos in new else new
        res = []
        for s in st:
            text = tok.decode(s["gen"], skip_special_tokens=True)
            if "</think>" in text:
                final = text.split("</think>")[-1].strip()
            else:  # still inside the thinking block when the budget ran out
                final = None if (s["think_open"] or "<think>" in text) else text.strip()
            res.append({"final": final, "tokens": len(s["gen"]), "titles": s["titles"], "static": len(s["static_titles"]), "forced": bool(s.get("forced")), "thinking": text[:300]})
        return res

    def summarise(res, correct):
        n = len(res)
        return {
            "accuracy": round(sum(correct) / n, 4),
            "thinking_cut_at_budget": round(sum(r["forced"] for r in res) / n, 4),
            "no_final_answer": round(sum(r["final"] is None for r in res) / n, 4),
            "mean_tokens": round(sum(r["tokens"] for r in res) / n),
            "median_tokens": sorted(r["tokens"] for r in res)[n // 2],
            "mean_notes": round(sum(len(r["titles"]) for r in res) / n, 2),
            "mean_notes_added_while_thinking": round(sum(len(r["titles"]) - r["static"] for r in res) / n, 2),
        }

    result = {"probe": {p_: [t for t, _ in notes_for(p_, "name")] for p_ in ("microwave", "cosmic microwave background", "schrodinger equation", "center of mass", "maxwell equations", "shirley temple", "skin")}}
    result["probe_triggers"] = find_triggers("The universe is filled with the Cosmic Microwave Background. Multiply the Gibbs free energy by two, then double check the nuclear magnetic resonance of pyridinium chlorochromate for Andy.", exists, freq, ntok)

    # ---- HotpotQA
    hp = bench["hotpot"][:n_hotpot]

    def hp_ok(final, answer):
        if final is None:
            return False
        a = answer.strip().lower()
        if a in ("yes", "no"):
            m = re.search(r"\b(yes|no)\b", final.lower())
            return bool(m) and m.group(1) == a
        return norm(a) in norm(final)

    contents = [f"Answer the question in a few words.\n\n{x['q']}" for x in hp]
    hres = {}
    for mode in ("none", "static", "dynamic", "oracle"):
        res = solve(contents, mode, max_total=768, batch=256, oracle_titles=[x["titles"] for x in hp])
        ok = [hp_ok(r["final"], x["answer"]) for r, x in zip(res, hp)]
        hres[mode] = {**summarise(res, ok), "bridge": round(sum(o for o, x in zip(ok, hp) if x["type"] == "bridge") / max(sum(x["type"] == "bridge" for x in hp), 1), 4),
                      "comparison": round(sum(o for o, x in zip(ok, hp) if x["type"] == "comparison") / max(sum(x["type"] == "comparison" for x in hp), 1), 4),
                      "gold_note_shown": round(sum(len(set(r["titles"]) & set(x["titles"])) for r, x in zip(res, hp)) / sum(len(x["titles"]) for x in hp), 4)}
        if mode == "dynamic":
            result["hotpot_examples"] = [{"q": x["q"], "answer": x["answer"], "gold": x["titles"], "notes": r["titles"], "added_while_thinking": r["titles"][r["static"]:], "final": (r["final"] or "")[:120]} for r, x in zip(res[:10], hp)]
    result["hotpot"] = {"n": len(hp), **hres}

    # ---- GPQA Diamond
    rng = random.Random(seed)
    gp = []
    for x in bench["gpqa"][:n_gpqa]:
        opts = [x["correct"]] + x["wrong"]
        order = rng.sample(range(4), 4)
        letters = "ABCD"
        body = "\n".join(f"{letters[p]}) {opts[o].strip()}" for p, o in enumerate(order))
        gp.append({"content": f"{x['q'].strip()}\n\n{body}\n\nFinish with a line of the form 'Answer: X' where X is A, B, C or D.", "gold": letters[order.index(0)], "domain": x["domain"]})

    def gp_ok(final, gold):
        if final is None:
            return False
        m = re.findall(r"Answer:\s*\**\(?([ABCD])\b", final)
        return bool(m) and m[-1] == gold

    gres = {}
    for mode in ("none", "dynamic") if gp else ():
        res = solve([x["content"] for x in gp], mode, max_total=2048, batch=100, answer_budget=64)
        ok = [gp_ok(r["final"], x["gold"]) for r, x in zip(res, gp)]
        gres[mode] = {**summarise(res, ok), "no_answer_line": round(sum(r["final"] is not None and not re.search(r"Answer:\s*\**\(?[ABCD]\b", r["final"]) for r in res) / len(res), 4)}
        if mode == "dynamic":
            result["gpqa_examples"] = [{"q": x["content"][:160], "notes": r["titles"], "added_while_thinking": r["titles"][r["static"]:]} for r, x in zip(res[:10], gp)]
    result["gpqa"] = {"n": len(gp), **gres}

    # ---- GSM8K (pure reasoning: the memory must not hurt)
    gs = bench["gsm8k"][:n_gsm]

    def gs_ok(final, gold):
        if final is None:
            return False
        tail = final.split("Answer:")[-1] if "Answer:" in final else final
        nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
        if not nums:
            return False
        try:
            return abs(float(nums[-1 if "Answer:" not in final else 0].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
        except ValueError:
            return False

    sres = {}
    for mode in ("none", "system_only", "dynamic"):
        res = solve([f"{x['q']}\n\nFinish with a line of the form 'Answer: <number>'." for x in gs], mode, max_total=1024, batch=256, answer_budget=64)
        sres[mode] = summarise(res, [gs_ok(r["final"], x["answer"]) for r, x in zip(res, gs)])
        if mode == "dynamic":
            result["gsm8k_examples"] = [{"q": x["q"][:120], "notes": r["titles"]} for r, x in zip(res[:8], gs)]
    result["gsm8k"] = {"n": len(gs), **sres}
    result["seconds_total"] = round(time.time() - t_start)
    return result


def calibrate(data_dir: str, popqa_dir: str, web_tokens: str, n=300):
    """How surprising is each note to the model? novelty = mean loss (nats/token) of the note text given its title.
    If the model already predicts a note well, the note tells it nothing new and can only distract.
    Compares notes known to be useful (gold articles of PopQA and HotpotQA) with the notes our triggers pick on
    GPQA and GSM8K."""
    import shutil

    import numpy as np
    import torch

    from .common import load_model, load_tokenizer

    shutil.copy(f"{data_dir}/notes.db", "/tmp/notes.db")
    db = sqlite3.connect("/tmp/notes.db")
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    tok = load_tokenizer()
    model = load_model()
    bos, pad = tok.bos_token_id, tok.pad_token_id
    weight = model.lm_head.weight.float()
    cache = {}

    def exists(surface):
        if surface not in cache:
            cache[surface] = db.execute("SELECT 1 FROM surf WHERE surface=? AND exact=1 LIMIT 1", (surface,)).fetchone() is not None
        return cache[surface]

    web = np.load(web_tokens)
    counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
    del web
    wc = {}

    def word_tokens(word):
        if word not in wc:
            wc[word] = tok(" " + word, add_special_tokens=False)["input_ids"]
        return wc[word]

    ntok = lambda w: len(word_tokens(w))  # noqa: E731
    freq = lambda w: float(counts[word_tokens(w.lower())[0]]) if len(word_tokens(w.lower())) == 1 else 0.0  # noqa: E731

    def by_title(title):
        return db.execute("SELECT title, lead FROM articles WHERE title=? LIMIT 1", (title,)).fetchone()

    def by_surface(surface):
        return db.execute("SELECT a.title, a.lead FROM surf s JOIN articles a ON a.id = s.id WHERE s.surface=? AND s.exact=1 ORDER BY s.len DESC LIMIT 1", (surface,)).fetchone()

    def novelty(rows, batch=128):
        """rows: (title, lead). Mean loss of the first 96 lead tokens given the header."""
        out = []
        for i in range(0, len(rows), batch):
            group = rows[i : i + batch]
            heads = [[bos] + tok(f"Reference note about {t}:\n", add_special_tokens=False)["input_ids"] for t, _ in group]
            bodies = [tok(lead, add_special_tokens=False)["input_ids"][:NOTE_TOKENS] for _, lead in group]
            width = max(len(h) + len(b) for h, b in zip(heads, bodies))
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, (h, b) in enumerate(zip(heads, bodies)):
                ids[r, : len(h) + len(b)] = torch.tensor(h + b)
                att[r, : len(h) + len(b)] = 1
            ids, att = ids.cuda(), att.cuda()
            with torch.inference_mode():
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                for r, (h, b) in enumerate(zip(heads, bodies)):
                    a, e = len(h), len(h) + len(b)
                    logits = hidden[r, a - 1 : e - 1].float() @ weight.T
                    lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[r, a:e, None])
                    out.append(-lp.mean().item())
        return out

    def describe(name, rows):
        rows = [r for r in rows if r]
        vals = novelty(rows)
        q = np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9]) if vals else [0] * 5
        order = np.argsort(vals)
        return {
            "n": len(vals), "p10": round(float(q[0]), 2), "p25": round(float(q[1]), 2), "median": round(float(q[2]), 2), "p75": round(float(q[3]), 2), "p90": round(float(q[4]), 2),
            "least_novel": [(rows[k][0], round(vals[k], 2)) for k in order[:6]], "most_novel": [(rows[k][0], round(vals[k], 2)) for k in order[-6:]],
        }

    rng = random.Random(0)
    pop = rng.sample(popqa, n)
    hp = bench["hotpot"][:n]
    out = {
        "popqa_gold_low_popularity": describe("p", [by_title(r["s_wiki_title"]) for r in pop if int(r["s_pop"]) < 1000]),
        "popqa_gold_high_popularity": describe("p", [by_title(r["s_wiki_title"]) for r in pop if int(r["s_pop"]) >= 10000]),
        "hotpot_gold": describe("h", [by_title(t) for x in hp for t in x["titles"]]),
    }
    for name, texts in (
        ("hotpot_triggered_not_gold", None),
        ("gpqa_triggered", [f"{x['q']} {x['correct']} {' '.join(x['wrong'])}" for x in bench["gpqa"]]),
        ("gsm8k_triggered", [x["q"] for x in bench["gsm8k"][:n]]),
    ):
        rows = []
        if texts is None:
            for x in hp:
                for surface, _ in find_triggers(x["q"], exists, freq, ntok):
                    row = by_surface(surface)
                    if row and row[0] not in x["titles"]:
                        rows.append(row)
        else:
            for t in texts:
                rows += [by_surface(surface) for surface, _ in find_triggers(t, exists, freq, ntok)]
        seen, uniq = set(), []
        for r in rows:
            if r and r[0] not in seen:
                seen.add(r[0])
                uniq.append(r)
        out[name] = describe(name, uniq)
    return out
