"""D6, second version: reasoning with the attention memory, on a complete Wikipedia.

The Hugging Face dump used so far (wikimedia/wikipedia 20231101.en) lacks many main articles ('Shirley Temple',
'Microwave', 'Skin', 'Manga', 'Norfolk', ...), so a name often resolved to a homonym. Here:
    - titles, short descriptions and summaries come from Wikimedia's structured-contents export (complete);
      disambiguation pages are skipped. Wikipedia's live API is not usable from Modal (HTTP 429).

Changes from the first version, each answering something the smoke tests showed:
    - no system prompt (it changes the model even without memory); the instruction sits in the note header
    - thinking budget: when it runs out the block is closed and the model must answer
    - optional novelty gate: a note the model already predicts well is not shown"""
import json
import os
import random
import re
import sqlite3
import time

from .d5_trigger import norm
from .d6_reason import find_triggers

GAP = 256
NOTE_TOKENS = 96


def _scan_member(args):
    """One jsonl member of the structured export -> (name, description, lead, wikidata id, infobox text, size)
    rows. Size is the length of the record: a cheap stand-in for how developed, hence how prominent, an article is."""
    import zipfile

    import orjson

    path, member, lang = args
    rows = []
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        for line in f:
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:  # a few records carry invalid UTF-8: skip them
                continue
            name, lead, desc = r.get("name") or "", r.get("abstract") or "", r.get("description") or ""
            if len(lead) < 80:
                continue
            head = lead[:200].lower()
            if lang == "fr":
                if name.startswith("Liste d") or "(homonymie)" in name or "homonymie" in desc.lower() or "homonymie" in head or "peut désigner" in head or "peut faire référence" in head:
                    continue
            else:
                if name.startswith("List of") or "(disambiguation)" in name or desc == "Topics referred to by the same term" or " may refer to" in head or " can refer to" in head:
                    continue
            info = []
            _flat_infobox(r.get("infoboxes"), info)
            rows.append((name, desc, lead[:700], (r.get("main_entity") or {}).get("identifier") or "", "\n".join(info)[:1500], len(line)))
    return rows


def _flat_infobox(parts, out):
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        name, value = p.get("name"), p.get("value")
        if value:
            out.append(f"{name}: {value}" if name and p.get("type") == "field" else str(value))
        elif name:
            out.append(str(name))
        _flat_infobox(p.get("has_parts"), out)


def build_notes(out_dir: str, lang: str = "en"):
    """CPU: Wikimedia's structured-contents export (complete, with abstract and short description) -> sqlite.
    The Wikidata id of each article is kept: it is what links the same entity across languages."""
    import shutil
    import zipfile
    from concurrent.futures import ProcessPoolExecutor

    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    path = hf_hub_download("wikimedia/structured-wikipedia", f"20240916.{lang}/{lang}wiki_namespace_0.zip", repo_type="dataset")
    members = [i.filename for i in zipfile.ZipFile(path).infolist() if i.file_size > 0]
    name = "notes3.db" if lang == "en" else f"notes3_{lang}.db"
    db_path = f"/tmp/{name}"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, description TEXT, lead TEXT, qid TEXT, infobox TEXT, size INTEGER)")
    db.execute("CREATE TABLE titles (surface TEXT, id INTEGER, exact INTEGER)")
    next_id = 0
    with ProcessPoolExecutor(max_workers=14) as pool:
        for rows in pool.map(_scan_member, [(path, m, lang) for m in members]):
            arts, surfs = [], []
            for title, desc, lead, qid, infobox, size in rows:
                base = re.sub(r"\s*\([^)]*\)$", "", title)
                surface = norm(base)
                if not surface:
                    continue
                arts.append((next_id, title, desc, lead, qid, infobox, size))
                surfs.append((surface, next_id, int(base == title)))
                if "," in base:  # 'Norfolk, Connecticut' also answers to 'Norfolk'
                    first = norm(base.split(",")[0])
                    if first and first != surface:
                        surfs.append((first, next_id, 0))
                next_id += 1
            db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", arts)
            db.executemany("INSERT INTO titles VALUES (?,?,?)", surfs)
    db.execute("CREATE INDEX surface_idx ON titles (surface)")
    db.execute("CREATE INDEX title_idx ON articles (title)")
    db.execute("CREATE INDEX qid_idx ON articles (qid)")
    db.commit()
    probe_list = {
        "en": ("shirley temple", "microwave", "skin", "manga", "norfolk", "cosmic microwave background", "maxwell equations", "pauli matrices", "tron"),
        "fr": ("allemagne", "londres", "guerre de cent ans", "energie libre de gibbs", "shirley temple", "cote d ivoire", "equations de maxwell", "victor hugo"),
    }[lang]
    probes = {
        p_: [tuple(r) for r in db.execute("SELECT a.title, a.description, a.qid FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 3", (p_,))]
        for p_ in probe_list
    }
    db.close()
    shutil.copy(db_path, f"{out_dir}/{name}")
    return {"lang": lang, "members": len(members), "articles": next_id, "db_gb": round(os.path.getsize(db_path) / 1e9, 2), "probes": probes}


def build_word_freq(tokens_path: str, out_path: str, model_key: str = "instruct"):
    """CPU: word frequencies (per million words) from a tokenized corpus, e.g. French Wikipedia. The tokenizer
    splits most French words, so token counts say nothing about how common a French word is."""
    import collections

    import numpy as np

    from .common import load_tokenizer

    tok = load_tokenizer(model_key)
    tokens = np.load(tokens_path)[:60_000_000]
    counts, total = collections.Counter(), 0
    for i in range(0, len(tokens), 2_000_000):
        words = re.findall(r"[^\W\d_]+", tok.decode(tokens[i : i + 2_000_000], skip_special_tokens=True).lower())
        counts.update(words)
        total += len(words)
    table = {w: round(c * 1e6 / total, 3) for w, c in counts.items() if c >= 5}
    with open(out_path, "w") as f:
        json.dump(table, f, ensure_ascii=False)
    return {"words": total, "kept": len(table), "examples": {w: table.get(w) for w in ("énergie", "guerre", "gibbs", "également", "allemagne", "capitale")}}


def run(data_dir: str, web_tokens: str, n_hotpot=250, n_gsm=150, n_gpqa=100, max_blocks=6, chunk=128, min_novelty=2.0, budget_scale=1.0, seed=0):
    import shutil

    import numpy as np
    import torch

    from .common import load_model, load_tokenizer

    t_start = time.time()
    shutil.copy(f"{data_dir}/notes2.db", "/tmp/notes2.db")
    db = sqlite3.connect("/tmp/notes2.db")
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad, eos = tok.bos_token_id, tok.pad_token_id, tok.eos_token_id
    weight = model.lm_head.weight.float()

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    # ---- dictionary
    title_cache = {}

    def title_for(surface):
        if surface not in title_cache:
            rows = [r[0] for r in db.execute("SELECT a.title FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,))]
            title_cache[surface] = min(rows, key=lambda t: (t.isupper(), -sum(c.islower() for c in t), len(t))) if rows else None
        return title_cache[surface]

    def exists(surface):
        return title_for(surface) is not None

    web = np.load(web_tokens)
    counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
    del web
    word_cache = {}

    def word_tokens(word):
        if word not in word_cache:
            word_cache[word] = enc(" " + word)
        return word_cache[word]

    def ntok(word):
        return len(word_tokens(word))

    def freq(word):
        ids = word_tokens(word.lower())
        return float(counts[ids[0]]) if len(ids) == 1 else 0.0

    # ---- notes: summary of the article, with its novelty for the model
    notes = {}  # title asked -> {"title", "block", "novelty"} or None

    def prepare(titles):
        """Read the summaries, build the note blocks and score their novelty. Fills `notes`."""
        todo = [t for t in dict.fromkeys(titles) if t and t not in notes]
        fresh = []
        for t in todo:
            row = db.execute("SELECT title, description, lead FROM articles WHERE title=? LIMIT 1", (t,)).fetchone()
            s = {"title": row[0], "description": row[1], "extract": row[2]} if row else None
            if not s:
                notes[t] = None
                continue
            desc = f" ({s['description']})" if s["description"] else ""
            head = [bos] + enc(f"Reference note about {s['title']}{desc}. Use it only if it is relevant, otherwise ignore it.\n")
            body = enc(s["extract"])[:NOTE_TOKENS]
            notes[t] = {"title": s["title"], "block": head + body, "head": len(head), "novelty": None}
            fresh.append(t)
        for i in range(0, len(fresh), 128):
            group = [notes[t] for t in fresh[i : i + 128]]
            width = max(len(g["block"]) for g in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, g in enumerate(group):
                ids[r, : len(g["block"])], att[r, : len(g["block"])] = torch.tensor(g["block"]), 1
            ids, att = ids.to(dev), att.to(dev)
            with torch.inference_mode():
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                for r, g in enumerate(group):
                    a, e = g["head"], len(g["block"])
                    lp = torch.log_softmax(hidden[r, a - 1 : e - 1].float() @ weight.T, dim=-1).gather(-1, ids[r, a:e, None])
                    g["novelty"] = -lp.mean().item()

    def chat_ids(content):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    def layout(blocks, tail):
        ids, mask = [], []
        for b in blocks:
            ids += b + [pad] * GAP
            mask += [1] * len(b) + [0] * GAP
        return ids + tail, mask + [1] * len(tail)

    sampling = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}  # Liquid's evaluation setting
    close_think = enc("\n</think>\n\n")

    def generate(group, max_new):
        seqs = [layout(s["blocks"], s["prompt"] + s["gen"]) for s in group]
        width = max(len(x[0]) for x in seqs)
        ids = torch.full((len(group), width), pad, dtype=torch.long)
        att = torch.zeros((len(group), width), dtype=torch.long)
        for r, (x, m) in enumerate(seqs):
            ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
        # read the prompt in slices so that one forward never holds more than ~130k tokens of activations
        chunk_ = max(64, 130_000 // len(group))
        extra = {"prefill_chunk_size": chunk_} if width > chunk_ else {}
        with torch.inference_mode():
            out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, **sampling, **extra)
        return [out[r, width:].tolist() for r in range(len(group))]

    def solve(contents, mode, think_budget, batch, oracle_titles=None, answer_budget=96, answer_prefix="", scans=None):
        """mode: 'none' | 'static' (notes from the question) | 'dynamic' (plus notes triggered by the model's own
        reasoning) | 'dynamic_gated' (same, novelty >= min_novelty only) | 'oracle' (given article titles)."""
        torch.manual_seed(seed)
        gated = mode.endswith("_gated")
        dynamic = mode.startswith("dynamic")
        st = [{"prompt": chat_ids(c), "gen": [], "done": False, "blocks": [], "titles": [], "seen": set(), "dropped": 0} for c in contents]
        for s in st:
            s["think_open"] = "<think>" in tok.decode(s["prompt"][-6:])

        def add_notes(s, titles):
            for t in titles:
                note = notes.get(t)
                if not note or note["title"] in s["titles"] or len(s["blocks"]) >= max_blocks:
                    continue
                if gated and note["novelty"] < min_novelty:
                    s["dropped"] += 1
                    continue
                s["blocks"].append(note["block"])
                s["titles"].append(note["title"])

        if mode != "none":
            wanted = []
            for k, (s, content) in enumerate(zip(st, contents)):
                if mode == "oracle":
                    s["want"] = list(oracle_titles[k])
                else:
                    trig = find_triggers((scans or contents)[k], exists, freq, ntok)
                    s["seen"].update(x for x, _ in trig)
                    s["want"] = [title_for(x) for x, _ in trig]
                wanted += s["want"]
            prepare(wanted)
            for s in st:
                add_notes(s, s["want"])
        for s in st:
            s["static"] = len(s["titles"])

        while True:
            active = sorted((s for s in st if not s["done"]), key=lambda s: len(s["prompt"]) + len(s["gen"]) + 400 * len(s["blocks"]))
            if not active:
                break
            for i in range(0, len(active), batch):
                group = active[i : i + batch]
                budget = min(chunk if dynamic else think_budget, think_budget - min(len(s["gen"]) for s in group))
                outs = generate(group, budget)
                wanted = []
                for s, new in zip(group, outs):
                    if eos in new:
                        new, s["done"] = new[: new.index(eos)], True
                    new = new[: think_budget - len(s["gen"])]
                    s["gen"] += new
                    if len(s["gen"]) >= think_budget:
                        s["done"] = True
                    s["want"] = []
                    if dynamic and not s["done"] and len(s["blocks"]) < max_blocks:
                        recent = tok.decode(s["gen"][-(len(new) + 16) :], skip_special_tokens=True)
                        for surface, _ in find_triggers(recent, exists, freq, ntok, max_concepts=2):
                            if surface not in s["seen"]:
                                s["seen"].add(surface)
                                s["want"].append(title_for(surface))
                    wanted += s["want"]
                if wanted:
                    prepare(wanted)
                    for s in group:
                        add_notes(s, s["want"])

        # thinking budget exhausted: close the block and ask for the answer (what a serving stack does)
        forced = [s for s in st if len(s["gen"]) >= think_budget and "</think>" not in tok.decode(s["gen"], skip_special_tokens=True)]
        for s in forced:
            s["gen"] += close_think + (enc(answer_prefix) if answer_prefix else [])
            s["forced"] = True
        for i in range(0, len(forced), batch):
            group = forced[i : i + batch]
            for s, new in zip(group, generate(group, answer_budget)):
                s["gen"] += new[: new.index(eos)] if eos in new else new
        res = []
        for s in st:
            text = tok.decode(s["gen"], skip_special_tokens=True)
            final = text.split("</think>")[-1].strip() if "</think>" in text else (None if (s["think_open"] or "<think>" in text) else text.strip())
            res.append({"final": final, "tokens": len(s["gen"]), "titles": s["titles"], "static": s["static"], "forced": bool(s.get("forced")), "dropped": s["dropped"]})
        return res

    def summarise(res, correct):
        n = len(res)
        return {
            "accuracy": round(sum(correct) / n, 4),
            "thinking_cut_at_budget": round(sum(r["forced"] for r in res) / n, 4),
            "mean_tokens": round(sum(r["tokens"] for r in res) / n),
            "mean_notes": round(sum(len(r["titles"]) for r in res) / n, 2),
            "mean_notes_added_while_thinking": round(sum(len(r["titles"]) - r["static"] for r in res) / n, 2),
            "mean_notes_dropped_by_gate": round(sum(r["dropped"] for r in res) / n, 2),
        }

    result = {"min_novelty": min_novelty}
    result["probe_triggers"] = find_triggers(
        "The universe is filled with the Cosmic Microwave Background. Multiply the Gibbs free energy by two, then double check Maxwell's equations and the nuclear magnetic resonance of pyridinium chlorochromate for Andy and Shirley Temple.",
        exists, freq, ntok, max_concepts=8,
    )

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
    hres, outputs = {}, {}
    for mode in ("none", "static", "dynamic", "oracle"):
        res = solve(contents, mode, think_budget=int(768 * budget_scale), batch=256, oracle_titles=[x["titles"] for x in hp], scans=[x["q"] for x in hp])
        ok = [hp_ok(r["final"], x["answer"]) for r, x in zip(res, hp)]
        outputs[mode] = (res, ok)
        gold_shown = sum(len({norm(t) for t in r["titles"]} & {norm(t) for t in x["titles"]}) for r, x in zip(res, hp)) / sum(len(x["titles"]) for x in hp)
        hres[mode] = {
            **summarise(res, ok),
            "bridge": round(sum(o for o, x in zip(ok, hp) if x["type"] == "bridge") / max(sum(x["type"] == "bridge" for x in hp), 1), 4),
            "comparison": round(sum(o for o, x in zip(ok, hp) if x["type"] == "comparison") / max(sum(x["type"] == "comparison" for x in hp), 1), 4),
            "gold_note_shown": round(gold_shown, 4),
        }
    base_ok = outputs["none"][1]
    for mode in ("static", "dynamic", "oracle"):
        ok = outputs[mode][1]
        hres[mode]["fixed"] = sum(o and not b for o, b in zip(ok, base_ok))
        hres[mode]["broken"] = sum(b and not o for o, b in zip(ok, base_ok))
    result["hotpot"] = {"n": len(hp), **hres}
    result["hotpot_examples"] = [
        {"q": x["q"], "answer": x["answer"], "gold": x["titles"], "notes": r["titles"], "added_while_thinking": r["titles"][r["static"]:], "final": (r["final"] or "")[:120]}
        for r, x in zip(outputs["dynamic"][0][:12], hp)
    ]

    # ---- GPQA Diamond
    rng = random.Random(seed)
    gp = []
    for x in bench["gpqa"][:n_gpqa]:
        opts = [x["correct"]] + x["wrong"]
        order = rng.sample(range(4), 4)
        body = "\n".join(f"{'ABCD'[p]}) {opts[o].strip()}" for p, o in enumerate(order))
        gp.append({"content": f"{x['q'].strip()}\n\n{body}\n\nFinish with a line of the form 'Answer: X' where X is A, B, C or D.", "gold": "ABCD"[order.index(0)],
                   "scan": x["q"] + "\n" + "\n".join(o.strip() for o in opts)})

    def gp_ok(final, gold):
        m = re.findall(r"Answer:\s*\**\(?([ABCD])\b", final or "")
        return bool(m) and m[-1] == gold

    gres, outputs = {}, {}
    for mode in ("none", "dynamic", "dynamic_gated") if gp else ():
        res = solve([x["content"] for x in gp], mode, think_budget=int(3072 * budget_scale), batch=100, answer_budget=16, answer_prefix="Answer:", scans=[x["scan"] for x in gp])
        ok = [gp_ok(r["final"], x["gold"]) for r, x in zip(res, gp)]
        outputs[mode] = (res, ok)
        gres[mode] = {**summarise(res, ok), "no_answer_letter": round(sum(not re.search(r"Answer:\s*\**\(?[ABCD]\b", r["final"] or "") for r in res) / len(res), 4)}
    if gp:
        for mode in ("dynamic", "dynamic_gated"):
            gres[mode]["fixed"] = sum(o and not b for o, b in zip(outputs[mode][1], outputs["none"][1]))
            gres[mode]["broken"] = sum(b and not o for o, b in zip(outputs[mode][1], outputs["none"][1]))
        result["gpqa_examples"] = [
            {"q": x["content"][:120], "notes": r["titles"], "kept_by_gate": g["titles"]} for r, g, x in zip(outputs["dynamic"][0][:12], outputs["dynamic_gated"][0], gp)
        ]
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
            return abs(float(nums[0 if "Answer:" in final else -1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
        except ValueError:
            return False

    sres = {}
    for mode in ("none", "dynamic"):
        res = solve([f"{x['q']}\n\nFinish with a line of the form 'Answer: <number>'." for x in gs], mode, think_budget=int(1024 * budget_scale), batch=256, answer_budget=24, answer_prefix="Answer:", scans=[x["q"] for x in gs])
        sres[mode] = summarise(res, [gs_ok(r["final"], x["answer"]) for r, x in zip(res, gs)])
        if mode == "dynamic":
            result["gsm8k_examples"] = [{"q": x["q"][:100], "notes": r["titles"]} for r, x in zip(res[:12], gs)]
    result["gsm8k"] = {"n": len(gs), **sres}

    nov = [n["novelty"] for n in notes.values() if n and n["novelty"] is not None]
    result["notes"] = {"fetched": len(notes), "usable": len(nov), "novelty_median": round(float(np.median(nov)), 2) if nov else None}
    result["seconds_total"] = round(time.time() - t_start)
    return result
