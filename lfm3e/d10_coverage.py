"""D10 (CPU only): how often is the answer in the note, depending on what the note contains?

For each benchmark question we take the articles the current system would show (found by name) and, where the
benchmark gives them, the articles that should be shown (gold). For each we check whether an accepted answer
appears in: the first 96 tokens of the summary (today's note), the first 256, the infobox, the first 1,000 tokens of
the body, the whole article. It is the ceiling of each option for the note content, before any GPU run."""
import gzip
import json
import random
import sqlite3
import urllib.request
import zipfile

from .d6_reason import find_triggers
from .d7_fr import MKQA_URL, fold

FIELDS = ("lead_96", "lead_256", "infobox", "lead_96_plus_infobox", "body_1000", "full_article")


def _flat(parts, out):
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        name, value = p.get("name"), p.get("value")
        if value:
            out.append(f"{name}: {value}" if name and p.get("type") == "field" else str(value))
        elif name:
            out.append(str(name))
        _flat(p.get("has_parts"), out)


def _scan(args):
    import orjson

    path, member, wanted = args
    rows = {}
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        for line in f:
            if b'"name"' not in line:
                continue
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            name = r.get("name")
            if name in wanted and name not in rows:
                info, body = [], []
                _flat(r.get("infoboxes") or r.get("infobox"), info)
                _flat(r.get("sections"), body)
                rows[name] = {"lead": r.get("abstract") or "", "infobox": "\n".join(info)[:20000], "body": "\n".join(body)[:400000], "keys": sorted(r.keys())}
    return rows


def run(data_dir: str, popqa_dir: str, web_tokens_en: str, n=1000, n_hotpot=250):
    import shutil
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    from .common import load_tokenizer

    tok = load_tokenizer()

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    dbs = {}
    for lang, name in (("fr", "notes2_fr.db"), ("en", "notes2.db")):
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
        "fr": (lambda w: float(fr_freq.get(w.lower(), 0.0)), lambda w: 3 if fr_freq.get(w.lower(), 0.0) < 1.0 else 1),
        "en": (lambda w: float(counts[word_tokens(w.lower())[0]]) if len(word_tokens(w.lower())) == 1 else 0.0, lambda w: len(word_tokens(w))),
    }
    cache = {}

    def lookup(lang, surface):
        key = (lang, surface)
        if key not in cache:
            rows = [r[0] for r in dbs[lang].execute("SELECT a.title FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,))]
            cache[key] = min(rows, key=lambda t: (t.isupper(), -sum(c.islower() for c in t), len(t))) if rows else None
        return cache[key]

    def triggered(lang, text):
        freq, ntok = rarity[lang]
        out = []
        for surface, _ in find_triggers(text, lambda s: lookup(lang, s) is not None, freq, ntok):
            t = lookup(lang, surface)
            if t and t not in out:
                out.append(t)
        return out[:3]

    # ---- the same samples as D5, D6, D7 and D9
    sets = {}
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)
    sets["popqa"] = ("en", [{"q": r["question"], "golds": json.loads(r["possible_answers"]), "gold_titles": [r["s_wiki_title"]]} for r in popqa[:n]])
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    hp = [x for x in bench["hotpot"][:n_hotpot] if x["answer"].strip().lower() not in ("yes", "no")]
    sets["hotpot"] = ("en", [{"q": x["q"], "golds": [x["answer"]], "gold_titles": x["titles"]} for x in hp])
    tq = []
    for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
        if len(x["question"]) < 300:
            tq.append({"q": x["question"], "golds": [x["answer"]["value"]] + list(x["answer"]["aliases"]), "gold_titles": []})
        if len(tq) >= 5000 + n:
            break
    random.Random(0).shuffle(tq)
    sets["triviaqa"] = ("en", tq[5000:])
    with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
        raw = gzip.decompress(r.read()).decode("utf-8").splitlines()
    mk = []
    for line in raw:
        x = json.loads(line)
        golds = {}
        for lang in ("fr", "en"):
            g = []
            for a in x["answers"].get(lang, []):
                if a.get("type") == "entity" and a.get("text"):
                    g += [a["text"]] + [al for al in a.get("aliases", []) if al]
            golds[lang] = list(dict.fromkeys(g))
        if golds["fr"] and golds["en"] and x["queries"].get("fr"):
            mk.append({"fr": x["queries"]["fr"], "en": x["query"], "golds": golds})
    random.Random(0).shuffle(mk)
    sets["mkqa_en"] = ("en", [{"q": x["en"], "golds": x["golds"]["en"], "gold_titles": []} for x in mk[:n]])
    sets["mkqa_fr"] = ("fr", [{"q": x["fr"], "golds": x["golds"]["fr"], "gold_titles": []} for x in mk[:n]])

    wanted = {"en": set(), "fr": set()}
    for lang, items in sets.values():
        for it in items:
            it["triggered"] = triggered(lang, it["q"])
            wanted[lang].update(it["triggered"])
            wanted[lang].update(it["gold_titles"])

    # ---- pull infobox and full text of the needed articles from the structured export
    articles = {"en": {}, "fr": {}}
    for lang in ("en", "fr"):
        path = hf_hub_download("wikimedia/structured-wikipedia", f"20240916.{lang}/{lang}wiki_namespace_0.zip", repo_type="dataset")
        members = [i.filename for i in zipfile.ZipFile(path).infolist() if i.file_size > 0]
        with ProcessPoolExecutor(max_workers=14) as pool:
            for part in pool.map(_scan, [(path, m, wanted[lang]) for m in members]):
                for k, v in part.items():
                    articles[lang].setdefault(k, v)

    def views(lang, title):
        a = articles[lang].get(title)
        if a is None:
            return None
        if "views" not in a:
            lead = enc(a["lead"])
            body = enc(a["body"][:60000])
            l96, l256 = tok.decode(lead[:96]), tok.decode(lead[:256])
            a["views"] = {
                "lead_96": l96, "lead_256": l256, "infobox": a["infobox"], "lead_96_plus_infobox": l96 + "\n" + a["infobox"],
                "body_1000": l256 + "\n" + tok.decode(body[:1000]), "full_article": a["lead"] + "\n" + a["infobox"] + "\n" + a["body"],
            }
            a["views"] = {k: fold(v) for k, v in a["views"].items()}
            a["infobox_tokens"] = len(enc(a["infobox"]))
        return a["views"]

    def has(text, golds):
        return any(fold(g) in text for g in golds if len(g) >= 3)

    result = {}
    for name, (lang, items) in sets.items():
        res = {"n": len(items)}
        for source in ("triggered", "gold_titles"):
            if source == "gold_titles" and not any(it["gold_titles"] for it in items):
                continue
            hit = {f: 0 for f in FIELDS}
            any_article = 0
            for it in items:
                vs = [v for v in (views(lang, t) for t in it[source]) if v]
                any_article += bool(vs)
                for f in FIELDS:
                    hit[f] += any(has(v[f], it["golds"]) for v in vs)
            res["current_system" if source == "triggered" else "gold_articles"] = {
                "questions_with_an_article": round(any_article / len(items), 4), **{f: round(hit[f] / len(items), 4) for f in FIELDS},
            }
        result[name] = res
    sizes = [a["infobox_tokens"] for lang in articles for a in articles[lang].values() if "infobox_tokens" in a]
    result["infobox"] = {"articles_checked": len(sizes), "with_infobox": round(sum(s > 0 for s in sizes) / max(len(sizes), 1), 4), "median_tokens_when_present": int(np.median([s for s in sizes if s > 0])) if any(sizes) else 0}
    result["example_infobox"] = next((a["infobox"][:600] for a in articles["en"].values() if a["infobox"]), "")
    result["record_keys"] = sorted({k for a in list(articles["en"].values())[:200] for k in a["keys"]})
    return result
