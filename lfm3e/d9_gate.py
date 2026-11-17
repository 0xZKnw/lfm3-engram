"""D9: the memory with the 'when' probe plugged in, end to end, no thinking.

A linear probe on the model's state at the end of the prompt (layer 16) is trained on TriviaQA to predict that the
closed-book answer will be wrong. The memory (notes found by name, complete Wikipedia) is then shown only when the
probe is worried. Because the answers with and without memory are both generated for every question, the gated
result is computed exactly for any threshold, without generating again.

Also reported: what the probe would allow as an abstention signal ("I don't know" instead of a wrong answer)."""
import gzip
import json
import random
import sqlite3
import time
import urllib.request

from .d6_reason import find_triggers
from .d7_fr import MKQA_URL, TEXT, fold

GAP = 256
NOTE_TOKENS = 96
PROBE_LAYER = 16


def run(data_dir: str, popqa_dir: str, web_tokens_en: str, n_probe=5000, n_probe_fr=2000, n_eval=1000, seed=0):
    import shutil

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    t_start = time.time()
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id

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
    title_cache = {}

    def lookup(lang, surface):
        key = (lang, surface)
        if key not in title_cache:
            rows = dbs[lang].execute("SELECT a.title, a.description, a.lead FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,)).fetchall()
            title_cache[key] = min(rows, key=lambda r: (r[0].isupper(), -sum(c.islower() for c in r[0]), len(r[0]))) if rows else None
        return title_cache[key]

    def note(lang, row):
        title, desc, lead = row
        return [bos] + enc(TEXT[lang]["head"].format(title=title, desc=f" ({desc})" if desc else "")) + enc(lead)[:NOTE_TOKENS]

    def chat_ids(lang, q):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": TEXT[lang]["ask"].format(q=q)}], add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    think = enc("<think>")
    no_think = {"suppress_tokens": think} if len(think) == 1 else {}
    captured = {}
    model.model.layers[PROBE_LAYER].register_forward_pre_hook(lambda m, a, k: captured.__setitem__("h", (a[0] if a else k["hidden_states"]).detach()), with_kwargs=True)

    def pad_left(seqs):
        width = max(len(x[0]) for x in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, (x, m) in enumerate(seqs):
            ids[r, width - len(x) :], att[r, width - len(x) :] = torch.tensor(x), torch.tensor(m)
        return ids.to(dev), att.to(dev), width

    def generate(seqs, batch=48, max_new=40):
        outs = []
        for i in range(0, len(seqs), batch):
            ids, att, width = pad_left(seqs[i : i + batch])
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, **no_think)
            outs += [o.strip() for o in tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
        return outs

    def features(prompts, batch=256):
        """State at the last prompt token, input of layer PROBE_LAYER (no memory shown: the decision comes first)."""
        out = []
        for i in range(0, len(prompts), batch):
            ids, att, _ = pad_left([(p, [1] * len(p)) for p in prompts[i : i + batch]])
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=att)
            out.append(captured["h"][:, -1].float())
        return torch.cat(out)

    def layout(blocks, prompt):
        ids, mask = [], []
        for b in blocks:
            ids += b + [pad] * GAP
            mask += [1] * len(b) + [0] * GAP
        return ids + prompt, mask + [1] * len(prompt)

    def correct(out, golds):
        low = fold(out)
        return any(fold(g) in low for g in golds if len(g) >= 3)

    # ---- data
    rng = random.Random(seed)
    tq = []
    for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
        if len(x["question"]) < 300:
            tq.append({"q": x["question"], "golds": [x["answer"]["value"]] + list(x["answer"]["aliases"])})
        if len(tq) >= n_probe + n_eval:
            break
    rng.shuffle(tq)
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)  # same first 1000 as D5
    pop = [{"q": r["question"], "golds": json.loads(r["possible_answers"])} for r in popqa[:n_eval]]
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
    random.Random(0).shuffle(mk)  # same sample as D7
    mk, mk_extra = mk[:n_eval], mk[n_eval : n_eval + n_probe_fr]

    # ---- the probe: trained on TriviaQA only
    # trained on English TriviaQA plus French MKQA questions that are not in the evaluation sample: a probe trained
    # on English only still ranks French questions, but its threshold does not carry over
    train = tq[:n_probe]
    p_train = [chat_ids("en", x["q"]) for x in train] + [chat_ids("fr", x["fr"]) for x in mk_extra]
    g_train = [x["golds"] for x in train] + [x["golds"]["fr"] for x in mk_extra]
    wrong = torch.tensor([not correct(o, g) for o, g in zip(generate([(p, [1] * len(p)) for p in p_train]), g_train)], device=dev, dtype=torch.float32)
    x_train = features(p_train)
    mu, sd = x_train.mean(0, keepdim=True), x_train.std(0, keepdim=True) + 1e-6
    probe = torch.nn.Linear(x_train.shape[1], 1).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1.0)
    xs = (x_train - mu) / sd
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(probe(xs)[:, 0], wrong)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def worry(prompts):
        with torch.no_grad():
            return torch.sigmoid(probe((features(prompts) - mu) / sd)[:, 0])

    def auc(score, label):
        order = score.argsort()
        ranks = torch.empty_like(score)
        ranks[order] = torch.arange(1, len(score) + 1, device=dev, dtype=score.dtype)
        pos = label.sum()
        return ((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))).item() if 0 < pos < len(label) else float("nan")

    def evaluate(items, lang, key):
        freq, ntok = rarity[lang]
        prompts = [chat_ids(lang, x[key]) for x in items]
        golds = [x["golds"][lang] if isinstance(x["golds"], dict) else x["golds"] for x in items]
        base = generate([(p, [1] * len(p)) for p in prompts])
        picks = []
        for x in items:
            found = []
            for surface, _ in find_triggers(x[key], lambda s: lookup(lang, s) is not None, freq, ntok):
                row = lookup(lang, surface)
                if row and row[0] not in [f[0] for f in found]:
                    found.append(row)
            picks.append(found[:3])
        mem = generate([layout([note(lang, row) for row in pick], p) if pick else (p, [1] * len(p)) for pick, p in zip(picks, prompts)])
        ok_b = torch.tensor([correct(o, g) for o, g in zip(base, golds)], device=dev)
        ok_m = torch.tensor([correct(o, g) for o, g in zip(mem, golds)], device=dev)
        has = torch.tensor([bool(p) for p in picks], device=dev)
        w = worry(prompts)
        n = len(items)
        res = {
            "n": n,
            "no_memory": round(ok_b.float().mean().item(), 4),
            "memory_always": round(ok_m.float().mean().item(), 4),
            "broken_when_always": int((ok_b & ~ok_m).sum()),
            "fixed_when_always": int((~ok_b & ok_m).sum()),
            "note_triggered": round(has.float().mean().item(), 4),
            "probe_auc": round(auc(w, (~ok_b).float()), 4),
            "memory_only_when_probe_worried": {},
            "abstain_when_worried_and_no_note": {},
        }
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            use = w >= thr
            ok = torch.where(use, ok_m, ok_b)
            res["memory_only_when_probe_worried"][str(thr)] = {
                "accuracy": round(ok.float().mean().item(), 4), "broken": int((ok_b & ~ok).sum()), "fixed": int((~ok_b & ok).sum()),
                "notes_shown": round((use & has).float().mean().item(), 4),
            }
            # replace the answer by "I don't know" when the probe is worried and the memory has nothing to show
            abst = use & ~has
            kept = ~abst
            res["abstain_when_worried_and_no_note"][str(thr)] = {
                "abstained": round(abst.float().mean().item(), 4),
                "of_which_would_have_been_wrong": round((~ok_b[abst]).float().mean().item(), 4) if abst.any() else None,
                "correct": round((ok & kept).float().mean().item(), 4),
                "wrong": round((~ok & kept).float().mean().item(), 4),
            }
        res["wrong_without_any_gate"] = round((~ok_b).float().mean().item(), 4)
        res["examples"] = [
            {"q": x[key], "gold": g[:1], "worry": round(float(w[i]), 2), "notes": [row[0] for row in picks[i]], "no_memory": base[i][:60], "memory": mem[i][:60]}
            for i, (x, g) in enumerate(list(zip(items, golds))[:10])
        ]
        return res

    result = {"probe": {"layer": PROBE_LAYER, "trained_on_english": len(train), "trained_on_french": len(mk_extra), "wrong_rate_train": round(wrong.mean().item(), 4)}}
    result["triviaqa_held_out"] = evaluate(tq[n_probe:], "en", "q")
    result["popqa"] = evaluate(pop, "en", "q")
    result["mkqa_fr"] = evaluate(mk, "fr", "fr")
    result["mkqa_en"] = evaluate(mk, "en", "en")
    result["seconds_total"] = round(time.time() - t_start)
    return result
