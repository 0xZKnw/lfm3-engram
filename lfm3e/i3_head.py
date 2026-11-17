"""I3: choice head v2, and a first look at letting the model choose by its own attention.

What changes from I2:
  - 'no note' is no longer a constant: it is a linear function of the question state, so the head can say
    "this prompt needs nothing" (a sum does not need a note about Easter)
  - every note goes through the head, including those found by the exact-title rules (names and technical terms)
  - 48 candidates per name instead of 12, and a feature saying whether a candidate's title is exactly the name
  - training prompts are no longer only quiz questions: instructions about a named thing (note wanted), French
    questions (note wanted), maths problems, questions that come with their own text, rewriting tasks (no note)
  - keys computed without a pass of the model (mean of input embeddings) are compared with the usual ones

Rules fixed before the run, on the development sets (the frozen tests of D14, now used for tuning):
  - v2 replaces v1 if the share of questions that get the right note is not significantly lower on PopQA nor on
    HotpotQA (McNemar, p < 0.05 counts as lower) and fewer prompts of the two unseen kinds (code, maths) get a note.
    HotpotQA was added after a dry run on 40 questions hinted that filtering exact-title notes may cost there.
  - if v2 fails only because every note goes through the head, the same head is kept with the filter applied
    to technical terms only, provided that setting passes the same test
  - static keys are kept (all-notes filter only) if they lose at most 1.5 points on PopQA and the loss is not
    significant on PopQA nor on HotpotQA

The final sets prepared here (never looked at, disjoint from training) are not touched by this run."""
import gzip
import json
import math
import random
import re
import time
import urllib.request

from .i2_head import INSTRUCTION, MKQA_URL, gold_by_title
from .triggers import norm, pick_spans

FR_INSTRUCTION = "Réponds en quelques mots.\n\n"
FR_TEMPLATES = {
    "director": "Qui a réalisé {s} ?", "screenwriter": "Qui a écrit le scénario {de_s} ?", "genre": "Quel est le genre {de_s} ?", "producer": "Qui a produit {s} ?",
    "author": "Qui est l'auteur {de_s} ?", "composer": "Qui a composé {s} ?", "country": "Dans quel pays se trouve {s} ?", "capital": "Quelle est la capitale {de_s} ?",
    "place of birth": "Dans quelle ville est né {s} ?", "father": "Qui est le père {de_s} ?", "sport": "Quel sport pratique {s} ?", "occupation": "Quelle est la profession {de_s} ?",
    "capital of": "De quel pays ou de quelle région {s} est-elle la capitale ?", "religion": "Quelle est la religion {de_s} ?", "mother": "Qui est la mère {de_s} ?", "color": "De quelle couleur est {s} ?",
}
EN_INSTRUCTIONS = (
    "Write two sentences about {s}.", "Tell me about {s}.", "Give me a short summary of {s}.", "I'm preparing a presentation on {s}. What are the key facts?",
    "Can you explain who or what {s} is?", "Write a short encyclopedia-style paragraph about {s}.",
)


def mcnemar(b, c):
    """Exact two-sided test on the pairs that differ (b: only the first is right, c: only the second)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n)


def bare_title(mem, i):
    """'Victor Hugo (romancier, poète)' -> 'Victor Hugo': the name as one would type it in a question."""
    t = mem.title(i)
    while True:
        u = re.sub(r"\s*\([^()]*(?:\([^()]*\)[^()]*)*\)$", "", t)
        if u == t or not u:
            return t
        t = u


def french_question(prop, name):
    de = f"d'{name}" if name[:1].lower() in "aeiouyhéèêâîôû" else f"de {name}"
    return FR_TEMPLATES[prop].format(s=name, de_s=de)


def prepare(mem, popqa_dir: str, data_dir: str, cap: int = 0, light: bool = False, seed: int = 0, n_hotpot: int = 7500):
    """CPU: development sets, final sets and training prompts, with their gold articles. Needs the note store and
    the tokenizer, not the model. Returns (data for this run, final sets)."""
    rng = random.Random(seed)
    n = cap or 1000

    def qid_ids(uri):
        q = (uri or "").rstrip("/").split("/")[-1]
        if q[:1] != "Q" or not q[1:].isdigit():
            return {}
        return {lang: i for lang in ("en", "fr") for i in [mem.store.by_qid(int(q[1:]), lang)] if i is not None}

    def names(text):
        has = type("Has", (), {"__contains__": lambda s, v: bool(mem.candidates(v, "en"))})()
        return {g[0] for g in mem.groups(text)[1]} | {s for _, _, s in pick_spans(text, has)}

    def qa_item(q, gold, golds=(), **more):
        return {"q": q, "gold": sorted(gold), "golds": list(golds), **more}

    # ---- PopQA: development, final, training; and the same entities asked in French
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)
    for r in popqa:
        r["ids"] = qid_ids(r["s_uri"])

    def popqa_items(rows):
        return [qa_item(r["question"], r["ids"].values(), json.loads(r["possible_answers"])) for r in rows]

    def popqa_fr(rows):
        out = []
        for r in rows:
            if "fr" in r["ids"] and r["prop"] in FR_TEMPLATES:
                out.append(qa_item(french_question(r["prop"], bare_title(mem, r["ids"]["fr"])), r["ids"].values()))
        return out

    dev = {"popqa": popqa_items(popqa[:n]), "popqa_fr": popqa_fr(popqa[:n])}
    final = {"popqa": popqa_items(popqa[n : 2 * n]), "popqa_fr": popqa_fr(popqa[n : 2 * n])}
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    n_hp = min(n, 250)
    dev["hotpot"] = [qa_item(x["q"], {i for i in (gold_by_title(mem, t) for t in x["titles"]) if i is not None}, [x["answer"]]) for x in bench["hotpot"][:n_hp]]
    dev["gsm8k"] = [{"q": x["q"], "gold": [], "answer": x["answer"]} for x in bench["gsm8k"][: (cap or 300)]]
    info = {}
    if not light:
        from datasets import load_dataset

        tq = []
        for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
            if len(x["question"]) < 300:
                tq.append(qa_item(x["question"], [], [x["answer"]["value"]] + list(x["answer"]["aliases"])))
            if len(tq) >= 14500:
                break
        head_part = tq[: 5000 + 1000]
        random.Random(0).shuffle(head_part)  # same held-out 1,000 as D9, D11, D14 and I2
        dev["triviaqa"] = head_part[5000:][:n]
        final["triviaqa"] = tq[13500:][:n]  # never used by any experiment
        with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
            raw = gzip.decompress(r.read()).decode("utf-8").splitlines()
        mk = []
        for line in raw:
            x = json.loads(line)
            ge = [s for a in x["answers"].get("en", []) if a.get("type") == "entity" and a.get("text") for s in [a["text"]] + [al for al in a.get("aliases", []) if al]]
            gf = [s for a in x["answers"].get("fr", []) if a.get("type") == "entity" and a.get("text") for s in [a["text"]] + [al for al in a.get("aliases", []) if al]]
            if ge and gf and x["queries"].get("fr"):
                mk.append((qa_item(x["query"], [], list(dict.fromkeys(ge))), qa_item(x["queries"]["fr"], [], list(dict.fromkeys(gf)))))
        random.Random(0).shuffle(mk)
        dev["mkqa_en"], dev["mkqa_fr"] = [a for a, _ in mk[:n]], [b for _, b in mk[:n]]
        final["mkqa_en"], final["mkqa_fr"] = [a for a, _ in mk[1000 : 1000 + n]], [b for _, b in mk[1000 : 1000 + n]]
        hp_val = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        used = {x["q"] for x in bench["hotpot"]}
        final["hotpot"] = []
        for i in random.Random(1).sample(range(len(hp_val)), len(hp_val)):
            x = hp_val[i]
            if x["question"] not in used:
                final["hotpot"].append(qa_item(x["question"], {i for i in (gold_by_title(mem, t) for t in sorted(set(x["supporting_facts"]["title"]))) if i is not None}, [x["answer"]]))
            if len(final["hotpot"]) >= n_hp:
                break
        gs = load_dataset("openai/gsm8k", "main", split="test")
        used = {x["q"] for x in bench["gsm8k"]}
        rest = [x for x in gs if x["question"] not in used]
        final["gsm8k"] = [{"q": x["question"], "gold": [], "answer": x["answer"].split("####")[-1].strip().replace(",", "")} for x in random.Random(1).sample(rest, min(len(rest), cap or 300))]

        def other(kind, loader, pick, limit):
            """Prompts of a kind, as plain texts. A source that fails to load is reported, not fatal."""
            try:
                texts = [t for t in (pick(x) for x in loader()) if t and 20 < len(t) < 2500]
                random.Random(seed).shuffle(texts)
                info[kind] = len(texts)
                return texts[:limit]
            except Exception as e:  # noqa: BLE001
                info[kind] = f"failed: {e!r}"[:200]
                return []

        dolly = lambda: load_dataset("databricks/databricks-dolly-15k", split="train")
        alpaca = lambda: load_dataset("tatsu-lab/alpaca", split="train")
        lim = 4 * cap if cap else 4000
        neg_sources = {
            "gsm8k_train": [x["question"] for x in load_dataset("openai/gsm8k", "main", split="train")][: (4 * cap if cap else 100000)],
            "with_own_text": other("with_own_text", dolly, lambda x: f"{x['context'][:1500]}\n\n{x['instruction']}" if x["category"] in ("closed_qa", "information_extraction", "summarization") and x["context"] else None, lim),
            "rewriting": other("rewriting", alpaca, lambda x: f"{x['instruction']}\n\n{x['input']}" if x["input"] else None, lim),
        }
        k = cap or 300
        dev["code"] = [{"q": t, "gold": []} for t in other("code", lambda: load_dataset("sahil2801/CodeAlpaca-20k", split="train"), lambda x: f"{x['instruction']}\n\n{x['input']}".strip(), k)]
        dev["maths"] = [{"q": t, "gold": []} for t in other("maths", lambda: load_dataset("EleutherAI/hendrycks_math", "algebra", split="test"), lambda x: x["problem"], k)]
        dev["creative"] = [{"q": t, "gold": []} for t in other("creative", dolly, lambda x: x["instruction"] if x["category"] in ("creative_writing", "brainstorming") and not x["context"] else None, k)]
    else:
        neg_sources = {"gsm8k_train": [x["q"] for x in bench["gsm8k"][150:]][: 4 * (cap or 40)]}
        dev["gsm8k"] = dev["gsm8k"][:150]

    # ---- nothing that shares a question, an article or a name with a development or final question is trained on
    held = [it for part in (dev, final) for key, items in part.items() if key not in ("gsm8k", "code", "maths", "creative") for it in items]
    held_gold = {i for it in held for i in it["gold"]}
    held_names = {s for it in held for s in names(it["q"])}
    held_q = {norm(it["q"]) for it in held}

    def clean(q, gold=()):
        return norm(q) not in held_q and not (set(gold) & held_gold) and not (names(q) & held_names)

    train = []

    def add(src, text, gold):
        train.append({"src": src, "text": text, "gold": sorted(gold)})

    rows = [r for r in popqa[2 * n :] if r["ids"] and clean(r["question"], r["ids"].values())]
    if cap:
        rows = rows[: 4 * cap]
    for r in rows:
        add("popqa", r["question"], r["ids"].values())
        add("popqa", INSTRUCTION + r["question"], r["ids"].values())
        if "fr" in r["ids"] and r["prop"] in FR_TEMPLATES:
            q = french_question(r["prop"], bare_title(mem, r["ids"]["fr"]))
            add("french", rng.choice(("", FR_INSTRUCTION)) + q, r["ids"].values())
    for r in rng.sample(rows, min(len(rows), cap or 3000)):
        add("instruction", rng.choice(EN_INSTRUCTIONS).format(s=r["subj"]), r["ids"].values())
    hp = []
    if light:
        pool = [(x["q"], x["titles"]) for x in bench["hotpot"][250:]]
    else:
        hp_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        pool = ((hp_train[i]["question"], sorted(set(hp_train[i]["supporting_facts"]["title"]))) for i in rng.sample(range(len(hp_train)), min(len(hp_train), 20000)))
    for q, titles in pool:
        gold = {i for i in (gold_by_title(mem, t) for t in titles) if i is not None}
        if len(gold) == 2 and clean(q, gold):
            hp.append((q, gold))
        if len(hp) >= (4 * cap if cap else n_hotpot):
            break
    for q, gold in hp:
        add("hotpot", q, gold)
        add("hotpot", INSTRUCTION + q, gold)
    for src, texts in neg_sources.items():
        for t in texts:
            if clean(t):
                add(src, t, [])
    info.update(train={s: sum(x["src"] == s for x in train) for s in sorted({x["src"] for x in train})}, dev={k: len(v) for k, v in dev.items()}, final={k: len(v) for k, v in final.items()})
    return {"dev": dev, "train": train, "info": info, "cap": cap, "light": light}, final


def run(hf_dir: str, prepared: dict, web_tokens: str = "", out_dir: str = "", n_att_train: int = 3000, device: str = "cuda", seed: int = 0):
    """hf_dir: an exported model folder without head. prepared: the first value returned by prepare()."""
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    mem, cfg = model.memory, model.memory.cfg
    assert model.__dict__["_mem_head_cpu"] is None, "the folder already has a head"
    token_freq = None
    if web_tokens and os.path.exists(web_tokens):
        web = np.load(web_tokens)
        token_freq = (np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))).astype(np.float32)
        mem.token_freq = token_freq
    K = cfg.candidates
    model.__dict__["_mem_key_cache_max"] = 2_000_000  # the development sets share their candidates across variants
    dev, train = prepared["dev"], prepared["train"]
    timing = {"load_s": round(time.time() - t_start)}

    # ------------------------------------------------------------ examples: one per name or term found in a text
    def examples(items, text_of):
        """-> texts, [(text index, kind, candidates, exact flags, index of the right one or K)], item of each text"""
        texts, ex, owner = [], [], []
        for n_item, it in enumerate(items):
            text, gold = text_of(it), set(it["gold"])
            groups = [g for g in mem.groups(text)[1] if g[2]]
            if not groups:
                continue
            texts.append(text)
            owner.append(n_item)
            for _, kind, cands, flags in groups:
                hit = [j for j, i in enumerate(cands) if i in gold]
                ex.append((len(texts) - 1, kind, cands, flags, hit[0] if hit else K))
        return texts, ex, owner

    t0 = time.time()
    cfg.gate_strict = True
    tr_texts, tr_ex, owner = examples(train, lambda it: it["text"])
    src_of_text = [train[n]["src"] for n in owner]
    all_ids = sorted({i for e in tr_ex for i in e[2]})
    row = {i: n for n, i in enumerate(all_ids)}
    emb = model.get_input_embeddings().weight
    with torch.inference_mode():
        seqs = [mem.key(i) for i in all_ids]
        keys_ctx = torch.cat([model._mem_states(seqs[a : a + 4096], last=False, batch=256).half() for a in range(0, len(seqs), 4096)])
        keys_sta = torch.stack([emb[torch.tensor(s, device=device)].float().mean(0) for s in seqs]).half()
        q_tr = model._mem_states([mem.question_ids(t) for t in tr_texts], last=True, batch=128)
    keys_ctx, keys_sta, q_tr = keys_ctx.clone(), keys_sta.clone(), q_tr.clone()
    prior_all = torch.tensor([mem.prior(i) for i in all_ids], device=device)
    timing["states_s"] = round(time.time() - t0)
    print(f"[i3] states done: {len(tr_texts)} texts, {len(tr_ex)} examples, {len(all_ids)} keys, {timing['states_s']} s", flush=True)

    def pack(q_states, ex):
        qi = torch.tensor([e[0] for e in ex], device=device, dtype=torch.long)
        cand = torch.zeros(len(ex), K, dtype=torch.long, device=device)
        mask = torch.zeros(len(ex), K, dtype=torch.bool, device=device)
        flag = torch.zeros(len(ex), K, device=device)
        for n, e in enumerate(ex):
            cand[n, : len(e[2])] = torch.tensor([row[i] for i in e[2]], device=device)
            mask[n, : len(e[2])] = True
            flag[n, : len(e[3])] = torch.tensor(e[3], device=device, dtype=torch.float)
        return q_states[qi], cand, mask, flag, torch.tensor([e[4] for e in ex], device=device, dtype=torch.long)

    def scores(h, keys, q, cand, mask, flag):
        qn = (q - h["q_mu"]) / h["q_sd"]
        kn = (keys[cand].float() - h["k_mu"]) / h["k_sd"]
        s = torch.einsum("bd,bkd->bk", F.normalize(qn @ h["q"].T, dim=-1), F.normalize(kn @ h["k"].T, dim=-1)) * h["log_t"].exp() + h["w_prior"] * prior_all[cand]
        null = h["null"].expand(len(q), 1)
        if "w_exact" in h:
            s = s + h["w_exact"] * flag
        if "null_w" in h:
            null = null + qn @ h["null_w"].T
        return torch.cat([s.masked_fill(~mask, -1e4), null], dim=1)

    def fit(ex, keys, v2, epochs=40, dim=256):
        torch.manual_seed(seed)
        q, cand, mask, flag, y = pack(q_tr, ex)
        d = q.shape[1]
        h = {"q": torch.nn.Linear(d, dim, bias=False).weight.detach().to(device), "k": torch.nn.Linear(d, dim, bias=False).weight.detach().to(device),
             "w_prior": torch.zeros(1, device=device), "null": torch.zeros(1, device=device), "log_t": torch.full((1,), 2.5, device=device)}
        if v2:
            h.update(w_exact=torch.zeros(1, device=device), null_w=torch.zeros(1, d, device=device))
        used = keys[cand[mask].unique()].float()
        stats = dict(q_mu=q.mean(0, keepdim=True), q_sd=q.std(0, keepdim=True) + 1e-6, k_mu=used.mean(0, keepdim=True), k_sd=used.std(0, keepdim=True) + 1e-6)
        params = [v.requires_grad_(True) for v in h.values()]
        h.update(stats)
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=device)
            for a in range(0, len(perm), 512):
                idx = perm[a : a + 512]
                loss = F.cross_entropy(scores(h, keys, q[idx], cand[idx], mask[idx], flag[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {k: v.detach() for k, v in h.items()}

    t0 = time.time()
    quiz = [e for e in tr_ex if src_of_text[e[0]] in ("popqa", "hotpot") and e[1] == "name"]  # what I2 trained on
    heads = {
        "v1": (fit(quiz, keys_ctx, v2=False), "contextual", False, K),
        "v2": (fit(tr_ex, keys_ctx, v2=True), "contextual", True, K),
        "v2_static": (fit(tr_ex, keys_sta, v2=True), "static", True, K),
    }
    heads["v2_12_candidates"] = (heads["v2"][0], "contextual", True, 12)
    heads["v2_filter_terms_only"] = (heads["v2"][0], "contextual", "concepts", K)
    timing["fit_s"] = round(time.time() - t0)
    print(f"[i3] heads fitted, {timing['fit_s']} s", flush=True)

    # ------------------------------------------------------------ what each head shows on the development sets
    def use(h, kind="contextual", gate_strict=True, n_cand=K):
        model.__dict__["_mem_head_cpu"] = None if h is None else {k: v.float().cpu() for k, v in h.items()}
        model.__dict__["_mem_head_dev"] = None
        if model.__dict__["_mem_static_keys"] != (kind == "static"):  # keys of the other kind: not comparable
            model.__dict__["_mem_key_cache"].clear()
        model.__dict__["_mem_static_keys"] = kind == "static"
        cfg.gate_strict, cfg.candidates = gate_strict, n_cand
        mem._cand.clear()

    def text_of(key, q):
        if key in ("popqa", "hotpot", "triviaqa", "mkqa_en"):
            return INSTRUCTION + q
        return FR_INSTRUCTION + q if key in ("popqa_fr", "mkqa_fr") else q

    def shown(key):
        items, plans = dev[key], []
        for a in range(0, len(items), 128):
            plans += model.memory_plan([mem.question_ids(text_of(key, it["q"])) for it in items[a : a + 128]])
        r = {"n": len(items), "with_note": round(sum(bool(p) for p in plans) / max(len(items), 1), 4), "notes_per_prompt": round(sum(map(len, plans)) / max(len(items), 1), 3)}
        right = None
        if any(it["gold"] for it in items):
            known = [(set(it["gold"]), set(p)) for it, p in zip(items, plans) if it["gold"]]
            right = [bool(g & p) for g, p in known]
            r.update(gold_known=len(known), right_note=round(sum(right) / max(len(known), 1), 4))
            if key == "hotpot":
                r["both_notes"] = round(sum(len(g & p) >= 2 for g, p in known) / max(len(known), 1), 4)
        return r, right, plans

    t0 = time.time()
    report, right_popqa, right_sets, examples_shown = {}, {}, {}, {}
    for name, (h, kind, gate, n_cand) in [("no_head", (None, "contextual", False, K))] + list(heads.items()):
        use(h, kind, gate, n_cand)
        report[name] = {}
        for key in dev:
            if not dev[key]:
                continue
            report[name][key], right, plans = shown(key)
            if key in ("popqa", "hotpot"):
                right_sets[(name, key)] = right
            if name == "v2" and key in ("gsm8k", "code", "maths", "creative"):
                examples_shown[key] = [{"q": it["q"][:140], "notes": [mem.title(i) for i in p]} for it, p in zip(dev[key], plans) if p][:4]
    timing["eval_s"] = round(time.time() - t0)

    def paired(a, b, key):
        x, y = right_sets[(a, key)], right_sets[(b, key)]
        only_a, only_b = sum(p and not q for p, q in zip(x, y)), sum(q and not p for p, q in zip(x, y))
        return {"only_" + a: only_a, "only_" + b: only_b, "p": round(mcnemar(only_a, only_b), 4)}

    def not_lower(a, b):
        """a is not significantly below b on PopQA nor on HotpotQA."""
        return all(not (t["only_" + b] > t["only_" + a] and t["p"] < 0.05) for t in (paired(a, b, k) for k in ("popqa", "hotpot")))

    kinds = [k for k in ("code", "maths") if dev.get(k)] or ["gsm8k"]  # gsm8k only if neither unseen kind could be loaded
    unseen = lambda name: sum(report[name][k]["with_note"] for k in kinds)
    tests = {f"{a}_vs_{b}": {k: paired(a, b, k) for k in ("popqa", "hotpot")} for a, b in (("v2", "v1"), ("v2_filter_terms_only", "v1"), ("v2_static", "v2"), ("v2_12_candidates", "v2"))}
    ok = {name: not_lower(name, "v1") and unseen(name) < unseen("v1") for name in ("v2", "v2_filter_terms_only")}
    static_ok = report["v2"]["popqa"]["right_note"] - report["v2_static"]["popqa"]["right_note"] <= 0.015 and not_lower("v2_static", "v2")
    chosen = ("v2_static" if static_ok else "v2") if ok["v2"] else ("v2_filter_terms_only" if ok["v2_filter_terms_only"] else "v1")
    print(f"[i3] evaluation done, {timing['eval_s']} s; exporting {chosen}", flush=True)

    files = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name in ("v1", "v2", "v2_static", "v2_filter_terms_only"):
            h, kind, gate, n_cand = heads[name]
            tensors = {k: v.float().cpu().numpy() for k, v in h.items()}
            if token_freq is not None:
                tensors["token_freq"] = token_freq
            meta = {"keys": kind, "gate_strict": str(gate), "candidates": str(n_cand), "trained_by": "lfm3e/i3_head.py", "variant": name}
            for target in [f"head_{name}.safetensors"] + (["head.safetensors"] if name == chosen else []):
                save_file(tensors, f"{out_dir}/{target}", metadata=meta)
                files[target] = os.path.getsize(f"{out_dir}/{target}")

    # ------------------------------------------------------------ can the model choose by its own attention?
    t0 = time.time()
    use(*heads["v2"])
    attention = attention_readout(model, mem, train, dev, n_train=n_att_train, device=device, rng=rng)
    timing["attention_s"] = round(time.time() - t0)
    timing["total_s"] = round(time.time() - t_start)
    return {"sizes": {"train_texts": len(tr_texts), "examples": len(tr_ex), "examples_by_kind": {k: sum(e[1] == k for e in tr_ex) for k in ("name", "title", "concept")},
                      "none_labels": round(sum(e[4] == K for e in tr_ex) / max(len(tr_ex), 1), 4), "keys": len(all_ids), "quiz_examples_v1": len(quiz), "prepared": prepared.get("info")},
            "heads": report, "paired_tests_popqa": tests, "exported": chosen, "rule": {"unseen_kinds_used": kinds, "passes": ok, "static_ok": static_ok}, "files": files, "examples_of_notes_v2": examples_shown, "attention": attention, "timing": timing}


def attention_readout(model, mem, train, dev, n_train, device, rng, n_cand=12, lead=24):
    """Shows the model the first lines of every candidate note at once, in random order, and reads how much the
    last token of the prompt looks at each one in the six global-attention layers.

    Two questions: does the most-looked-at candidate tend to be the right article (compared with the head and
    with 'most developed'), and does the model look less at the notes of a prompt that needs none?"""
    import torch
    import torch.nn.functional as F

    cfg = mem.cfg
    layers = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
    backbone = getattr(model, cfg.backbone_attr)
    caught = {}

    def hook(i):
        def fn(module, args, kwargs, output):
            hidden = args[0] if args else kwargs["hidden_states"]
            caught[i] = (output[1][:, :, -1, :].float(), torch.sigmoid(module.gate_proj(hidden[:, -1])).float())  # [B, heads, T], [B, heads]
        return fn

    def collect(items):
        """items: (text, candidate ids). -> per item a tensor [layers * heads, candidates] of gated attention."""
        out = []
        for a in range(0, len(items), 16):
            group = items[a : a + 16]
            seqs = []
            for text, cands in group:
                order = list(range(len(cands)))
                rng.shuffle(order)
                ids, att, block = [], [], []
                for pos, j in enumerate(order):
                    head_t, lead_t, _ = mem.store.note(cands[j])
                    b = [mem.bos] + head_t + lead_t[:lead]
                    ids += b
                    att += [1] * len(b)
                    block += [j] * len(b)
                tail = mem.question_ids(text)
                ids += [mem.pad] * cfg.gap + tail
                att += [0] * cfg.gap + [1] * len(tail)
                block += [-1] * (cfg.gap + len(tail))
                seqs.append((ids, att, block))
            width = max(len(s[0]) for s in seqs)
            I = torch.full((len(seqs), width), mem.pad, dtype=torch.long)
            A = torch.zeros((len(seqs), width), dtype=torch.long)
            B = torch.full((len(seqs), width), -1, dtype=torch.long)
            for r, (i_, a_, b_) in enumerate(seqs):
                I[r, width - len(i_) :], A[r, width - len(i_) :], B[r, width - len(i_) :] = torch.tensor(i_), torch.tensor(a_), torch.tensor(b_)
            I, A, B = I.to(device), A.to(device), B.to(device)
            with torch.inference_mode():
                backbone(input_ids=I, attention_mask=A, position_ids=(A.cumsum(-1) - 1).masked_fill(A == 0, 1))
            onehot = F.one_hot(B.clamp(min=0), n_cand).float() * (B >= 0).float()[:, :, None]  # [B, T, cand]
            per = torch.stack([torch.einsum("bht,btc->bhc", caught[i][0], onehot) * caught[i][1][:, :, None] for i in layers], dim=1)  # [B, layers, heads, cand]
            for r, (_, cands) in enumerate(group):
                out.append(per[r].reshape(-1, n_cand)[:, : len(cands)].clone().cpu())
        return out

    def name_examples(items, text_of, limit):
        ex = []
        for it in items:
            text, gold = text_of(it), set(it["gold"])
            for _, kind, cands, flags in mem.groups(text)[1]:
                cands, flags = cands[:n_cand], flags[:n_cand]
                if kind == "name" and len(cands) >= 2:
                    hit = [j for j, i in enumerate(cands) if i in gold]
                    ex.append((text, cands, hit[0] if hit else -1, flags))
            if len(ex) >= limit:
                break
        return ex

    quiz = [it for it in train if it["src"] in ("popqa", "hotpot") and not it["text"].startswith(INSTRUCTION)]
    rng.shuffle(quiz)
    tr = name_examples(quiz, lambda it: it["text"], n_train)
    neg_tr = name_examples([it for it in train if it["src"] == "gsm8k_train"], lambda it: it["text"], n_train // 3)
    dv = name_examples(dev["popqa"] + dev["hotpot"], lambda it: INSTRUCTION + it["q"], 10**9)
    neg_dv = name_examples(dev["gsm8k"], lambda it: it["q"], 10**9)
    if not tr or not dv:
        return {"skipped": "no ambiguous names"}

    old = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    handles = [backbone.layers[i].self_attn.register_forward_hook(hook(i), with_kwargs=True) for i in layers]
    try:
        f_tr, f_neg_tr, f_dv, f_neg_dv = (collect([(e[0], e[1]) for e in ex]) for ex in (tr, neg_tr, dv, neg_dv))
    finally:
        for h in handles:
            h.remove()
        model.config._attn_implementation = old

    # --- which article? learned weights over the 96 (layer, head) pairs, trained on training questions
    n_feat = f_tr[0].shape[0]
    w = torch.zeros(n_feat, requires_grad=True)
    scale = torch.tensor(3.0, requires_grad=True)
    inside_tr = [(f, e[2]) for f, e in zip(f_tr, tr) if e[2] >= 0]
    opt = torch.optim.Adam([w, scale], lr=0.05)
    for _ in range(60):
        loss = sum(F.cross_entropy(((w[:, None] * torch.log(f + 1e-6)).sum(0) * scale.exp() / n_feat)[None], torch.tensor([y])) for f, y in inside_tr) / max(len(inside_tr), 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    inside = [(f, e[2], e[0], e[1], e[3]) for f, e in zip(f_dv, dv) if e[2] >= 0]
    acc = lambda pick: round(sum(pick(f) == y for f, y, _, _, _ in inside) / max(len(inside), 1), 4)
    with torch.no_grad():
        which = {"n": len(inside), "most_developed": round(sum(y == 0 for _, y, _, _, _ in inside) / max(len(inside), 1), 4),
                 "attention_total": acc(lambda f: int(f.sum(0).argmax())), "attention_learned_weights": acc(lambda f: int((w[:, None] * torch.log(f + 1e-6)).sum(0).argmax()))}
        best_single = max(range(n_feat), key=lambda k: sum(int(f[k].argmax()) == y for f, y in inside_tr))
        which["attention_best_single_head"] = {"layer": layers[best_single // (n_feat // len(layers))], "head": best_single % (n_feat // len(layers)), "accuracy": acc(lambda f: int(f[best_single].argmax()))}
    # the head on the very same examples
    q = model._mem_states([mem.question_ids(t) for _, _, t, _, _ in inside], last=True, batch=128)
    right = 0
    for n, (_, y, _, cands, flags) in enumerate(inside):
        picks = model._mem_choose(q[n], cands, flags)
        right += bool(picks) and picks[0] == cands[y]
    which["head_v2_first_choice"] = round(right / max(len(inside), 1), 4)

    # --- any note at all? how much the prompt looks at the notes, quiz questions against maths problems
    def auc(pos, neg):
        if not pos or not neg:
            return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return round(wins / (len(pos) * len(neg)), 4)

    total = lambda f: float(f.sum())
    feats = lambda f: torch.log(f.sum(1) + 1e-6)
    whether = {"quiz_prompts": len(inside), "maths_prompts": len(f_neg_dv), "auc_total_attention": auc([total(f) for f, _, _, _, _ in inside], [total(f) for f in f_neg_dv])}
    if f_neg_tr and inside_tr:
        X = torch.stack([feats(f) for f, _ in inside_tr] + [feats(f) for f in f_neg_tr])
        Y = torch.tensor([1.0] * len(inside_tr) + [0.0] * len(f_neg_tr))
        mu, sd = X.mean(0), X.std(0) + 1e-6
        wl, bl = torch.zeros(n_feat, requires_grad=True), torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([wl, bl], lr=0.05, weight_decay=1e-3)
        for _ in range(200):
            loss = F.binary_cross_entropy_with_logits(((X - mu) / sd) @ wl + bl, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            sc = lambda f: float(((feats(f) - mu) / sd) @ wl + bl)
            whether["auc_learned_weights"] = auc([sc(f) for f, _, _, _, _ in inside], [sc(f) for f in f_neg_dv])
    return {"which_article": which, "any_note": whether, "train_examples": len(inside_tr), "maths_train_examples": len(f_neg_tr), "candidates_shown": n_cand, "lead_tokens_shown": lead}
