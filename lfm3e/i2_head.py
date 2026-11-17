"""I2: the article-choice head, retrained in the format of the integrated model, then exported.

Same head as D11 and D14 (two linear maps on frozen states, the article size as a prior, a 'none of them'
option), with three differences:
  - questions, notes and keys are built by the integrated model's own code (modeling_lfm3_mem.py), so what the
    head sees in training is what it sees at run time
  - each question is seen both alone and behind the instruction used in the measurements
  - a second head also gets prompts that are not knowledge questions (maths word problems, GSM8K train), where
    the right choice for every name is 'none'

Rule fixed before the run: the head trained with these extra prompts is the one exported if the share of test
questions that get the right note falls by less than 1 point on PopQA and on HotpotQA, compared with the head
trained on questions only. Otherwise the questions-only head is exported.

The test sets are the frozen ones of D14. No question that shares an article, a name or its text with them is
used for training."""
import gzip
import json
import random
import re
import time
import urllib.request

from .scoring import matches
from .triggers import norm, pick_spans

K = 12
INSTRUCTION = "Answer the question in a few words.\n\n"
MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def gold_by_title(mem, title):
    """The article with exactly this title, or None. The store has no title index: the name index gives the
    candidates, its 'exact' flag tells a bare title from a qualified one ('Skin' / 'Skin (2018 film)')."""
    base = re.sub(r"\s*\([^)]*\)$", "", title)
    for i, exact in mem.store.ids_for(norm(base), "en"):
        if exact == int(base == title):
            t = mem.title(i)
            if t == title or t.startswith(title + " ("):
                return i
    return None


def frozen_tests(mem, popqa_dir, data_dir, n_eval=1000, n_hotpot_eval=250, light=False):
    """The four test sets of D14, same questions in the same order. Gold articles are ids of the note store."""
    def by_qid(uri):
        q = (uri or "").rstrip("/").split("/")[-1]
        return mem.store.by_qid(int(q[1:]), "en") if q[:1] == "Q" and q[1:].isdigit() else None

    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)
    for r in popqa:
        r["gold"] = {i for i in [by_qid(r["s_uri"])] if i is not None}
    tests = {"popqa": [{"q": r["question"], "golds": json.loads(r["possible_answers"]), "gold": r["gold"]} for r in popqa[:n_eval]]}
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    tests["hotpot"] = [{"q": x["q"], "golds": [x["answer"]], "gold": {i for i in (gold_by_title(mem, t) for t in x["titles"]) if i is not None}, "n_titles": len(x["titles"])} for x in bench["hotpot"][:n_hotpot_eval]]
    if not light:
        from datasets import load_dataset

        tq = []
        for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
            if len(x["question"]) < 300:
                tq.append({"q": x["question"], "golds": [x["answer"]["value"]] + list(x["answer"]["aliases"]), "gold": set()})
            if len(tq) >= 5000 + n_eval:
                break
        random.Random(0).shuffle(tq)  # same held-out 1,000 as D9, D11 and D14
        tests["triviaqa"] = tq[5000:]
        with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
            raw = gzip.decompress(r.read()).decode("utf-8").splitlines()
        mk = []
        for line in raw:
            x = json.loads(line)
            ge = [s for a in x["answers"].get("en", []) if a.get("type") == "entity" and a.get("text") for s in [a["text"]] + [al for al in a.get("aliases", []) if al]]
            gf = [a for a in x["answers"].get("fr", []) if a.get("type") == "entity" and a.get("text")]
            if ge and gf and x["queries"].get("fr"):
                mk.append({"q": x["query"], "golds": list(dict.fromkeys(ge)), "gold": set()})
        random.Random(0).shuffle(mk)
        tests["mkqa_en"] = mk[:n_eval]
    return tests, popqa[n_eval:], bench


def prepare(mem, popqa_dir: str, data_dir: str, cap: int = 0, n_hotpot: int = 7500, light: bool = False, seed: int = 0):
    """CPU: the frozen tests, the training questions and the other prompts, with their gold article ids. Needs
    the note store and the tokenizer, not the model."""
    rng = random.Random(seed)

    def names(q):
        """Names the system looks up in this text, plus those the rule of D5 would have picked."""
        has = type("Has", (), {"__contains__": lambda s, v: bool(mem.candidates(v, "en"))})()
        return set(mem.spans(q, "en")) | {s for _, _, s in pick_spans(q, has)}

    tests, popqa_rest, bench = frozen_tests(mem, popqa_dir, data_dir, n_eval=(cap or 1000), n_hotpot_eval=(min(cap, 250) or 250), light=light)
    test_gold = {i for items in tests.values() for it in items for i in it["gold"]}
    test_names = {s for items in tests.values() for it in items for s in names(it["q"])}
    test_questions = {norm(it["q"]) for items in tests.values() for it in items}

    def clean(q, gold):
        return norm(q) not in test_questions and not (gold & test_gold) and not (names(q) & test_names)

    qa = [{"src": "popqa", "q": r["question"], "gold": r["gold"]} for r in popqa_rest if r["gold"] and clean(r["question"], r["gold"])]
    if cap:
        qa = qa[: 4 * cap]
    hp = []
    if light:
        for x in bench["hotpot"][250:]:
            gold = {i for i in (gold_by_title(mem, t) for t in x["titles"]) if i is not None}
            if len(gold) == 2 and clean(x["q"], gold):
                hp.append({"src": "hotpot", "q": x["q"], "gold": gold})
        neg_src = [x["q"] for x in bench["gsm8k"][150:]]
        gsm_test = [x["q"] for x in bench["gsm8k"][:150]]
    else:
        from datasets import load_dataset

        hp_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        for i in rng.sample(range(len(hp_train)), min(len(hp_train), 20000)):
            x = hp_train[i]
            gold = {i for i in (gold_by_title(mem, t) for t in sorted(set(x["supporting_facts"]["title"]))) if i is not None}
            if len(gold) == 2 and clean(x["question"], gold):
                hp.append({"src": "hotpot", "q": x["question"], "gold": gold})
            if len(hp) >= (4 * cap if cap else n_hotpot):
                break
        neg_src = [x["question"] for x in load_dataset("openai/gsm8k", "main", split="train")]
        gsm_test = [x["q"] for x in bench["gsm8k"]]
    if cap:
        hp, neg_src, gsm_test = hp[: 4 * cap], neg_src[: 4 * cap], gsm_test[:cap]
    neg = [{"src": "gsm8k", "q": q, "gold": set()} for q in neg_src if clean(q, set())]
    plain = lambda items: [{**it, "gold": sorted(it["gold"])} for it in items]
    return {"tests": {k: plain(v) for k, v in tests.items()}, "qa": plain(qa), "hp": plain(hp), "neg": plain(neg), "gsm_test": gsm_test, "cap": cap, "light": light}


def run(hf_dir: str, popqa_dir: str, data_dir: str, web_tokens: str = "", out_dir: str = "", cap: int = 0, n_hotpot: int = 7500, n_preview: int = 200,
        light: bool = False, device: str = "cuda", seed: int = 0, prepared=None):
    """hf_dir: an exported model folder without head (export_hf.export). cap > 0 shortens everything (dry run).
    light: no dataset download, the benchmark file is split instead (local test on CPU)."""
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf_dir, trust_remote_code=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    mem = model.memory
    assert model.__dict__["_mem_head_cpu"] is None, "the folder already has a head"
    token_freq = None
    if web_tokens and os.path.exists(web_tokens):
        web = np.load(web_tokens)
        token_freq = (np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))).astype(np.float32)
        mem.token_freq = token_freq
    timing = {"load_s": round(time.time() - t_start)}

    t0 = time.time()
    if prepared is None:
        prepared = prepare(mem, popqa_dir, data_dir, cap=cap, n_hotpot=n_hotpot, light=light, seed=seed)
    tests = {k: [{**it, "gold": set(it["gold"])} for it in v] for k, v in prepared["tests"].items()}
    qa, hp, neg = ([{**it, "gold": set(it["gold"])} for it in prepared[k]] for k in ("qa", "hp", "neg"))
    gsm_test = prepared["gsm_test"]
    timing["data_s"] = round(time.time() - t0)

    print(f"[i2] data ready: {len(qa)} popqa, {len(hp)} hotpot, {len(neg)} other prompts, {timing['data_s']} s", flush=True)
    # ------------------------------------------------------------ examples: (text, candidates, right one or K)
    def examples(items, variants):
        texts, ex = [], []
        for it in items:
            for text in variants(it["q"]):
                cands = [c for c in (mem.candidates(s, "en") for s in mem.spans(text, "en")) if c]
                if not cands:
                    continue
                texts.append(text)
                for cs in cands:
                    hit = [j for j, i in enumerate(cs) if i in it["gold"]]
                    ex.append((len(texts) - 1, cs, hit[0] if hit else K))
        return texts, ex

    t0 = time.time()
    qa_texts, qa_ex = examples(qa + hp, lambda q: (q, INSTRUCTION + q))
    neg_texts, neg_ex = examples(neg, lambda q: (q,))
    all_ids = sorted({i for _, cs, _ in qa_ex + neg_ex for i in cs})
    row = {i: n for n, i in enumerate(all_ids)}
    with torch.inference_mode():
        keys = torch.cat([model._mem_states([mem.key(i) for i in all_ids[a : a + 4096]], last=False, batch=256).half() for a in range(0, len(all_ids), 4096)]) if all_ids else torch.zeros(0, model.config.hidden_size, device=device).half()
        q_qa = model._mem_states([mem.question_ids(t) for t in qa_texts], last=True, batch=256)
        q_neg = model._mem_states([mem.question_ids(t) for t in neg_texts], last=True, batch=256)
    keys, q_qa, q_neg = keys.clone(), q_qa.clone(), q_neg.clone()  # out of inference mode: plain tensors
    prior_all = torch.tensor([mem.prior(i) for i in all_ids], device=device)
    timing["states_s"] = round(time.time() - t0)

    print(f"[i2] states done: {len(all_ids)} keys, {len(qa_texts) + len(neg_texts)} texts, {timing['states_s']} s", flush=True)
    def pack(q_states, ex):
        qi = torch.tensor([e[0] for e in ex], device=device, dtype=torch.long)
        cand = torch.zeros(len(ex), K, dtype=torch.long, device=device)
        mask = torch.zeros(len(ex), K, dtype=torch.bool, device=device)
        for n, (_, cs, _) in enumerate(ex):
            cand[n, : len(cs)] = torch.tensor([row[i] for i in cs], device=device)
            mask[n, : len(cs)] = True
        return q_states[qi], cand, mask, torch.tensor([e[2] for e in ex], device=device, dtype=torch.long)

    def scores(h, q, cand, mask):
        kn = (keys[cand].float() - h["k_mu"]) / h["k_sd"]
        s = torch.einsum("bd,bkd->bk", F.normalize(((q - h["q_mu"]) / h["q_sd"]) @ h["q"].T, dim=-1), F.normalize(kn @ h["k"].T, dim=-1)) * h["log_t"].exp() + h["w_prior"] * prior_all[cand]
        return torch.cat([s.masked_fill(~mask, -1e4), h["null"].expand(len(q), 1)], dim=1)

    def fit(parts, epochs=40, dim=256):
        """parts: packed example sets, concatenated. Same recipe as D11: AdamW, 40 passes, batches of 512."""
        torch.manual_seed(seed)
        q, cand, mask, y = (torch.cat(x) for x in zip(*parts))
        d = q.shape[1]
        h = {"q": torch.nn.Linear(d, dim, bias=False).weight.detach().to(device), "k": torch.nn.Linear(d, dim, bias=False).weight.detach().to(device),
             "w_prior": torch.zeros(1, device=device), "null": torch.zeros(1, device=device), "log_t": torch.full((1,), 2.5, device=device)}
        used = keys[cand[mask].unique()].float()
        h.update(q_mu=q.mean(0, keepdim=True), q_sd=q.std(0, keepdim=True) + 1e-6, k_mu=used.mean(0, keepdim=True), k_sd=used.std(0, keepdim=True) + 1e-6)
        params = [h[k].requires_grad_(True) for k in ("q", "k", "w_prior", "null", "log_t")]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=device)
            for a in range(0, len(perm), 512):
                idx = perm[a : a + 512]
                loss = F.cross_entropy(scores(h, q[idx], cand[idx], mask[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {k: v.detach() for k, v in h.items()}

    t0 = time.time()
    packed_qa, packed_neg = pack(q_qa, qa_ex), pack(q_neg, neg_ex)
    heads = {"none": None, "questions": fit([packed_qa])}
    if neg_ex:
        heads["questions_and_other_prompts"] = fit([packed_qa, packed_neg])
    timing["fit_s"] = round(time.time() - t0)

    print(f"[i2] heads fitted, {timing['fit_s']} s", flush=True)
    # ------------------------------------------------------------ what each head shows on the frozen tests
    def use(h):
        model.__dict__["_mem_head_cpu"] = None if h is None else {k: v.float().cpu() for k, v in h.items()}
        model.__dict__["_mem_head_dev"] = None

    def plans(questions, prefix=INSTRUCTION):
        out = []
        for a in range(0, len(questions), 256):
            out += model.memory_plan([mem.question_ids(prefix + q) for q in questions[a : a + 256]])
        return out

    def shown(name):
        r = {}
        for key, items in tests.items():
            p = plans([it["q"] for it in items])
            r[key] = {"n": len(items), "with_note": round(sum(bool(x) for x in p) / len(items), 4), "notes_per_question": round(sum(map(len, p)) / len(items), 3)}
            if key in ("popqa", "hotpot"):
                known = [(it, x) for it, x in zip(items, p) if it["gold"]]
                r[key].update(gold_known=len(known), right_note=round(sum(bool(it["gold"] & set(x)) for it, x in known) / max(len(known), 1), 4))
            if key == "hotpot":
                both = [(it, x) for it, x in zip(items, p) if len(it["gold"]) == 2]
                r[key]["both_notes"] = round(sum(it["gold"] <= set(x) for it, x in both) / max(len(both), 1), 4)
        p = plans(gsm_test, prefix="")
        r["gsm8k"] = {"n": len(gsm_test), "with_note": round(sum(bool(x) for x in p) / max(len(gsm_test), 1), 4), "notes_per_question": round(sum(map(len, p)) / max(len(gsm_test), 1), 3)}
        if name != "none":
            r["gsm8k"]["examples"] = [{"q": q[:160], "notes": [mem.title(i) for i in x]} for q, x in zip(gsm_test, p) if x][:5]
        return r

    def choice_accuracy(h):
        """On the names of the PopQA and HotpotQA tests that several articles share: how often the head's first
        choice is the right article when it is among the candidates, and how often it says 'none' when it is not."""
        texts, ex = examples([{"q": it["q"], "gold": it["gold"]} for key in ("popqa", "hotpot") for it in tests[key] if it["gold"]], lambda q: (INSTRUCTION + q,))
        ex = [e for e in ex if len(e[1]) >= 2]
        if not ex:
            return {}
        new = sorted({i for _, cs, _ in ex for i in cs} - set(row))
        nonlocal keys, prior_all
        if new:
            with torch.inference_mode():
                extra = model._mem_states([mem.key(i) for i in new], last=False, batch=256).half()
            keys = torch.cat([keys, extra.clone()])
            prior_all = torch.cat([prior_all, torch.tensor([mem.prior(i) for i in new], device=device)])
            for i in new:
                row[i] = len(row)
        with torch.inference_mode():
            q = model._mem_states([mem.question_ids(t) for t in texts], last=True, batch=256)
        qs, cand, mask, y = pack(q.clone(), ex)
        with torch.no_grad():
            pred = scores(h, qs, cand, mask).argmax(1)
        inside, outside = y < K, y == K
        return {"ambiguous_names": len(ex), "gold_among_candidates": int(inside.sum()), "first_choice_right": round((pred[inside] == y[inside]).float().mean().item(), 4) if inside.any() else None,
                "gold_absent": int(outside.sum()), "says_none_when_absent": round((pred[outside] == K).float().mean().item(), 4) if outside.any() else None,
                "baseline_most_developed": round((y[inside] == 0).float().mean().item(), 4) if inside.any() else None}

    t0 = time.time()
    report = {}
    for name, h in heads.items():
        use(h)
        report[name] = {"shown": shown(name)}
        if h is not None:
            report[name]["choice"] = choice_accuracy(h)
    timing["eval_s"] = round(time.time() - t0)

    chosen = "questions"
    if "questions_and_other_prompts" in heads:
        a, b = report["questions"]["shown"], report["questions_and_other_prompts"]["shown"]
        if all(a[k]["right_note"] - b[k]["right_note"] < 0.01 for k in ("popqa", "hotpot")):
            chosen = "questions_and_other_prompts"

    print(f"[i2] evaluation done, {timing['eval_s']} s; exporting {chosen}", flush=True)
    # ------------------------------------------------------------ export
    files = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, h in heads.items():
            if h is None:
                continue
            tensors = {k: v.float().cpu().numpy() for k, v in h.items()}
            if token_freq is not None:
                tensors["token_freq"] = token_freq
            for target in [f"head_{name}.safetensors"] + (["head.safetensors"] if name == chosen else []):
                save_file(tensors, f"{out_dir}/{target}", metadata={"trained_on": name, "format": "integrated model, see lfm3e/i2_head.py"})
                files[target] = os.path.getsize(f"{out_dir}/{target}")

    # ------------------------------------------------------------ preview: a few answers through generate()
    t0 = time.time()
    use(heads[chosen])
    preview = {}
    items = tests["popqa"][:n_preview]
    if items:
        prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + it["q"]}], add_generation_prompt=True, enable_thinking=False, tokenize=False) for it in items]
        right = {"memory": 0, "no_memory": 0}
        for a in range(0, len(items), 50):
            enc = tok(prompts[a : a + 50], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            for mode, on in (("memory", True), ("no_memory", False)):
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=40, do_sample=False, memory=on)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
                right[mode] += sum(matches(t, it["golds"]) for t, it in zip(texts, items[a : a + 50]))
        preview = {"n": len(items), "what": "first PopQA test questions, greedy, a preview and not a measurement", **{k: round(v / len(items), 4) for k, v in right.items()}}
    timing["preview_s"] = round(time.time() - t0)
    timing["total_s"] = round(time.time() - t_start)

    return {
        "sizes": {"questions": len(qa) + len(hp), "popqa": len(qa), "hotpot": len(hp), "other_prompts": len(neg), "examples_questions": len(qa_ex), "examples_other_prompts": len(neg_ex),
                  "keys": len(all_ids), "tests": {k: len(v) for k, v in tests.items()}, "gsm8k_test": len(gsm_test)},
        "labels": {"questions_none": round(sum(e[2] == K for e in qa_ex) / max(len(qa_ex), 1), 4)},
        "heads": report, "exported": chosen, "files": files, "preview": preview, "timing": timing, "cap": cap, "light": light,
    }
