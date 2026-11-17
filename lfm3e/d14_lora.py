"""D14: teach the model to read its notes. A LoRA on the six global-attention layers, active only when a note is
shown, plus the article-choice head trained on more than one benchmark.

Training examples (PopQA, HotpotQA and TriviaQA questions that share no entity or name with the tests):
    right note shown, alone or among homonyms    -> the answer
    only wrong notes (homonyms)                  -> what the model answers without notes if that is right,
    right article but the answer is not in it       otherwise "I don't know."
Where it exists, the target is the model's own correct answer (given the right note alone), so its style is kept.

Frozen tests, never used for training or tuning: PopQA (first 1,000 of the shuffled set, as since D5), TriviaQA
(the 1,000 held out since D9), HotpotQA validation (the 250 of D6), MKQA English (1,000, a benchmark that is not
in the training mix at all). Each is run without memory, with memory, and with memory plus the LoRA.

Answers are matched on whole words (lfm3e.scoring); the right article is found by Wikidata id for PopQA."""
import gzip
import json
import math
import random
import time
import urllib.request

from .d5_trigger import norm
from .d7_fr import MKQA_URL
from .mem import K, ChoiceHead, Memory, States, add_lora, generate, pad_left
from .scoring import matches

IDK = "I don't know."
ABSTAIN_WORDS = ("i don't know", "i do not know", "not sure", "no information", "cannot determine", "can't determine", "i'm not aware", "unable to")


def run(data_dir: str, popqa_dir: str, web_tokens: str, save_path: str = "", n_train=2500, n_eval=1000, n_hotpot_eval=250, epochs=3, rank=16, lr=2e-4, seed=0):
    import shutil

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    t_start = time.time()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tok = load_tokenizer()
    model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = "cuda"
    shutil.copy(f"{data_dir}/notes3.db", "/tmp/notes3.db")
    mem = Memory(tok, "/tmp/notes3.db", web_tokens)
    states = States(model, mem)
    timing = {}

    def qid_of(uri):
        return (uri or "").rstrip("/").split("/")[-1]

    def yes_no(text, golds):
        g = golds[0].strip().lower()
        if g in ("yes", "no"):
            import re

            m = re.search(r"\b(yes|no)\b", text.lower())
            return bool(m) and m.group(1) == g
        return None

    def right(text, golds):
        yn = yes_no(text, golds)
        return yn if yn is not None else matches(text, golds)

    def abstains(text):
        low = text.lower()
        return any(w in low for w in ABSTAIN_WORDS)

    # ------------------------------------------------------------ frozen tests
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)
    tests = {"popqa": [{"q": r["question"], "golds": json.loads(r["possible_answers"]), "gold_rows": [x for x in [mem.by_qid(qid_of(r["s_uri"]))] if x]} for r in popqa[:n_eval]]}
    bench = json.load(open(f"{data_dir}/benchmarks.json"))
    tests["hotpot"] = [{"q": x["q"], "golds": [x["answer"]], "gold_rows": [r for r in (mem.by_title(t) for t in x["titles"]) if r]} for x in bench["hotpot"][:n_hotpot_eval]]
    tq = []
    for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
        if len(x["question"]) < 300:
            tq.append({"q": x["question"], "golds": [x["answer"]["value"]] + list(x["answer"]["aliases"]), "gold_rows": []})
        if len(tq) >= 5000 + n_eval + 3 * n_train:
            break
    head_part, tq_rest = tq[: 5000 + n_eval], tq[5000 + n_eval :]
    random.Random(0).shuffle(head_part)  # same held-out 1,000 as D9 and D11
    tests["triviaqa"] = head_part[5000:]
    with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
        raw = gzip.decompress(r.read()).decode("utf-8").splitlines()
    mk = []
    for line in raw:
        x = json.loads(line)
        ge = [s for a in x["answers"].get("en", []) if a.get("type") == "entity" and a.get("text") for s in [a["text"]] + [al for al in a.get("aliases", []) if al]]
        gf = [a for a in x["answers"].get("fr", []) if a.get("type") == "entity" and a.get("text")]
        if ge and gf and x["queries"].get("fr"):
            mk.append({"q": x["query"], "golds": list(dict.fromkeys(ge)), "gold_rows": []})
    random.Random(0).shuffle(mk)
    tests["mkqa_en"] = mk[:n_eval]

    # nothing that shares an article or a name with a test question may be trained on
    test_titles = {r[0] for items in tests.values() for it in items for r in it["gold_rows"]}
    test_names = {s for items in tests.values() for it in items for s in mem.spans(it["q"])}
    test_questions = {norm(it["q"]) for items in tests.values() for it in items}

    def clean(q, gold_rows):
        return norm(q) not in test_questions and not any(r[0] in test_titles for r in gold_rows) and not any(s in test_names for s in mem.spans(q))

    # ------------------------------------------------------------ training questions
    pool = []
    for r in popqa[n_eval:]:
        rows = [x for x in [mem.by_qid(qid_of(r["s_uri"]))] if x]
        if rows and clean(r["question"], rows):
            pool.append({"src": "popqa", "q": r["question"], "golds": json.loads(r["possible_answers"]), "gold_rows": rows})
    hp_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    for i in rng.sample(range(len(hp_train)), min(len(hp_train), 8 * n_train)):
        x = hp_train[i]
        rows = [r for r in (mem.by_title(t) for t in sorted(set(x["supporting_facts"]["title"]))) if r]
        if len(rows) == 2 and clean(x["question"], rows):
            pool.append({"src": "hotpot", "q": x["question"], "golds": [x["answer"]], "gold_rows": rows})
        if sum(p["src"] == "hotpot" for p in pool) >= 3 * n_train:
            break
    for x in tq_rest:
        if clean(x["q"], []):
            pool.append({"src": "triviaqa", **x})
    by_src = {s: [p for p in pool if p["src"] == s] for s in ("popqa", "hotpot", "triviaqa")}
    for s in by_src:
        rng.shuffle(by_src[s])

    # ------------------------------------------------------------ the head, trained on PopQA and HotpotQA
    t0 = time.time()
    head_items = by_src["popqa"][: 4 * n_train] + by_src["hotpot"][: 2 * n_train]
    ex_q, ex_c, ex_y = [], [], []
    for qi, it in enumerate(head_items):
        gold = {r[0] for r in it["gold_rows"]}
        for surface in mem.spans(it["q"]):
            cs = mem.candidates(surface)
            titles = [c[0] for c in cs]
            hit = [j for j, t in enumerate(titles) if t in gold]
            ex_q.append(qi)
            ex_c.append(cs)
            ex_y.append(hit[0] if hit else K)
    states.add_keys([c for cs in ex_c for c in cs])
    q_head = states.questions([it["q"] for it in head_items])
    head = ChoiceHead(q_head.shape[1], dev)
    head.fit(q_head[ex_q], ex_c, ex_y, states)
    timing["head_s"] = round(time.time() - t0)

    def notes_for(q, q_state, top=3):
        """What the system shows: the head's picks for each name, then exact-title notes for the rest."""
        rows = []
        for surface in mem.spans(q):
            for c in head.choose(q_state, mem.candidates(surface), states, top=top):
                if c[0] not in [r[0] for r in rows]:
                    rows.append(c)
        for r in mem.strict(q):
            if r[0] not in [x[0] for x in rows] and not any(norm(r[0]).split(" (")[0] == norm(x[0]).split(" (")[0] for x in rows):
                rows.append(r)
        return rows[:3]

    def seq(rows, q):
        return mem.layout([mem.note(r) for r in rows], mem.chat(q))

    def has_answer(rows, golds):
        return any(matches(mem.note_text(r), golds) for r in rows)

    # ------------------------------------------------------------ training examples
    t0 = time.time()
    train_items = by_src["popqa"][-n_train:] + by_src["hotpot"][-n_train:] + by_src["triviaqa"][:n_train]  # disjoint from the head's HotpotQA part
    states.add_keys([c for it in train_items for s_ in mem.spans(it["q"]) for c in mem.candidates(s_)])
    q_train = states.questions([it["q"] for it in train_items])
    plain_out = generate(model, mem, [seq([], it["q"]) for it in train_items], batch=48)
    clean_idx = [i for i, it in enumerate(train_items) if it["gold_rows"]]
    clean_out = dict(zip(clean_idx, generate(model, mem, [seq(train_items[i]["gold_rows"], train_items[i]["q"]) for i in clean_idx])))
    examples, kinds = [], {}
    for i, it in enumerate(train_items):
        gold = it["gold_rows"]
        homonyms = [c for s in mem.spans(it["q"]) for c in mem.candidates(s) if c[0] not in {r[0] for r in gold}]
        rng.shuffle(homonyms)
        system = notes_for(it["q"], q_train[i])
        options = [("system", system)] if system else []
        if gold:
            options += [("right_alone", gold), ("right_with_homonyms", gold + homonyms[:2])]
        if homonyms:
            options.append(("homonyms_only", homonyms[:2]))
        if not options:
            continue
        kind, rows = rng.choice(options)
        rows = list(rows)[:3]
        rng.shuffle(rows)
        is_yn = yes_no("", it["golds"]) is not None
        usable = has_answer(rows, it["golds"]) or bool(is_yn and gold and all(g in rows for g in gold))
        if usable:
            own = clean_out.get(i)
            target = own if own and right(own, it["golds"]) and not abstains(own) else f"{it['golds'][0]}."
        else:
            target = plain_out[i] if right(plain_out[i], it["golds"]) else IDK
            kind += "_no_answer_in_notes"
        examples.append({"rows": rows, "q": it["q"], "target": target})
        kinds[kind] = kinds.get(kind, 0) + 1
    timing["examples_s"] = round(time.time() - t0)
    targets_idk = sum(e["target"] == IDK for e in examples)

    # ------------------------------------------------------------ LoRA training
    t0 = time.time()
    params, switch = add_lora(model, r=rank, alpha=2 * rank)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    built = []
    for e in examples:
        ids, att, _ = seq(e["rows"], e["q"])
        ans = mem.enc(e["target"])[:48] + [mem.eos]
        built.append((ids + ans, att + [1] * len(ans), len(ans)))
    order = sorted(range(len(built)), key=lambda i: len(built[i][0]))
    batches, cur, cur_w = [], [], 0
    for i in order:
        w = len(built[i][0])
        if cur and (len(cur) + 1) * max(cur_w, w) > 12_000:
            batches.append(cur)
            cur, cur_w = [], 0
        cur.append(i)
        cur_w = max(cur_w, w)
    if cur:
        batches.append(cur)
    steps, total = len(batches) * epochs, 0
    losses = []
    for ep in range(epochs):
        rng.shuffle(batches)
        for b in batches:
            (ids, att), width = pad_left([(built[i][0], built[i][1]) for i in b], mem.pad, dev)
            hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            rows_h, rows_y = [], []
            for r, i in enumerate(b):
                n = built[i][2]
                rows_h.append(hidden[r, width - n - 1 : width - 1])
                rows_y.append(ids[r, width - n :])
            logits = model.lm_head(torch.cat(rows_h)).float()
            loss = F.cross_entropy(logits, torch.cat(rows_y))
            opt.zero_grad()
            loss.backward()
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * total / max(steps, 1)))
            opt.step()
            total += 1
            losses.append(loss.item())
    switch.on = False
    timing["train_s"] = round(time.time() - t0)
    if save_path:
        torch.save({
            "lora": [p_.detach().cpu() for p_ in params], "rank": rank,
            "head": {"q": head.q.state_dict(), "k": head.k.state_dict(), "w_prior": head.w_prior.detach().cpu(), "null": head.null.detach().cpu(), "log_t": head.log_t.detach().cpu(), "stats": [t.cpu() for t in head.stats]},
        }, save_path)
    train_tokens = sum(len(x[0]) for x in built) * epochs

    # ------------------------------------------------------------ evaluation on the frozen tests
    t0 = time.time()
    result = {
        "training": {
            "pool": {s: len(v) for s, v in by_src.items()}, "head_examples": len(ex_y), "examples": len(examples), "kinds": kinds,
            "targets_i_dont_know": round(targets_idk / max(len(examples), 1), 4), "tokens": train_tokens, "steps": total,
            "loss_first_last": [round(sum(losses[:20]) / max(len(losses[:20]), 1), 3), round(sum(losses[-20:]) / max(len(losses[-20:]), 1), 3)],
            "lora_parameters": sum(p.numel() for p in params), "rank": rank,
        },
        "tests": {},
    }

    def rates(outs, items):
        ok = [right(o, it["golds"]) for o, it in zip(outs, items)]
        ab = [abstains(o) and not k for o, k in zip(outs, ok)]
        n = len(items)
        return ok, {"correct": round(sum(ok) / n, 4), "abstains": round(sum(ab) / n, 4), "wrong": round((n - sum(ok) - sum(ab)) / n, 4)}

    def run_test(items, rows_per_q, lora):
        """Questions without notes always go through the untouched model."""
        outs = [None] * len(items)
        with_n = [i for i, r in enumerate(rows_per_q) if r]
        without = [i for i, r in enumerate(rows_per_q) if not r]
        switch.on = False
        for i, o in zip(without, generate(model, mem, [seq([], items[i]["q"]) for i in without], batch=48)):
            outs[i] = o
        switch.on = lora
        for i, o in zip(with_n, generate(model, mem, [seq(rows_per_q[i], items[i]["q"]) for i in with_n])):
            outs[i] = o
        switch.on = False
        return outs

    for name, items in tests.items():
        states.add_keys([c for it in items for s_ in mem.spans(it["q"]) for c in mem.candidates(s_)])
        qs = states.questions([it["q"] for it in items])
        system_rows = [notes_for(it["q"], qs[i]) for i, it in enumerate(items)]
        ok0, res = rates(run_test(items, [[] for _ in items], False), items)
        out = {"n": len(items), "no_memory": res}
        plans = {"memory": (system_rows, False), "memory_lora": (system_rows, True)}
        if any(it["gold_rows"] for it in items):
            gold_rows = [it["gold_rows"] for it in items]
            wrong_rows = []
            for it in items:  # homonyms only: the notes the system must not be fooled by
                g = {r[0] for r in it["gold_rows"]}
                wrong_rows.append([c for s in mem.spans(it["q"]) for c in mem.candidates(s) if c[0] not in g][:2])
            plans.update({"right_notes": (gold_rows, False), "right_notes_lora": (gold_rows, True), "homonyms_only": (wrong_rows, False), "homonyms_only_lora": (wrong_rows, True)})
        for plan, (rows_per_q, lora) in plans.items():
            ok, res = rates(run_test(items, rows_per_q, lora), items)
            res.update(fixed=sum(a and not b for a, b in zip(ok, ok0)), broken=sum(b and not a for a, b in zip(ok, ok0)), note_shown=round(sum(bool(r) for r in rows_per_q) / len(items), 4))
            if any(it["gold_rows"] for it in items):
                res["right_article_shown"] = round(sum(any(r[0] in {g[0] for g in it["gold_rows"]} for r in rows) for rows, it in zip(rows_per_q, items)) / len(items), 4)
            out[plan] = res
        result["tests"][name] = out
    timing["eval_s"] = round(time.time() - t0)
    result["timing"] = timing
    result["seconds_total"] = round(time.time() - t_start)
    return result
