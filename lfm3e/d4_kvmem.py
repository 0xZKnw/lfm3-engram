"""D4: does an entity-keyed attention memory work on the frozen model, with no training at all?

In LFM3 anything further back than 256 tokens is visible only to the 6 global-attention layers (the convolutions see
7 tokens, the sliding-window attentions 256). So "the model read a passage long ago" is fully described by the
passage's keys and values at those 6 layers. Storing them per entity and making them attendable when the entity's
name shows up is the proposed memory.

This script emulates that exactly with plain forwards: [passage] [256 masked pad tokens] [chat prompt]. The masked
gap hides the passage from the convolutions and the sliding-window layers of the prompt; the global layers skip the
pads and still see the passage, with ordinary rotary distances.

Conditions, on PopQA questions the model fails (and on questions it gets right, to measure damage):
    in_prompt      passage pasted inside the user turn (classic RAG prompt)
    prefix_contig  passage right before the chat prompt, no gap
    memory         passage + gap + prompt          <- the proposed memory
    wrong_memory   another entity's passage + gap + prompt
    distractors    3 other passages and the right one, each followed by a gap (name collisions)"""
import json
import random
import time

from .d3_entity import ABSTAIN, is_correct

GAP = 256


def run(data_dir: str, n_baseline=5000, n_fail=400, n_ok=150, n_think=120, lens=(32, 64, 128), seed=0):
    import torch

    from .common import load_model, load_tokenizer

    rng = random.Random(seed)
    t_start = time.time()
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id
    qa = json.load(open(f"{data_dir}/popqa.json"))
    articles = json.load(open(f"{data_dir}/articles.json"))

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def chat_ids(question, context=None):
        content = f"Answer the question in a few words.\n\n{question}"
        if context:
            content = f"Use this text to answer.\n\n{context}\n\n{content}"
        text = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = enc(text)
        return ids if ids[0] == bos else [bos] + ids

    think_ids = enc("<think>")
    no_think = {"suppress_tokens": think_ids} if len(think_ids) == 1 else {}

    def generate(seqs, batch=32, max_new=40, think=False):
        """seqs: list of (ids, mask) with mask 0 on tokens the model must not see."""
        extra = {} if think else no_think
        outs = []
        for i in range(0, len(seqs), batch):
            group = seqs[i : i + batch]
            width = max(len(s[0]) for s in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, (s, m) in enumerate(group):
                ids[r, width - len(s) :] = torch.tensor(s)
                att[r, width - len(s) :] = torch.tensor(m)
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False, **extra)
            outs += [o.strip() for o in tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
        return outs

    def plain(ids):
        return ids, [1] * len(ids)

    def with_memory(passages, prompt):
        ids, mask = [], []
        for p in passages:
            ids += p + [pad] * GAP
            mask += [1] * len(p) + [0] * GAP
        return ids + prompt, mask + [1] * len(prompt)

    # ---- closed-book baseline
    rng.shuffle(qa)
    items = []
    for r in qa:
        if r["s_wiki_title"] in articles and r["subj"]:
            items.append({"q": r["question"], "answers": json.loads(r["possible_answers"]), "title": r["s_wiki_title"], "pop": int(r["s_pop"])})
        if len(items) >= n_baseline:
            break
    for it, out in zip(items, generate([plain(chat_ids(it["q"])) for it in items], batch=48)):
        it["base_out"] = out
        it["status"] = "correct" if is_correct(out, it["answers"]) else ("abstain" if ABSTAIN.search(out) else "wrong")
    result = {"baseline": {"n": len(items), **{s: round(sum(it["status"] == s for it in items) / len(items), 4) for s in ("correct", "wrong", "abstain")}}}

    fail, ok, seen = [], [], set()
    for it in items:
        group, limit = (ok, n_ok) if it["status"] == "correct" else (fail, n_fail)
        if len(group) >= limit or it["title"] in seen:
            continue
        seen.add(it["title"])
        it["art"] = enc(f"{it['title']}\n\n{articles[it['title']]}")[: max(lens)]
        it["prompt"] = chat_ids(it["q"])
        group.append(it)
    result["selection"] = {"fail": len(fail), "ok": len(ok)}

    def passage(it, n):
        return [bos] + it["art"][:n]

    def rate(outs, group, subset=None):
        idx = range(len(group)) if subset is None else subset
        idx = list(idx)
        return round(sum(is_correct(outs[i], group[i]["answers"]) for i in idx) / len(idx), 4) if idx else None

    result["by_passage_len"] = {}
    for n in lens:
        other_f = fail[1:] + fail[:1]
        other_o = ok[1:] + ok[:1]
        covered = [i for i, it in enumerate(fail) if is_correct(tok.decode(it["art"][:n]), it["answers"])]
        runs = {
            "in_prompt": generate([plain(chat_ids(it["q"], tok.decode(it["art"][:n]))) for it in fail]),
            "prefix_contig": generate([plain(passage(it, n) + it["prompt"]) for it in fail]),
            "memory": generate([with_memory([passage(it, n)], it["prompt"]) for it in fail]),
            "wrong_memory": generate([with_memory([passage(o, n)], it["prompt"]) for it, o in zip(fail, other_f)]),
        }
        ok_mem = generate([with_memory([passage(it, n)], it["prompt"]) for it in ok])
        ok_wrong = generate([with_memory([passage(o, n)], it["prompt"]) for it, o in zip(ok, other_o)])
        res = {
            "answer_in_passage": round(len(covered) / len(fail), 4),
            "fix_rate_all_failures": {k: rate(v, fail) for k, v in runs.items()},
            "fix_rate_when_answer_in_passage": {k: rate(v, fail, covered) for k, v in runs.items()},
            "abstain_rate_wrong_memory": round(sum(bool(ABSTAIN.search(o)) for o in runs["wrong_memory"]) / len(fail), 4),
            "still_correct_with_own_memory": rate(ok_mem, ok),
            "still_correct_with_wrong_memory": rate(ok_wrong, ok),
            "examples": [
                {"q": it["q"], "answers": it["answers"][:2], "before": it["base_out"][:80], "memory": runs["memory"][i][:80], "wrong_memory": runs["wrong_memory"][i][:80]}
                for i, it in enumerate(fail[:8])
            ],
        }
        if n == 64:
            seqs = []
            for i, it in enumerate(fail):
                others = [passage(fail[(i + k) % len(fail)], n) for k in (1, 2, 3)]
                others.insert(rng.randrange(4), passage(it, n))
                seqs.append(with_memory(others, it["prompt"]))
            outs = generate(seqs, batch=16)
            res["fix_rate_all_failures"]["distractors"] = rate(outs, fail)
            res["fix_rate_when_answer_in_passage"]["distractors"] = rate(outs, fail, covered)
        result["by_passage_len"][str(n)] = res
    # ---- default behaviour (thinking allowed), on a subset: does the memory still help, and does it shorten thinking?
    sub = fail[:n_think]
    if sub:
        def final(o):
            return o.split("</think>")[-1] if ("</think>" in o or "<think>" not in o) else None

        res = {"n": len(sub)}
        for name, seqs in (
            ("no_memory", [plain(it["prompt"]) for it in sub]),
            ("memory_64", [with_memory([passage(it, 64)], it["prompt"]) for it in sub]),
        ):
            finals = [final(o) for o in generate(seqs, max_new=768, think=True)]
            res[name] = {
                "correct": round(sum(f is not None and is_correct(f, it["answers"]) for f, it in zip(finals, sub)) / len(sub), 4),
                "unfinished_after_768_tokens": round(sum(f is None for f in finals) / len(sub), 4),
            }
        result["thinking_allowed"] = res
    result["seconds_total"] = round(time.time() - t_start)
    return result
