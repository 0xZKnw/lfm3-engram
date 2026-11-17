"""D12: audit of things in our own setup that may cost accuracy.

1. Repetition penalty. The model's generation_config sets repetition_penalty=1.1, which every run so far inherited.
   It lowers the score of any token already in the input, including the tokens of the note: copying the answer
   from the note is penalised. Compared: 1.1 on everything (what we ran), no penalty, 1.1 on everything except the
   note.
2. Batching. Are left-padded batches equal to one question at a time? Does the empty layout (256 masked tokens, no
   note) change the answers?
3. Answer length. We stop at 40 new tokens: are answers cut before the model states them?

PopQA, no thinking, right article given (summary, 96 tokens), complete dictionary."""
import json
import random
import sqlite3
import time

from .d7_fr import fold

GAP = 256


def run(data_dir: str, popqa_dir: str, n=600, seed=0):
    import shutil

    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    from .common import load_model, load_tokenizer

    t_start = time.time()
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id
    shutil.copy(f"{data_dir}/notes3.db", "/tmp/notes3.db")
    db = sqlite3.connect("/tmp/notes3.db")

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def chat_ids(q):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": f"Answer the question in a few words.\n\n{q}"}], add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    def note(title):
        row = db.execute("SELECT title, description, lead FROM articles WHERE title=? LIMIT 1", (title,)).fetchone()
        if not row:
            return None
        return [bos] + enc(f"Reference note about {row[0]}{f' ({row[1]})' if row[1] else ''}. Use it only if it is relevant, otherwise ignore it.\n") + enc(row[2])[:96]

    think = enc("<think>")
    no_think = {"suppress_tokens": think} if len(think) == 1 else {}

    class PenaltyOutsideMemory(LogitsProcessor):
        """Repetition penalty that ignores the tokens of the memory (and padding)."""

        def __init__(self, counted, penalty):
            self.counted, self.penalty = counted, penalty  # counted: [B, width] bool for the prompt part

        def __call__(self, input_ids, scores):
            width = self.counted.shape[1]
            mask = torch.cat([self.counted, torch.ones_like(input_ids[:, width:], dtype=torch.bool)], dim=1)
            ids = torch.where(mask, input_ids, torch.full_like(input_ids, pad))
            picked = scores.gather(1, ids)
            picked = torch.where(picked < 0, picked * self.penalty, picked / self.penalty)
            out = scores.scatter(1, ids, picked)
            out[:, pad] = scores[:, pad]
            return out

    def generate(seqs, mode="default", batch=32, max_new=40):
        """seqs: (ids, attention mask, counted-for-penalty mask). mode: default (1.1 on all) | none | outside_memory."""
        outs, lens = [], []
        for i in range(0, len(seqs), batch):
            group = seqs[i : i + batch]
            width = max(len(x[0]) for x in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            cnt = torch.zeros((len(group), width), dtype=torch.bool)
            for r, (x, m, c) in enumerate(group):
                ids[r, width - len(x) :], att[r, width - len(x) :], cnt[r, width - len(x) :] = torch.tensor(x), torch.tensor(m), torch.tensor(c, dtype=torch.bool)
            kw = dict(no_think)
            if mode == "none":
                kw["repetition_penalty"] = 1.0
            elif mode == "outside_memory":
                kw["repetition_penalty"] = 1.0
                kw["logits_processor"] = LogitsProcessorList([PenaltyOutsideMemory(cnt.to(dev), 1.1)])
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False, **kw)
            new = gen[:, width:]
            outs += [o.strip() for o in tok.batch_decode(new, skip_special_tokens=True)]
            lens += [int((row != pad).sum().item()) for row in new]
        return outs, lens

    def plain(p):
        return p, [1] * len(p), [1] * len(p)

    def with_note(block, p, gap=GAP):
        return block + [pad] * gap + p, [1] * len(block) + [0] * gap + [1] * len(p), [0] * (len(block) + gap) + [1] * len(p)

    def acc(outs, items):
        return round(sum(any(fold(g) in fold(o) for g in it["golds"] if len(g) >= 3) for o, it in zip(outs, items)) / len(items), 4)

    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(seed).shuffle(popqa)
    items = []
    for r in popqa:
        block = note(r["s_wiki_title"])
        if block:
            items.append({"q": r["question"], "golds": json.loads(r["possible_answers"]), "block": block, "prompt": chat_ids(r["question"])})
        if len(items) >= n:
            break
    result = {"n": len(items), "generation_config": {k: getattr(model.generation_config, k, None) for k in ("do_sample", "temperature", "top_k", "repetition_penalty")}}

    # 1. repetition penalty
    rp = {}
    for mode in ("default", "none", "outside_memory"):
        base, _ = generate([plain(it["prompt"]) for it in items], mode)
        mem, _ = generate([with_note(it["block"], it["prompt"]) for it in items], mode)
        rp[mode] = {"no_memory": acc(base, items), "right_note": acc(mem, items)}
    result["repetition_penalty"] = rp

    # 2. batching and layout (default penalty, as in every run so far)
    sub = items[:200]
    one, _ = generate([plain(it["prompt"]) for it in sub], batch=1)
    many, _ = generate([plain(it["prompt"]) for it in sub], batch=48)
    empty, _ = generate([(([pad] * GAP) + it["prompt"], [0] * GAP + [1] * len(it["prompt"]), [0] * GAP + [1] * len(it["prompt"])) for it in sub], batch=48)
    result["batching"] = {
        "n": len(sub),
        "one_at_a_time_accuracy": acc(one, sub), "batched_accuracy": acc(many, sub), "empty_layout_accuracy": acc(empty, sub),
        "batched_identical_to_single": round(sum(a == b for a, b in zip(one, many)) / len(sub), 4),
        "empty_layout_identical_to_batched": round(sum(a == b for a, b in zip(many, empty)) / len(sub), 4),
    }

    # 3. answer length
    out40, len40 = generate([plain(it["prompt"]) for it in items], max_new=40)
    out96, _ = generate([plain(it["prompt"]) for it in items], max_new=96)
    mem40, mlen40 = generate([with_note(it["block"], it["prompt"]) for it in items], max_new=40)
    mem96, _ = generate([with_note(it["block"], it["prompt"]) for it in items], max_new=96)
    result["answer_length"] = {
        "no_memory": {"cut_at_40_tokens": round(sum(x >= 40 for x in len40) / len(items), 4), "accuracy_40": acc(out40, items), "accuracy_96": acc(out96, items)},
        "right_note": {"cut_at_40_tokens": round(sum(x >= 40 for x in mlen40) / len(items), 4), "accuracy_40": acc(mem40, items), "accuracy_96": acc(mem96, items)},
    }
    result["seconds_total"] = round(time.time() - t_start)
    return result
