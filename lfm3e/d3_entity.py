"""D3: the Engram hypothesis in miniature, in assistant mode.

For entities the instruct model fails on in closed-book chat QA (PopQA), learn ONE vector per entity from the raw
text of its Wikipedia article. The vector is added to the residual stream at the last token of each mention of the
entity's name, at one layer. Then ask the question again in chat format, without the article, with the vector
injected at the name in the question.

    fix rate   : failures that become correct with the entity's own vector
    control    : same, with another entity's vector (measures noise, not knowledge)
    break rate : questions that were correct and become wrong
    ceiling    : failures answered correctly when the article excerpt is put in the prompt"""
import json
import os
import random
import re
import time

ABSTAIN = re.compile(
    r"don't know|do not know|not sure|no information|not aware|not familiar|cannot|can't|couldn't|unable|unclear|"
    r"no (?:specific|reliable|widely|public)|not (?:enough|widely|publicly)|i'm sorry|unknown",
    re.I,
)
TRAIN_LEN, FULL_LEN = 256, 384


def fetch_articles(out_dir: str):
    """CPU: PopQA questions + the English Wikipedia article of every question subject."""
    import glob
    from concurrent.futures import ProcessPoolExecutor

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    os.makedirs(out_dir, exist_ok=True)
    qa = [dict(r) for r in load_dataset("akariasai/PopQA", split="test")]
    titles = sorted({r["s_wiki_title"] for r in qa if r["s_wiki_title"]})
    root = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=["20231101.en/*.parquet"])
    files = sorted(glob.glob(f"{root}/20231101.en/*.parquet"))
    articles = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for part in pool.map(_scan, files, [titles] * len(files)):
            articles.update(part)
    with open(f"{out_dir}/popqa.json", "w") as f:
        json.dump(qa, f)
    with open(f"{out_dir}/articles.json", "w") as f:
        json.dump(articles, f)
    return {"questions": len(qa), "subjects": len(titles), "articles_found": len(articles), "parquet_files": len(files)}


def _scan(path, titles):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["title", "text"])
    sub = table.filter(pc.is_in(table["title"], value_set=pa.array(titles)))
    return {t: x[:12000] for t, x in zip(sub["title"].to_pylist(), sub["text"].to_pylist())}


def find_ends(canon_ids, patterns):
    """Positions of the last token of every occurrence of any pattern (lists of canonical ids)."""
    ends = set()
    n = len(canon_ids)
    for pat in patterns:
        k = len(pat)
        if k == 0:
            continue
        for i in range(n - k + 1):
            if canon_ids[i] == pat[0] and canon_ids[i : i + k] == pat:
                ends.add(i + k - 1)
    return sorted(ends)


def is_correct(text, answers):
    low = text.lower()
    return any(a.lower() in low for a in answers if a)


class Injector:
    """Adds `vectors[b]` at positions where `mask[b, t]` is 1, at the input of one layer. Only acts on forwards
    whose length matches the mask (the prompt prefill), so cached decode steps are left alone."""

    def __init__(self, model, layer):
        self.vectors = self.mask = None
        self.hook = model.model.layers[layer].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.vectors is None or args[0].shape[1] != self.mask.shape[1]:
            return None
        h = args[0]
        return (h + (self.mask[:, :, None] * self.vectors[:, None, :]).to(h.dtype),) + args[1:], kwargs

    def set(self, vectors=None, mask=None):
        self.vectors, self.mask = vectors, mask

    def remove(self):
        self.hook.remove()


def run(data_dir: str, layers=(3, 9, 15, 21), n_baseline=5000, n_think=400, n_fail=240, n_ok=80, epochs=40, lr=3e-3, cap=1.0, seed=0):
    import numpy as np
    import torch

    from .common import build_compress_table, load_model, load_tokenizer
    from .d0_speed import chunked_ce_backward

    rng = random.Random(seed)
    t_start = time.time()
    tok = load_tokenizer()
    tok.padding_side = "left"
    model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = "cuda"
    compress, _ = build_compress_table(tok)
    bos, pad = tok.bos_token_id, tok.pad_token_id
    qa = json.load(open(f"{data_dir}/popqa.json"))
    articles = json.load(open(f"{data_dir}/articles.json"))

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def canon(ids):
        return [int(compress[i]) for i in ids]

    def chat_ids(question, context=None):
        content = f"Answer the question in a few words.\n\n{question}"
        if context:
            content = f"Use this text to answer.\n\n{context}\n\n{content}"
        text = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = enc(text)
        return ids if ids[0] == bos else [bos] + ids

    # The model often reopens a <think> block even with enable_thinking=False. For a short, comparable readout we
    # forbid that token, which enforces the documented no-thinking mode.
    think_ids = enc("<think>")
    no_think = think_ids if len(think_ids) == 1 else None

    def generate(prompts, injector=None, vectors=None, masks=None, batch=48, max_new=40, think=False):
        """prompts: list of id lists. vectors: [N, D] tensor, masks: list of position lists (per prompt)."""
        extra = {} if think or no_think is None else {"suppress_tokens": no_think}
        outs = []
        for i in range(0, len(prompts), batch):
            group = prompts[i : i + batch]
            width = max(len(p) for p in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            mask = torch.zeros((len(group), width))
            for r, p in enumerate(group):
                off = width - len(p)
                ids[r, off:] = torch.tensor(p)
                att[r, off:] = 1
                if masks is not None:
                    for pos in masks[i + r]:
                        mask[r, off + pos] = 1.0
            if injector is not None:
                injector.set(vectors[i : i + len(group)], mask.to(dev))
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False, **extra)
            if injector is not None:
                injector.set()
            outs += tok.batch_decode(gen[:, width:], skip_special_tokens=True)
        return outs

    # ---- 1. closed-book baseline on a random sample of PopQA
    rng.shuffle(qa)
    items = []
    for r in qa:
        title = r["s_wiki_title"]
        if title not in articles or not r["subj"]:
            continue
        items.append({
            "q": r["question"], "answers": json.loads(r["possible_answers"]), "subj": r["subj"], "title": title,
            "pop": int(r["s_pop"]), "prop": r["prop"],
        })
        if len(items) >= n_baseline:
            break
    prompts = [chat_ids(it["q"]) for it in items]
    for it, out in zip(items, generate(prompts)):
        it["base_out"] = out.strip()
        it["status"] = "correct" if is_correct(out, it["answers"]) else ("abstain" if ABSTAIN.search(out) else "wrong")
    result = {"baseline": {"n": len(items), "think_token_suppressed": no_think is not None}}
    # reference: default behaviour (thinking allowed) on a subset, answer read after </think>
    sub = items[:n_think]
    if sub:
        outs = generate([chat_ids(it["q"]) for it in sub], max_new=384, think=True)
        finals = [o.split("</think>")[-1] if ("</think>" in o or "<think>" not in o) else None for o in outs]
        result["baseline_thinking_allowed"] = {
            "n": len(sub),
            "correct": round(sum(f is not None and is_correct(f, it["answers"]) for f, it in zip(finals, sub)) / len(sub), 4),
            "unfinished_after_384_tokens": round(sum(f is None for f in finals) / len(sub), 4),
            "same_questions_no_thinking_correct": round(sum(it["status"] == "correct" for it in sub) / len(sub), 4),
        }
    for name, lo, hi in (("all", 0, 10**12), ("pop<1k", 0, 1000), ("pop 1k-10k", 1000, 10_000), ("pop>10k", 10_000, 10**12)):
        sub = [it for it in items if lo <= it["pop"] < hi]
        if sub:
            result["baseline"][name] = {"n": len(sub), **{s: round(sum(it["status"] == s for it in sub) / len(sub), 4) for s in ("correct", "wrong", "abstain")}}
    result["baseline"]["examples"] = [{k: it[k] for k in ("q", "answers", "base_out", "status")} for it in items[:12]]

    # ---- 2. keep entities whose article states the answer and whose name is found in article and question
    chosen = {"fail": [], "ok": []}
    seen_titles = set()
    for it in items:
        kind = "ok" if it["status"] == "correct" else "fail"
        if len(chosen[kind]) >= (n_ok if kind == "ok" else n_fail) or it["title"] in seen_titles:
            continue
        art = enc(f"{it['title']}\n\n{articles[it['title']]}")[: FULL_LEN - 1]
        if len(art) < FULL_LEN - 1:
            continue
        if not is_correct(tok.decode(art[: TRAIN_LEN - 1]), it["answers"]):
            continue
        pats = [canon(enc(it["subj"])), canon(enc(" " + it["subj"]))]
        row = [bos] + art
        art_ends = find_ends(canon(row), pats)
        prompt = chat_ids(it["q"])
        q_ends = find_ends(canon(prompt), pats)
        if not q_ends or not any(e < TRAIN_LEN - 8 for e in art_ends):
            continue
        it.update(row=row, art_ends=art_ends, prompt=prompt, q_ends=q_ends, n_mentions=len(art_ends))
        chosen[kind].append(it)
        seen_titles.add(it["title"])
    ents = chosen["fail"] + chosen["ok"]
    n_f, n_e = len(chosen["fail"]), len(ents)
    result["selection"] = {
        "fail": n_f, "ok": len(chosen["ok"]),
        "mean_mentions_in_article": round(float(np.mean([it["n_mentions"] for it in ents])), 2),
        "mean_pop_fail": round(float(np.mean([it["pop"] for it in chosen["fail"]]))),
        "mean_pop_ok": round(float(np.mean([it["pop"] for it in chosen["ok"]]))) if chosen["ok"] else None,
    }

    # ---- 3. ceiling: the article excerpt in the prompt
    ctx_prompts = [chat_ids(it["q"], tok.decode(it["row"][1:TRAIN_LEN])) for it in chosen["fail"]]
    ctx_out = generate(ctx_prompts, batch=16)
    result["ceiling_article_in_prompt"] = round(sum(is_correct(o, it["answers"]) for o, it in zip(ctx_out, chosen["fail"])) / max(n_f, 1), 4)

    rows = torch.tensor([it["row"] for it in ents], device=dev)  # [E, FULL_LEN]
    art_mask = torch.zeros((n_e, FULL_LEN), device=dev)
    for i, it in enumerate(ents):
        art_mask[i, it["art_ends"]] = 1.0
    prompts = [it["prompt"] for it in ents]
    q_masks = [it["q_ends"] for it in ents]
    control = list(range(1, n_e)) + [0]  # every entity gets its neighbour's vector

    def article_loss(injector, vectors):
        """Mean loss on the trained part and on the held-out continuation of every article."""
        import torch.nn.functional as F

        tr = ho = 0.0
        with torch.inference_mode():
            for i in range(0, n_e, 16):
                ids = rows[i : i + 16]
                if injector is not None:
                    injector.set(vectors[i : i + 16], art_mask[i : i + 16])
                logits = model(input_ids=ids).logits.float()
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
                tr += loss[:, : TRAIN_LEN - 1].sum().item()
                ho += loss[:, TRAIN_LEN - 1 :].sum().item()
        if injector is not None:
            injector.set()
        return round(tr / (n_e * (TRAIN_LEN - 1)), 4), round(ho / (n_e * (FULL_LEN - TRAIN_LEN)), 4)

    result["article_loss_no_vector"] = dict(zip(("trained_part", "held_out_part"), article_loss(None, None)))
    result["layers"] = {}

    for layer in layers:
        t0 = time.time()
        # residual norm at this layer, to express the vector in units of the stream it is added to
        norms = []
        tap = model.model.layers[layer].register_forward_pre_hook(lambda m, a, k: norms.append(a[0].float().norm(dim=-1).mean().item()), with_kwargs=True)
        with torch.inference_mode():
            model.model(input_ids=rows[:16])
        tap.remove()
        scale = norms[0]

        v = torch.nn.Parameter(torch.zeros(n_e, model.config.hidden_size, device=dev))
        opt = torch.optim.Adam([v], lr=lr)
        inj = Injector(model, layer)
        trajectory = []
        for _ in range(epochs):
            total = 0.0
            for i in range(0, n_e, 16):  # one full-batch step per epoch: every entity's vector gets a gradient
                idx = torch.arange(i, min(i + 16, n_e), device=dev)
                ids = rows[idx, :TRAIN_LEN]
                inj.set(v[idx] * scale, art_mask[idx, :TRAIN_LEN])
                hidden = model.model(input_ids=ids).last_hidden_state
                total += chunked_ce_backward(model, hidden, ids, rows_per_chunk=8) * len(idx)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                v.data.mul_((cap / v.data.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0))
            trajectory.append(round(total / n_e, 4))
        inj.set()
        vec = (v.detach() * scale)
        own = generate(prompts, inj, vec, q_masks)
        ctl = generate(prompts, inj, vec[control], q_masks)
        ok_own = [is_correct(o, it["answers"]) for o, it in zip(own, ents)]
        ok_ctl = [is_correct(o, it["answers"]) for o, it in zip(ctl, ents)]
        loss_own, loss_ctl = article_loss(inj, vec), article_loss(inj, vec[control])
        inj.remove()
        n_ok_ents = n_e - n_f
        result["layers"][str(layer)] = {
            "residual_norm": round(scale, 3),
            "vector_norm_rel_mean": round(v.detach().norm(dim=-1).mean().item(), 3),
            "fix_rate_own_vector": round(sum(ok_own[:n_f]) / max(n_f, 1), 4),
            "fix_rate_control_vector": round(sum(ok_ctl[:n_f]) / max(n_f, 1), 4),
            "break_rate_own_vector": round(1 - sum(ok_own[n_f:]) / n_ok_ents, 4) if n_ok_ents else None,
            "break_rate_control_vector": round(1 - sum(ok_ctl[n_f:]) / n_ok_ents, 4) if n_ok_ents else None,
            "abstain_rate_own_on_fail": round(sum(bool(ABSTAIN.search(o)) for o in own[:n_f]) / max(n_f, 1), 4),
            "article_loss_own": dict(zip(("trained_part", "held_out_part"), loss_own)),
            "article_loss_control": dict(zip(("trained_part", "held_out_part"), loss_ctl)),
            "train_loss_first_last": [trajectory[0], trajectory[-1]],
            "seconds": round(time.time() - t0),
            "examples": [
                {"q": it["q"], "answers": it["answers"][:3], "before": it["base_out"], "own": own[i].strip(), "control": ctl[i].strip()}
                for i, it in enumerate(ents[:10])
            ],
        }
    result["seconds_total"] = round(time.time() - t_start)
    return result
