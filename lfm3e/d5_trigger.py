"""D5: make the attention memory safe when the passage is wrong, with no training.

Part A (robustness knobs, same question sets as D4, oracle vs random passage):
    - framing: raw passage, "Reference note" header, system instruction to ignore unrelated notes
    - attention malus: subtract `b` from the attention scores towards memory tokens at the 6 global layers

Part B (end to end, the system finds the entity by itself):
    - spans of the question that match Wikipedia titles (homonyms included) give candidate passages
    - the frozen model picks the candidate: PMI = log p(question | passage) - log p(question)
    - trigger only if PMI > 0, otherwise the model runs untouched"""
import json
import math
import os
import random
import re
import time
import unicodedata

from .d3_entity import ABSTAIN, is_correct
from .triggers import STOP, norm, pick_spans, question_spans  # noqa: F401 (re-exported)

GAP = 256
SYSTEM = (
    "Reference notes may appear before the conversation. Use a note only if it is about what the user asks. "
    "If a note is about something else, ignore it completely."
)


def title_bases(title):
    """Surface forms a title answers to: 'Skin (2018 film)' -> skin; 'Norfolk, Virginia' -> norfolk virginia, norfolk."""
    base = re.sub(r"\s*\([^)]*\)$", "", title)
    out = {norm(base)}
    if "," in base:
        out.add(norm(base.split(",")[0]))
    out.discard("")
    return out


def _scan(path, surfaces):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["title", "text"])
    titles = table["title"].to_pylist()
    rows = [k for k, t in enumerate(titles) if "(disambiguation)" not in t and not t.startswith("List of") and not title_bases(t).isdisjoint(surfaces)]
    texts = table["text"].take(rows).to_pylist()
    return [(titles[k], len(x), x[:1000]) for k, x in zip(rows, texts)]


def build_candidates(data_dir: str, per_surface: int = 8):
    """CPU: for every eligible span of every PopQA question, the Wikipedia articles whose title matches it."""
    import glob
    from concurrent.futures import ProcessPoolExecutor

    from huggingface_hub import snapshot_download

    qa = json.load(open(f"{data_dir}/popqa.json"))
    surfaces = {s[2] for r in qa for s in question_spans(r["question"])}
    root = snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=["20231101.en/*.parquet"])
    files = sorted(glob.glob(f"{root}/20231101.en/*.parquet"))
    cands = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for part in pool.map(_scan, files, [surfaces] * len(files)):
            for title, length, lead in part:
                for b in title_bases(title) & surfaces:
                    cands.setdefault(b, []).append({"title": title, "len": length, "lead": lead})
    for b in cands:
        cands[b] = sorted(cands[b], key=lambda c: -c["len"])[:per_surface]
    with open(f"{data_dir}/candidates.json", "w") as f:
        json.dump(cands, f)
    sizes = [len(v) for v in cands.values()]
    return {"surfaces_asked": len(surfaces), "surfaces_with_candidates": len(cands), "ambiguous_surfaces": sum(s > 1 for s in sizes), "mean_candidates": round(sum(sizes) / max(len(sizes), 1), 2)}


def run(data_dir: str, n_baseline=2500, n_fail=400, n_ok=150, n_e2e=1000, n_mem=64, n_mem_e2e=128, biases=(0.25, 0.5, 1.0, 2.0), seed=0):
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
    cands = json.load(open(f"{data_dir}/candidates.json"))

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def chat_ids(question, system=False):
        msgs = ([{"role": "system", "content": SYSTEM}] if system else []) + [{"role": "user", "content": f"Answer the question in a few words.\n\n{question}"}]
        ids = enc(tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    def passage(title, text, n, header=False):
        body = f"Reference note about {title}:\n{text}" if header else f"{title}\n\n{text}"
        return [bos] + enc(body)[:n]

    def with_memory(passages, prompt):
        ids, mask, mem = [], [], []
        for p in passages:
            ids += p + [pad] * GAP
            mask += [1] * len(p) + [0] * GAP
            mem += [1] * len(p) + [0] * GAP
        return ids + prompt, mask + [1] * len(prompt), mem + [0] * len(prompt)

    def plain(prompt):
        return prompt, [1] * len(prompt), [0] * len(prompt)

    # ---- attention malus on memory tokens, at the global-attention layers only
    state = {"cols": None, "b": 0.0, "applied": 0}

    def bias_hook(module, args, kwargs):
        m = kwargs.get("attention_mask")
        if state["cols"] is None or state["b"] == 0.0 or m is None:
            return None
        dtype = kwargs["hidden_states"].dtype
        fm = torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, torch.finfo(dtype).min) if m.dtype == torch.bool else m.clone()
        cols = state["cols"]
        if fm.shape[0] != cols.shape[0]:
            fm = fm.expand(cols.shape[0], *fm.shape[1:]).clone()
        width = min(cols.shape[1], fm.shape[-1])
        add = torch.zeros(fm.shape[0], fm.shape[-1], dtype=fm.dtype, device=fm.device)
        add[:, :width] = cols[:, :width].to(fm.dtype) * (-state["b"])
        kwargs["attention_mask"] = fm + add[:, None, None, :]
        state["applied"] += 1
        return args, kwargs

    for i, kind in enumerate(model.config.layer_types):
        if kind == "full_attention":
            model.model.layers[i].self_attn.register_forward_pre_hook(bias_hook, with_kwargs=True)

    think_ids = enc("<think>")
    no_think = {"suppress_tokens": think_ids} if len(think_ids) == 1 else {}

    def pad_batch(group):
        width = max(len(s[0]) for s in group)
        ids = torch.full((len(group), width), pad, dtype=torch.long)
        att = torch.zeros((len(group), width), dtype=torch.long)
        mem = torch.zeros((len(group), width))
        for r, (s, m, c) in enumerate(group):
            off = width - len(s)
            ids[r, off:], att[r, off:], mem[r, off:] = torch.tensor(s), torch.tensor(m), torch.tensor(c, dtype=torch.float)
        return ids.to(dev), att.to(dev), mem.to(dev), width

    def generate(seqs, b=0.0, batch=32, max_new=40):
        outs = []
        for i in range(0, len(seqs), batch):
            ids, att, mem, width = pad_batch(seqs[i : i + batch])
            state["cols"], state["b"] = mem, b
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, **no_think)
            state["cols"] = None
            outs += [o.strip() for o in tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
        return outs

    def prompt_logprob(seqs, prompt_lens, batch=48):
        """Per-token log p of the chat prompt (first token excluded), for each sequence."""
        weight = model.lm_head.weight.float()
        out = []
        for i in range(0, len(seqs), batch):
            ids, att, _, width = pad_batch(seqs[i : i + batch])
            with torch.inference_mode():
                hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
                for r, n in enumerate(prompt_lens[i : i + batch]):
                    logits = hidden[r, width - n : width - 1].float() @ weight.T
                    lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[r, width - n + 1 : width, None])
                    out.append(lp[:, 0].tolist())
        return out

    def chat_offsets(question):
        """Prompt ids, and for each id whether it is a question token outside / inside a given character range."""
        msgs = [{"role": "user", "content": f"Answer the question in a few words.\n\n{question}"}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(e["input_ids"]), [tuple(o) for o in e["offset_mapping"]]
        if ids[0] != bos:
            ids, offs = [bos] + ids, [(0, 0)] + offs
        start = text.find(question)
        return ids, offs, (start, start + len(question))

    def status(out, answers):
        return "correct" if is_correct(out, answers) else ("abstain" if ABSTAIN.search(out) else "wrong")

    def rates(outs, group):
        st = [status(o, it["answers"]) for o, it in zip(outs, group)]
        return {s: round(st.count(s) / max(len(st), 1), 4) for s in ("correct", "wrong", "abstain")}

    # ---- same question sets as D4
    rng.shuffle(qa)
    e2e_rows = qa[:n_e2e]
    items = []
    for r in qa:
        if r["s_wiki_title"] in articles and r["subj"]:
            items.append({"q": r["question"], "answers": json.loads(r["possible_answers"]), "title": r["s_wiki_title"]})
        if len(items) >= n_baseline:
            break
    for it, out in zip(items, generate([plain(chat_ids(it["q"])) for it in items], batch=48)):
        it["status"] = status(out, it["answers"])
    fail, ok, seen = [], [], set()
    for it in items:
        group, limit = (ok, n_ok) if it["status"] == "correct" else (fail, n_fail)
        if len(group) >= limit or it["title"] in seen:
            continue
        seen.add(it["title"])
        group.append(it)
    result = {"selection": {"fail": len(fail), "ok": len(ok)}}

    # ---- Part A: framing and attention malus, oracle passage vs another entity's passage
    def setting(header, system, b):
        def seqs(group, shift):
            src = group[shift:] + group[:shift]
            return [with_memory([passage(o["title"], articles[o["title"]][:1500], n_mem, header)], chat_ids(it["q"], system)) for it, o in zip(group, src)]

        res = {
            "fix_own": rates(generate(seqs(fail, 0), b), fail)["correct"],
            "fix_wrong_passage": rates(generate(seqs(fail, 1), b), fail)["correct"],
            "keep_own": rates(generate(seqs(ok, 0), b), ok)["correct"],
            "keep_wrong_passage": rates(generate(seqs(ok, 1), b), ok)["correct"],
        }
        if system:  # the system prompt alone changes behaviour: reference without any memory
            res["fix_no_memory"] = rates(generate([plain(chat_ids(it["q"], True)) for it in fail]), fail)["correct"]
            res["keep_no_memory"] = rates(generate([plain(chat_ids(it["q"], True)) for it in ok]), ok)["correct"]
        return res

    part_a = {"raw": setting(False, False, 0.0), "header": setting(True, False, 0.0), "raw_system": setting(False, True, 0.0), "header_system": setting(True, True, 0.0)}
    for b in biases:
        part_a[f"raw_malus_{b}"] = setting(False, False, b)
    # hook sanity on questions the model gets right: a huge malus must hide a wrong passage completely
    def wrong_seqs(group):
        src = group[1:] + group[:1]
        return [with_memory([passage(o["title"], articles[o["title"]][:1500], n_mem)], chat_ids(it["q"])) for it, o in zip(group, src)]

    part_a["hook"] = {
        "calls_with_malus_applied": state["applied"],
        "keep_wrong_passage_tiny_malus": rates(generate(wrong_seqs(ok), 1e-4), ok)["correct"],
        "keep_wrong_passage_huge_malus": rates(generate(wrong_seqs(ok), 60.0), ok)["correct"],
    }
    best_b = max(biases, key=lambda b: sum(part_a[f"raw_malus_{b}"][k] for k in ("fix_own", "keep_own", "keep_wrong_passage")))
    part_a["best_malus"] = best_b
    result["part_a"] = part_a

    # ---- Part B: end to end on random questions, the system finds the entity
    rows = []
    for r in e2e_rows:
        q = r["question"]
        ids, offs, (q0, q1) = chat_offsets(q)
        words = [m.span() for m in re.finditer(r"\S+", q)]
        row = {"q": q, "answers": json.loads(r["possible_answers"]), "gold": r["s_wiki_title"], "pop": int(r["s_pop"]), "ids": ids, "spans": []}
        for i, j, surface in pick_spans(q, cands):
            m0, m1 = q0 + words[i][0], q0 + words[j - 1][1]
            # question tokens that are not part of the name: the only ones that can tell homonyms apart
            ctx = [a_ >= q0 and b_ <= q1 and not (a_ < m1 and b_ > m0) and b_ > a_ for a_, b_ in offs]
            row["spans"].append({"surface": surface, "ctx": ctx[1:], "cands": [dict(c) for c in cands[surface][:8]]})
        rows.append(row)
    prompts = [row["ids"] for row in rows]
    base_lp = prompt_logprob([plain(p) for p in prompts], [len(p) for p in prompts])
    pairs = [(k, s_, c) for k, row in enumerate(rows) for s_, span in enumerate(row["spans"]) for c in range(len(span["cands"]))]
    seqs = [with_memory([passage(rows[k]["spans"][s_]["cands"][c]["title"], rows[k]["spans"][s_]["cands"][c]["lead"], n_mem_e2e)], prompts[k]) for k, s_, c in pairs]
    for (k, s_, c), lp in zip(pairs, prompt_logprob(seqs, [len(prompts[k]) for k, _, _ in pairs])):
        span = rows[k]["spans"][s_]
        cand = span["cands"][c]
        diff = [x - y for x, y in zip(lp, base_lp[k])]
        cand["pmi_all"] = sum(diff)
        cand["pmi"] = sum(d for d, keep in zip(diff, span["ctx"]) if keep)
        cand["prior"] = math.log(max(cand["len"], 1))

    keys = {
        "longest_article": lambda c: c["prior"],
        "pmi_whole_prompt": lambda c: c["pmi_all"],
        "pmi_outside_name": lambda c: c["pmi"],
        "pmi_outside_name_plus_0.5_prior": lambda c: c["pmi"] + 0.5 * c["prior"],
        "pmi_outside_name_plus_1_prior": lambda c: c["pmi"] + 1.0 * c["prior"],
        "pmi_outside_name_plus_2_prior": lambda c: c["pmi"] + 2.0 * c["prior"],
    }
    linkable = [(row, span) for row in rows for span in row["spans"] if any(c["title"] == row["gold"] for c in span["cands"])]
    ambiguous = [(row, span) for row, span in linkable if len(span["cands"]) > 1]

    def link_acc(pairs_, key):
        return round(sum(max(span["cands"], key=key)["title"] == row["gold"] for row, span in pairs_) / max(len(pairs_), 1), 4)

    result["linking"] = {
        "questions": len(rows),
        "gold_article_exists": round(sum(row["gold"] in articles for row in rows) / len(rows), 4),
        "span_found": round(sum(bool(row["spans"]) for row in rows) / len(rows), 4),
        "gold_among_candidates": round(len({id(row) for row, _ in linkable}) / len(rows), 4),
        "ambiguous_share_of_linkable": round(len(ambiguous) / max(len(linkable), 1), 4),
        "mean_candidates_when_ambiguous": round(sum(len(span["cands"]) for _, span in ambiguous) / max(len(ambiguous), 1), 2),
        "accuracy_all_linkable": {name: link_acc(linkable, key) for name, key in keys.items()},
        "accuracy_ambiguous_only": {name: link_acc(ambiguous, key) for name, key in keys.items()},
    }

    def auto(row, mode):
        chosen = []
        for span in row["spans"]:
            cs = span["cands"]
            if mode == "longest":
                chosen.append(max(cs, key=keys["longest_article"]))
            elif mode == "pmi":  # best candidate, only if it makes the rest of the question more likely
                best = max(cs, key=keys["pmi_outside_name"])
                if best["pmi"] > 0:
                    chosen.append(best)
            elif mode == "combo":
                chosen.append(max(cs, key=keys["pmi_outside_name_plus_1_prior"]))
            elif mode == "top3":
                chosen += sorted(cs, key=keys["pmi_outside_name_plus_1_prior"], reverse=True)[:3][::-1]
            elif mode == "all":
                chosen += sorted(cs, key=keys["pmi_outside_name_plus_1_prior"], reverse=True)[:6][::-1]
        return chosen

    def e2e(mode, header=False, system=False, b=0.0):
        picks = [auto(row, mode) for row in rows]
        seqs_ = [
            with_memory([passage(c["title"], c["lead"], n_mem_e2e, header) for c in pick], chat_ids(row["q"], system)) if pick else plain(chat_ids(row["q"], system))
            for row, pick in zip(rows, picks)
        ]
        outs = generate(seqs_, b, batch=16 if mode in ("top3", "all") else 32)
        res = rates(outs, rows)
        res["triggered"] = round(sum(bool(p_) for p_ in picks) / len(rows), 4)
        res["gold_passage_shown"] = round(sum(any(c["title"] == row["gold"] for c in p_) for row, p_ in zip(rows, picks)) / len(rows), 4)
        res["wrong_passage_only"] = round(sum(bool(p_) and not any(c["title"] == row["gold"] for c in p_) for row, p_ in zip(rows, picks)) / len(rows), 4)
        return res, outs

    base_out = generate([plain(p_) for p_ in prompts])
    oracle = generate([
        with_memory([passage(row["gold"], articles[row["gold"]][:1500], n_mem_e2e)], p_) if row["gold"] in articles else plain(p_)
        for row, p_ in zip(rows, prompts)
    ])
    runs = {
        "auto_top1_pmi": e2e("pmi"),
        "auto_top1_longest_article": e2e("longest"),
        "auto_top1_pmi_plus_prior": e2e("combo"),
        "auto_top3": e2e("top3"),
        "auto_all_candidates": e2e("all"),
        "auto_top3_header_system": e2e("top3", header=True, system=True),
        "auto_top1_pmi_plus_prior_header_system": e2e("combo", header=True, system=True),
    }
    was_ok = [k for k, o in enumerate(base_out) if is_correct(o, rows[k]["answers"])]
    result["end_to_end"] = {"no_memory": rates(base_out, rows), "oracle_entity": rates(oracle, rows)}
    for name, (res, outs) in runs.items():
        res["keeps_answers_that_were_correct"] = round(sum(is_correct(outs[k], rows[k]["answers"]) for k in was_ok) / max(len(was_ok), 1), 4)
        result["end_to_end"][name] = res
    best_name = max(runs, key=lambda n: runs[n][0]["correct"])

    # PopQA names are often homonyms that the question alone cannot resolve: split by how ambiguous the name is
    def subset(idx):
        acc = lambda outs: round(sum(is_correct(outs[k], rows[k]["answers"]) for k in idx) / max(len(idx), 1), 4)  # noqa: E731
        return {"n": len(idx), "no_memory": acc(base_out), "oracle_entity": acc(oracle), **{name: acc(outs) for name, (_, outs) in runs.items()}}

    n_cands = [sum(len(span["cands"]) for span in row["spans"]) for row in rows]
    result["end_to_end_by_ambiguity"] = {
        "no_candidate": subset([k for k, n in enumerate(n_cands) if n == 0]),
        "one_candidate": subset([k for k, n in enumerate(n_cands) if n == 1]),
        "2_to_4_candidates": subset([k for k, n in enumerate(n_cands) if 2 <= n <= 4]),
        "5_or_more_candidates": subset([k for k, n in enumerate(n_cands) if n >= 5]),
    }
    result["examples"] = [
        {
            "q": row["q"], "answers": row["answers"][:2], "gold": row["gold"],
            "candidates": [(c["title"], round(c["pmi"], 2), round(c["pmi_all"], 1)) for span in row["spans"] for c in span["cands"]][:6],
            "no_memory": base_out[k][:70], best_name: runs[best_name][1][k][:70],
        }
        for k, row in enumerate(rows[:14])
    ]
    result["seconds_total"] = round(time.time() - t_start)
    return result
