"""D11: a richer note and a trained head that picks the right article among homonyms.

  note   = summary (96 tokens) + infobox facts (128 tokens)            [D10: +5 to +9 points of coverage]
  head   = two linear maps on the frozen model's states: the question's state against each candidate note's
           state, plus the article size as a prior, plus a 'none of them' option. Trained on PopQA questions that
           are not in the evaluation sample; the model itself is untouched.

Candidates for a name are all the articles whose title is that name, with or without a qualifier in brackets
('Skin', 'Skin (1995 film)', 'Skin (musician)', ...). No thinking, same layout as D4/D5/D9."""
import json
import math
import random
import sqlite3
import time

from .d5_trigger import norm, pick_spans
from .d6_reason import find_triggers
from .d7_fr import MKQA_URL, fold

GAP = 256
LEAD_TOKENS, INFO_TOKENS = 96, 128
K = 12


def run(data_dir: str, popqa_dir: str, web_tokens_en: str, n_eval=1000, n_train=13000, epochs=40, seed=0):
    import gzip
    import shutil
    import urllib.request

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    t_start = time.time()
    torch.manual_seed(seed)
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id
    shutil.copy(f"{data_dir}/notes3.db", "/tmp/notes3.db")
    db = sqlite3.connect("/tmp/notes3.db")

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    # ---- dictionary
    cand_cache, exact_cache = {}, {}

    def candidates(surface):
        """Articles answering to this name, most developed first: (title, description, lead, infobox, size)."""
        if surface not in cand_cache:
            rows, seen = [], set()
            for r in db.execute("SELECT a.title, a.description, a.lead, a.infobox, a.size FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? ORDER BY a.size DESC LIMIT 60", (surface,)):
                if r[0] not in seen:
                    seen.add(r[0])
                    rows.append(r)
            cand_cache[surface] = rows[:K]
        return cand_cache[surface]

    def exact(surface):
        if surface not in exact_cache:
            rows = db.execute("SELECT a.title, a.description, a.lead, a.infobox, a.size FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,)).fetchall()
            exact_cache[surface] = min(rows, key=lambda r: (r[0].isupper(), -sum(c.islower() for c in r[0]), len(r[0]))) if rows else None
        return exact_cache[surface]

    def by_title(title):
        return db.execute("SELECT title, description, lead, infobox, size FROM articles WHERE title=? LIMIT 1", (title,)).fetchone()

    class HasCandidates:
        def __contains__(self, surface):
            return bool(candidates(surface))

    has_candidates = HasCandidates()
    web = np.load(web_tokens_en)
    counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
    del web
    wc = {}

    def word_tokens(word):
        if word not in wc:
            wc[word] = enc(" " + word)
        return wc[word]

    freq = lambda w: float(counts[word_tokens(w.lower())[0]]) if len(word_tokens(w.lower())) == 1 else 0.0  # noqa: E731
    ntok = lambda w: len(word_tokens(w))  # noqa: E731

    def note(row, rich):
        title, desc, lead, infobox = row[:4]
        ids = [bos] + enc(f"Reference note about {title}{f' ({desc})' if desc else ''}. Use it only if it is relevant, otherwise ignore it.\n") + enc(lead)[:LEAD_TOKENS]
        if rich and infobox:
            ids += enc("\nFacts:\n" + infobox)[: INFO_TOKENS + 3]
        return ids

    def chat_ids(q):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": f"Answer the question in a few words.\n\n{q}"}], add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    think = enc("<think>")
    no_think = {"suppress_tokens": think} if len(think) == 1 else {}
    captured = {}
    model.model.embedding_norm.register_forward_pre_hook(lambda m, a, k: captured.__setitem__("h", (a[0] if a else k["hidden_states"]).detach()), with_kwargs=True)

    def pad_batch(seqs, left=True):
        width = max(len(x[0]) for x in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, (x, m) in enumerate(seqs):
            sl = slice(width - len(x), width) if left else slice(0, len(x))
            ids[r, sl], att[r, sl] = torch.tensor(x), torch.tensor(m)
        return ids.to(dev), att.to(dev), width

    def generate(seqs, batch=32, max_new=40):
        outs = []
        for i in range(0, len(seqs), batch):
            ids, att, width = pad_batch(seqs[i : i + batch])
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, **no_think)
            outs += [o.strip() for o in tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
        return outs

    def question_states(questions, batch=256):
        out = []
        for i in range(0, len(questions), batch):
            ids, att, _ = pad_batch([(p, [1] * len(p)) for p in (chat_ids(q) for q in questions[i : i + batch])])
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=att)
            out.append(captured["h"][:, -1].float())
        return torch.cat(out)

    key_cache = {}

    def key_states(rows, batch=256):
        """Mean state over a short version of the note (title, description, start of the summary), cached by title."""
        todo = [r for r in {r[0]: r for r in rows}.values() if r[0] not in key_cache]
        for i in range(0, len(todo), batch):
            group = todo[i : i + batch]
            seqs = [([bos] + enc(f"{r[0]} ({r[1]}). {r[2]}" if r[1] else f"{r[0]}. {r[2]}"))[:72] for r in group]
            ids, att, _ = pad_batch([(s, [1] * len(s)) for s in seqs], left=False)
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=att)
            m = att[:, :, None].float()
            mean = ((captured["h"].float() * m).sum(1) / m.sum(1)).half().cpu()
            for r, v in zip(group, mean):
                key_cache[r[0]] = v

    def layout(blocks, prompt):
        ids, mask = [], []
        for b in blocks:
            ids += b + [pad] * GAP
            mask += [1] * len(b) + [0] * GAP
        return ids + prompt, mask + [1] * len(prompt)

    def correct(out, golds):
        low = fold(out)
        return any(fold(g) in low for g in golds if len(g) >= 3)

    # ---- data: PopQA split as in D5/D9 (first 1000 = evaluation, the rest = training of the head)
    popqa = json.load(open(f"{popqa_dir}/popqa.json"))
    random.Random(0).shuffle(popqa)
    pop_eval = [{"q": r["question"], "golds": json.loads(r["possible_answers"]), "gold": r["s_wiki_title"]} for r in popqa[:n_eval]]
    # the head must never have seen an evaluation entity, nor an evaluation name (homonyms share it), in training
    eval_titles = {r["s_wiki_title"] for r in popqa[:n_eval]}
    eval_names = {norm(r["subj"] or "") for r in popqa[:n_eval]} - {""}
    rest = popqa[n_eval:]
    kept = [r for r in rest if r["s_wiki_title"] not in eval_titles and norm(r["subj"] or "") not in eval_names]
    split_info = {"candidates_for_training": len(rest), "removed_same_entity_or_same_name_as_evaluation": len(rest) - len(kept)}
    pop_train = [{"q": r["question"], "gold": r["s_wiki_title"]} for r in kept[:n_train]]

    def spans_of(q):
        return [s for _, _, s in pick_spans(q, has_candidates)]

    def examples(items):
        """One example per name span: its candidates and the index of the right one (K = none of them)."""
        out = []
        for qi, it in enumerate(items):
            for surface in spans_of(it["q"]):
                cs = candidates(surface)
                titles = [c[0] for c in cs]
                out.append({"qi": qi, "cands": cs, "y": titles.index(it["gold"]) if it["gold"] in titles else K})
        return out

    ex_train, ex_eval = examples(pop_train), examples(pop_eval)
    key_states([c for e in ex_train + ex_eval for c in e["cands"]])
    q_train, q_eval = question_states([it["q"] for it in pop_train]), question_states([it["q"] for it in pop_eval])
    q_mu, q_sd = q_train.mean(0, keepdim=True), q_train.std(0, keepdim=True) + 1e-6
    all_keys = torch.stack(list(key_cache.values())).float().to(dev)
    k_mu, k_sd = all_keys.mean(0, keepdim=True), all_keys.std(0, keepdim=True) + 1e-6
    del all_keys

    def tensors(exs, q_states):
        q = (q_states[[e["qi"] for e in exs]] - q_mu) / q_sd
        keys = torch.zeros(len(exs), K, q.shape[1], device=dev)
        prior = torch.zeros(len(exs), K, device=dev)
        mask = torch.zeros(len(exs), K, dtype=torch.bool, device=dev)
        for i, e in enumerate(exs):
            for j, c in enumerate(e["cands"]):
                keys[i, j] = (key_cache[c[0]].float().to(dev) - k_mu[0]) / k_sd[0]
                prior[i, j] = math.log(max(c[4], 1)) - 10.0
                mask[i, j] = True
        return q, keys, prior, mask, torch.tensor([e["y"] for e in exs], device=dev)

    class Head(torch.nn.Module):
        def __init__(self, d, dim=256, use_prior=True):
            super().__init__()
            self.use_prior = use_prior
            self.q, self.k = torch.nn.Linear(d, dim, bias=False), torch.nn.Linear(d, dim, bias=False)
            self.w_prior, self.null, self.log_t = torch.nn.Parameter(torch.zeros(())), torch.nn.Parameter(torch.zeros(())), torch.nn.Parameter(torch.tensor(2.5))

        def forward(self, q, keys, prior, mask):
            s = torch.einsum("bd,bkd->bk", F.normalize(self.q(q), dim=-1), F.normalize(self.k(keys), dim=-1)) * self.log_t.exp()
            if self.use_prior:
                s = s + self.w_prior * prior
            s = s.masked_fill(~mask, -1e4)
            return torch.cat([s, self.null.expand(len(q), 1)], dim=1)  # last column: none of the candidates

    tq_, tk, tp, tm, ty = tensors(ex_train, q_train)

    def fit(use_prior):
        h = Head(tq_.shape[1], use_prior=use_prior).to(dev)
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            perm = torch.randperm(len(ty), device=dev)
            for i in range(0, len(perm), 512):
                idx = perm[i : i + 512]
                loss = F.cross_entropy(h(tq_[idx], tk[idx], tp[idx], tm[idx]), ty[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return h

    head, head_no_prior = fit(True), fit(False)
    del tq_, tk, tp, tm

    eq, ek, ep, em, ey = tensors(ex_eval, q_eval)
    with torch.no_grad():
        sc = head(eq, ek, ep, em)
        sc_np = head_no_prior(eq, ek, ep, em)
        raw = torch.einsum("bd,bkd->bk", F.normalize(eq, dim=-1), F.normalize(ek, dim=-1)).masked_fill(~em, -1e4)
    has_gold = ey < K
    amb = has_gold & (em.sum(1) > 1)
    result = {
        "head": {
            "split": split_info, "train_questions": len(pop_train), "train_examples": len(ex_train), "train_gold_among_candidates": round(sum(e["y"] < K for e in ex_train) / max(len(ex_train), 1), 4),
            "eval_examples": len(ex_eval), "eval_gold_among_candidates": round(has_gold.float().mean().item(), 4),
            "eval_name_ambiguous": round(amb.float().sum().item() / max(has_gold.sum().item(), 1), 4),
            "mean_candidates_when_ambiguous": round(em[amb].sum(1).float().mean().item(), 2) if amb.any() else None,
            "pick_right_article_all": {
                "largest_article": round((ep.masked_fill(~em, -1e4).argmax(1) == ey)[has_gold].float().mean().item(), 4),
                "untrained_cosine": round((raw.argmax(1) == ey)[has_gold].float().mean().item(), 4),
                "trained_head": round((sc[:, :K].argmax(1) == ey)[has_gold].float().mean().item(), 4),
                "trained_head_without_size_prior": round((sc_np[:, :K].argmax(1) == ey)[has_gold].float().mean().item(), 4),
            },
            "pick_right_article_ambiguous_names": {
                "largest_article": round((ep.masked_fill(~em, -1e4).argmax(1) == ey)[amb].float().mean().item(), 4),
                "untrained_cosine": round((raw.argmax(1) == ey)[amb].float().mean().item(), 4),
                "trained_head": round((sc[:, :K].argmax(1) == ey)[amb].float().mean().item(), 4),
                "trained_head_without_size_prior": round((sc_np[:, :K].argmax(1) == ey)[amb].float().mean().item(), 4),
                "trained_head_in_top_2": round((sc[:, :K].topk(2, dim=1).indices == ey[:, None]).any(1)[amb].float().mean().item(), 4),
                "trained_head_in_top_3": round((sc[:, :K].topk(3, dim=1).indices == ey[:, None]).any(1)[amb].float().mean().item(), 4),
                "largest_article_in_top_3": round((ep.masked_fill(~em, -1e4).topk(3, dim=1).indices == ey[:, None]).any(1)[amb].float().mean().item(), 4),
            },
            "says_none_when_gold_absent": round((sc.argmax(1) == K)[~has_gold].float().mean().item(), 4) if (~has_gold).any() else None,
            "says_none_when_gold_present": round((sc.argmax(1) == K)[has_gold].float().mean().item(), 4),
            "prior_weight": round(head.w_prior.item(), 3),
        }
    }

    # ---- end to end
    def choose(q, state, top):
        """Notes picked by the head for the name spans of a question."""
        rows = []
        for surface in spans_of(q):
            cs = candidates(surface)
            if not cs:
                continue
            key_states(cs)
            keys = torch.zeros(1, K, state.shape[0], device=dev)
            prior = torch.zeros(1, K, device=dev)
            mask = torch.zeros(1, K, dtype=torch.bool, device=dev)
            for j, c in enumerate(cs):
                keys[0, j], prior[0, j], mask[0, j] = (key_cache[c[0]].float().to(dev) - k_mu[0]) / k_sd[0], math.log(max(c[4], 1)) - 10.0, True
            with torch.no_grad():
                s = head(((state - q_mu[0]) / q_sd[0])[None], keys, prior, mask)[0]
            for j in s[:K].topk(min(top, len(cs))).indices.tolist():
                if s[j] > s[K] and cs[j][0] not in [r[0] for r in rows]:
                    rows.append(cs[j])
        return rows

    def strict(q):
        rows = []
        for surface, _ in find_triggers(q, lambda s: exact(s) is not None, freq, ntok):
            r = exact(surface)
            if r and r[0] not in [x[0] for x in rows]:
                rows.append(r)
        return rows[:3]

    def evaluate(items, with_gold):
        prompts = [chat_ids(it["q"]) for it in items]
        states = question_states([it["q"] for it in items])
        base = generate([(p, [1] * len(p)) for p in prompts], batch=48)
        ok_b = [correct(o, it["golds"]) for o, it in zip(base, items)]
        strict_rows = [strict(it["q"]) for it in items]
        plans = {
            "exact_title_summary_only": (strict_rows, False),
            "exact_title_summary_and_facts": (strict_rows, True),
        }
        # no training at all: the most developed articles carrying the name (closest to D5, on the complete source)
        for top in (1, 3):
            plans[f"largest_top{top}_summary_and_facts"] = ([[c for surface in spans_of(it["q"]) for c in candidates(surface)[:top]][:3] or sr for it, sr in zip(items, strict_rows)], True)
        for top in (1, 2, 3):
            picked = []
            for it, st, sr in zip(items, states, strict_rows):
                rows = choose(it["q"], st, top)
                for r in sr:  # concept notes and names the head did not cover
                    if r[0] not in [x[0] for x in rows] and not any(fold(r[0]).split(" (")[0] == fold(x[0]).split(" (")[0] for x in rows):
                        rows.append(r)
                picked.append(rows[:3])
            plans[f"head_top{top}_summary_and_facts"] = (picked, True)
        if with_gold:
            gold_rows = [[r] if (r := by_title(it["gold"])) else [] for it in items]
            plans["right_article_summary_only"] = (gold_rows, False)
            plans["right_article_summary_and_facts"] = (gold_rows, True)
        res = {"n": len(items), "no_memory": round(sum(ok_b) / len(items), 4)}
        for name, (rows_per_q, rich) in plans.items():
            outs = generate([layout([note(r, rich) for r in rows], p) if rows else (p, [1] * len(p)) for rows, p in zip(rows_per_q, prompts)])
            ok = [correct(o, it["golds"]) for o, it in zip(outs, items)]
            res[name] = {
                "accuracy": round(sum(ok) / len(items), 4), "fixed": sum(m and not b for b, m in zip(ok_b, ok)), "broken": sum(b and not m for b, m in zip(ok_b, ok)),
                "note_shown": round(sum(bool(r) for r in rows_per_q) / len(items), 4),
            }
            if with_gold:
                res[name]["right_article_shown"] = round(sum(any(r[0] == it["gold"] for r in rows) for rows, it in zip(rows_per_q, items)) / len(items), 4)
        return res

    result["popqa"] = evaluate(pop_eval, True)
    tq = []
    for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"):
        if len(x["question"]) < 300:
            tq.append({"q": x["question"], "golds": [x["answer"]["value"]] + list(x["answer"]["aliases"])})
        if len(tq) >= 5000 + n_eval:
            break
    random.Random(0).shuffle(tq)
    result["triviaqa"] = evaluate(tq[5000:], False)
    with urllib.request.urlopen(urllib.request.Request(MKQA_URL, headers={"User-Agent": "lfm3-memory-research/0.1"}), timeout=120) as r:
        raw_lines = gzip.decompress(r.read()).decode("utf-8").splitlines()
    mk = []
    for line in raw_lines:
        x = json.loads(line)
        ge, gf = [], []
        for lang, g in (("en", ge), ("fr", gf)):
            for a in x["answers"].get(lang, []):
                if a.get("type") == "entity" and a.get("text"):
                    g += [a["text"]] + [al for al in a.get("aliases", []) if al]
        if ge and gf and x["queries"].get("fr"):
            mk.append({"q": x["query"], "golds": list(dict.fromkeys(ge))})
    random.Random(0).shuffle(mk)
    result["mkqa_en"] = evaluate(mk[:n_eval], False)
    result["seconds_total"] = round(time.time() - t_start)
    return result
