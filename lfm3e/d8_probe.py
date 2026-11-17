"""D8: can a small trained head on the frozen model's internal state find the right note by meaning?

Two probes, both with LFM3 frozen:

  WHAT  question -> the Wikipedia article of its answer, among a large pool of notes. The question does not name
        the answer, so exact-name keys cannot find it. Query = hidden state at the end of the chat prompt;
        key = mean hidden state over the note. Two linear maps trained with a contrastive loss on TriviaQA.
        Reported on held-out questions, separately for answers never seen in training.

  WHEN  does the model need help? A linear probe on the same state predicts whether the model's own closed-book
        answer is wrong."""
import json
import random
import sqlite3
import time

from .d7_fr import fold

Q_LAYERS = (8, 12, 16, 20, 24)
K_LAYERS = (12, 20, 24)


def run(data_dir: str, n_train=40000, n_test=3000, pool=200000, n_gate_train=5000, n_gate_test=2000, dim=256, epochs=15, seed=0):
    import shutil

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    t_start = time.time()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    shutil.copy(f"{data_dir}/notes2.db", "/tmp/notes2.db")
    db = sqlite3.connect("/tmp/notes2.db")
    tok = load_tokenizer()
    model = load_model()
    dev = "cuda"
    bos, pad = tok.bos_token_id, tok.pad_token_id

    def enc(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def article(title):
        return db.execute("SELECT title, description, lead FROM articles WHERE title=? LIMIT 1", (title,)).fetchone()

    # ---- data: (question, answer article), TriviaQA
    def pairs(split, limit):
        out = []
        for x in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split):
            ans = x["answer"]
            title = (ans.get("matched_wiki_entity_name") or (ans["value"] if ans.get("type") == "WikipediaEntity" else "")).strip()
            row = article(title) if title else None
            if row and len(x["question"]) < 300:
                out.append({"q": x["question"], "title": row[0], "golds": [a for a in [x["answer"]["value"]] + list(x["answer"]["aliases"]) if len(a) >= 3]})
            if len(out) >= limit:
                break
        return out

    both = pairs("train", n_train + n_test)  # held-out part of train
    if len(both) < 200:
        return {"error": "too few usable TriviaQA pairs", "pairs": len(both)}
    rng.shuffle(both)
    test, train = both[:n_test], both[n_test:]
    seen_titles = {p["title"] for p in train}

    # ---- pool of notes: every answer article plus random distractors
    titles = list(dict.fromkeys([p["title"] for p in train + test]))
    max_id = db.execute("SELECT MAX(id) FROM articles").fetchone()[0]
    have = set(titles)
    for i in rng.sample(range(max_id + 1), min(pool, max_id + 1)):
        if len(titles) >= pool:
            break
        row = db.execute("SELECT title FROM articles WHERE id=?", (i,)).fetchone()
        if row and row[0] not in have:
            have.add(row[0])
            titles.append(row[0])
    index = {t: i for i, t in enumerate(titles)}

    # ---- hidden states: last prompt token for questions, mean over the note for keys
    states = {}

    def tap(i):
        def hook(module, args, kwargs):
            states[i] = (args[0] if args else kwargs["hidden_states"]).detach()

        return hook

    hooks = [model.model.layers[i].register_forward_pre_hook(tap(i), with_kwargs=True) for i in set(Q_LAYERS + K_LAYERS) if i < 24]
    hooks.append(model.model.embedding_norm.register_forward_pre_hook(tap(24), with_kwargs=True))

    def chat_ids(q):
        ids = enc(tok.apply_chat_template([{"role": "user", "content": f"Answer the question in a few words.\n\n{q}"}], add_generation_prompt=True, enable_thinking=False, tokenize=False))
        return ids if ids[0] == bos else [bos] + ids

    def question_features(questions, batch=256):
        out = np.zeros((len(questions), len(Q_LAYERS), model.config.hidden_size), dtype=np.float16)
        for i in range(0, len(questions), batch):
            seqs = [chat_ids(q) for q in questions[i : i + batch]]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, width - len(s) :], att[r, width - len(s) :] = torch.tensor(s), 1
            with torch.inference_mode():
                model.model(input_ids=ids.to(dev), attention_mask=att.to(dev))
            for k, layer in enumerate(Q_LAYERS):
                out[i : i + len(seqs), k] = states[layer][:, -1].float().cpu().numpy()
        return out

    def note_features(titles_, batch=256):
        out = np.zeros((len(titles_), len(K_LAYERS), model.config.hidden_size), dtype=np.float16)
        for i in range(0, len(titles_), batch):
            seqs = []
            for t in titles_[i : i + batch]:
                title, desc, lead = article(t)
                seqs.append(([bos] + enc(f"{title} ({desc}). {lead}" if desc else f"{title}. {lead}"))[:80])
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)], att[r, : len(s)] = torch.tensor(s), 1
            ids, att = ids.to(dev), att.to(dev)
            with torch.inference_mode():
                model.model(input_ids=ids, attention_mask=att)
            m = att[:, :, None].float()
            for k, layer in enumerate(K_LAYERS):
                out[i : i + len(seqs), k] = ((states[layer].float() * m).sum(1) / m.sum(1)).cpu().numpy()
        return out

    q_train, q_test = question_features([p["q"] for p in train]), question_features([p["q"] for p in test])
    keys = note_features(titles)
    for h in hooks:
        h.remove()
    y_train = torch.tensor([index[p["title"]] for p in train], device=dev)
    y_test = torch.tensor([index[p["title"]] for p in test], device=dev)
    unseen = torch.tensor([p["title"] not in seen_titles for p in test], device=dev)

    def standardise(a, ref):
        mu, sd = ref.mean(0, keepdim=True), ref.std(0, keepdim=True) + 1e-6
        return (a - mu) / sd

    def recall(scores_fn, qs):
        hits = {1: [], 5: [], 20: [], 100: []}
        for i in range(0, qs.shape[0], 512):
            top = scores_fn(qs[i : i + 512]).topk(100, dim=-1).indices
            gold = y_test[i : i + 512, None]
            for k in hits:
                hits[k].append((top[:, :k] == gold).any(-1))
        hits = {k: torch.cat(v) for k, v in hits.items()}
        return {
            "all": {f"recall@{k}": round(v.float().mean().item(), 4) for k, v in hits.items()},
            "answer_never_seen_in_training": {f"recall@{k}": round(v[unseen].float().mean().item(), 4) for k, v in hits.items()},
        }

    result = {
        "train_pairs": len(train), "test_pairs": len(test), "pool": len(titles),
        "test_answers_never_seen_in_training": round(unseen.float().mean().item(), 4), "what": {},
    }
    best = None
    for qi, ql in enumerate(Q_LAYERS):
        for ki, kl in enumerate(K_LAYERS):
            qt = torch.tensor(q_train[:, qi].astype(np.float32), device=dev)
            qe = standardise(torch.tensor(q_test[:, qi].astype(np.float32), device=dev), qt)
            kk = torch.tensor(keys[:, ki].astype(np.float32), device=dev)
            kk = standardise(kk, kk)
            qt = standardise(qt, qt)
            wq, wk = torch.nn.Linear(qt.shape[1], dim, bias=False).to(dev), torch.nn.Linear(kk.shape[1], dim, bias=False).to(dev)
            opt = torch.optim.AdamW(list(wq.parameters()) + list(wk.parameters()), lr=1e-3, weight_decay=1e-2)
            for _ in range(epochs):
                perm = torch.randperm(qt.shape[0], device=dev)
                for i in range(0, len(perm), 2048):
                    idx = perm[i : i + 2048]
                    a = F.normalize(wq(qt[idx]), dim=-1)
                    b = F.normalize(wk(kk[y_train[idx]]), dim=-1)
                    logits = a @ b.T / 0.05
                    same = y_train[idx][:, None] == y_train[idx][None, :]  # two questions with the same answer are not negatives
                    logits = logits.masked_fill(same & ~torch.eye(len(idx), dtype=torch.bool, device=dev), -1e4)
                    loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                proj = F.normalize(wk(kk), dim=-1)
                r = recall(lambda q: F.normalize(wq(q), dim=-1) @ proj.T, qe)
            name = f"query_layer_{ql}_key_layer_{kl}"
            result["what"][name] = r
            score = r["answer_never_seen_in_training"]["recall@5"]
            if best is None or score > best[0]:
                best = (score, name)
            if ql == kl:  # no training at all: cosine between the raw states
                with torch.no_grad():
                    raw = F.normalize(kk, dim=-1)
                    result["what"][f"untrained_cosine_layer_{ql}"] = recall(lambda q: F.normalize(q, dim=-1) @ raw.T, qe)
    result["what_best"] = {"config": best[1], **result["what"][best[1]]}

    # ---- WHEN: does the model's own state say that its closed-book answer will be wrong?
    think = enc("<think>")
    no_think = {"suppress_tokens": think} if len(think) == 1 else {}

    def answers(items, batch=64):
        outs = []
        for i in range(0, len(items), batch):
            seqs = [chat_ids(p["q"]) for p in items[i : i + batch]]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, width - len(s) :], att[r, width - len(s) :] = torch.tensor(s), 1
            with torch.inference_mode():
                gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=24, do_sample=False, **no_think)
            outs += tok.batch_decode(gen[:, width:], skip_special_tokens=True)
        return outs

    def wrong(items):
        return torch.tensor([not any(fold(g) in fold(o) for g in p["golds"]) for p, o in zip(items, answers(items))], device=dev, dtype=torch.float32)

    g_tr, g_te = train[:n_gate_train], test[:n_gate_test]
    w_tr, w_te = wrong(g_tr), wrong(g_te)

    def auc(score, label):
        order = score.argsort()
        ranks = torch.empty_like(score)
        ranks[order] = torch.arange(1, len(score) + 1, device=dev, dtype=score.dtype)
        pos = label.sum()
        return ((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))).item()

    result["when"] = {"model_wrong_without_memory": {"train": round(w_tr.mean().item(), 4), "test": round(w_te.mean().item(), 4)}, "auc_by_layer": {}}
    for qi, ql in enumerate(Q_LAYERS):
        ref = torch.tensor(q_train[: len(g_tr), qi].astype(np.float32), device=dev)
        xt, xe = standardise(ref, ref), standardise(torch.tensor(q_test[: len(g_te), qi].astype(np.float32), device=dev), ref)
        probe = torch.nn.Linear(xt.shape[1], 1).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1.0)
        for _ in range(200):
            loss = F.binary_cross_entropy_with_logits(probe(xt)[:, 0], w_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = probe(xe)[:, 0]
            result["when"]["auc_by_layer"][str(ql)] = round(auc(s, w_te), 4)
            if ql == 16:
                # if we only called the memory for the half of the questions the probe is most worried about
                top = s >= s.median()
                result["when"]["layer_16_split"] = {"wrong_rate_in_flagged_half": round(w_te[top].mean().item(), 4), "wrong_rate_in_other_half": round(w_te[~top].mean().item(), 4)}
    result["examples"] = [{"q": p["q"], "answer_article": p["title"]} for p in test[:6]]
    result["seconds_total"] = round(time.time() - t_start)
    return result
