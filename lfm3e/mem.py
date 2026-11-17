"""The memory as one library: dictionary, notes, layout, generation, article-choice head, LoRA.

Collects what the experiments D4 to D12 established, with the fixes found on review:
  - the right article is identified by its Wikidata id when the benchmark gives one (titles get renamed)
  - names with a comma qualifier are homonyms too ('Norfolk, Connecticut' for 'Norfolk'): notes3.db rebuilt
  - the repetition penalty does not count the tokens of the notes (D12: +4.3 points)
  - answers are matched on whole words (lfm3e.scoring)"""
import math
import sqlite3

from .d5_trigger import pick_spans
from .d6_reason import find_triggers

GAP = 256
LEAD_TOKENS, INFO_TOKENS = 96, 128
K = 12
GLOBAL_LAYERS = (2, 6, 10, 14, 18, 22)
ROW = "a.title, a.description, a.lead, a.infobox, a.size, a.qid"


class Memory:
    """Dictionary access and note construction. A row is (title, description, lead, infobox, size, qid)."""

    def __init__(self, tok, db_path, web_tokens_path):
        import numpy as np

        self.tok = tok
        self.bos, self.pad, self.eos = tok.bos_token_id, tok.pad_token_id, tok.eos_token_id
        self.db = sqlite3.connect(db_path)
        web = np.load(web_tokens_path)
        self.counts = np.bincount(web, minlength=len(tok)).astype(np.float64) * (1e6 / len(web))
        self._cand, self._exact, self._words = {}, {}, {}

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ---- dictionary
    def candidates(self, surface):
        """Every article answering to this name, most developed first."""
        if surface not in self._cand:
            rows, seen = [], set()
            for r in self.db.execute(f"SELECT {ROW} FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? ORDER BY a.size DESC LIMIT 80", (surface,)):
                if r[0] not in seen:
                    seen.add(r[0])
                    rows.append(r)
            self._cand[surface] = rows[:K]
        return self._cand[surface]

    def __contains__(self, surface):  # lets pick_spans ask "does this name have candidates?"
        return bool(self.candidates(surface))

    def exact(self, surface):
        if surface not in self._exact:
            rows = self.db.execute(f"SELECT {ROW} FROM titles t JOIN articles a ON a.id = t.id WHERE t.surface=? AND t.exact=1 LIMIT 8", (surface,)).fetchall()
            self._exact[surface] = min(rows, key=lambda r: (r[0].isupper(), -sum(c.islower() for c in r[0]), len(r[0]))) if rows else None
        return self._exact[surface]

    def by_qid(self, qid):
        return self.db.execute(f"SELECT {ROW} FROM articles a WHERE a.qid=? ORDER BY a.size DESC LIMIT 1", (qid,)).fetchone() if qid else None

    def by_title(self, title):
        return self.db.execute(f"SELECT {ROW} FROM articles a WHERE a.title=? LIMIT 1", (title,)).fetchone()

    # ---- triggers
    def _word_tokens(self, word):
        if word not in self._words:
            self._words[word] = self.enc(" " + word)
        return self._words[word]

    def _freq(self, word):
        ids = self._word_tokens(word.lower())
        return float(self.counts[ids[0]]) if len(ids) == 1 else 0.0

    def spans(self, question):
        """Capitalised names of the question that have candidates (homonyms included)."""
        return [s for _, _, s in pick_spans(question, self)]

    def strict(self, text, limit=3):
        """Precision-first notes: exact titles only, rare names and technical terms (rules of D6)."""
        rows = []
        for surface, _ in find_triggers(text, lambda s: self.exact(s) is not None, self._freq, lambda w: len(self._word_tokens(w))):
            r = self.exact(surface)
            if r and r[0] not in [x[0] for x in rows]:
                rows.append(r)
        return rows[:limit]

    # ---- notes and prompts
    def note(self, row, rich=True):
        title, desc, lead, infobox = row[:4]
        ids = [self.bos] + self.enc(f"Reference note about {title}{f' ({desc})' if desc else ''}. Use it only if it is relevant, otherwise ignore it.\n") + self.enc(lead)[:LEAD_TOKENS]
        if rich and infobox:
            ids += self.enc("\nFacts:\n" + infobox)[: INFO_TOKENS + 3]
        return ids

    def note_text(self, row, rich=True):
        return self.tok.decode(self.note(row, rich)[1:])

    def chat(self, question, instruction="Answer the question in a few words.\n\n"):
        text = self.tok.apply_chat_template([{"role": "user", "content": f"{instruction}{question}"}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = self.enc(text)
        return ids if ids[0] == self.bos else [self.bos] + ids

    def layout(self, blocks, tail):
        """[note][masked gap]...[tail]. Returns ids, attention mask, and the mask of tokens the repetition penalty
        may count (everything except notes and padding)."""
        ids, att, counted = [], [], []
        for b in blocks:
            ids += b + [self.pad] * GAP
            att += [1] * len(b) + [0] * GAP
            counted += [0] * (len(b) + GAP)
        return ids + tail, att + [1] * len(tail), counted + [1] * len(tail)


def pad_left(seqs, pad, device):
    import torch

    width = max(len(s[0]) for s in seqs)
    out = [torch.full((len(seqs), width), pad, dtype=torch.long)] + [torch.zeros((len(seqs), width), dtype=torch.long) for _ in range(len(seqs[0]) - 1)]
    for r, s in enumerate(seqs):
        for t, part in zip(out, s):
            t[r, width - len(part) :] = torch.tensor(part)
    return [t.to(device) for t in out], width


def generate(model, mem, seqs, max_new=40, batch=32, penalty=1.1, think=False):
    """Greedy generation for (ids, attention mask, counted mask) sequences. The repetition penalty of the model's
    own default (1.1) is applied to the prompt and to what is generated, not to the notes."""
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class PenaltyOutsideMemory(LogitsProcessor):
        def __init__(self, counted):
            self.counted = counted

        def __call__(self, input_ids, scores):
            width = self.counted.shape[1]
            mask = torch.cat([self.counted, torch.ones_like(input_ids[:, width:], dtype=torch.bool)], dim=1)
            ids = torch.where(mask, input_ids, torch.full_like(input_ids, mem.pad))
            picked = scores.gather(1, ids)
            out = scores.scatter(1, ids, torch.where(picked < 0, picked * penalty, picked / penalty))
            out[:, mem.pad] = scores[:, mem.pad]
            return out

    think_ids = mem.enc("<think>")
    kw = {} if think or len(think_ids) != 1 else {"suppress_tokens": think_ids}
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(seqs), batch):
        (ids, att, counted), width = pad_left(seqs[i : i + batch], mem.pad, dev)
        procs = LogitsProcessorList([PenaltyOutsideMemory(counted.bool())]) if penalty != 1.0 else LogitsProcessorList()
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, logits_processor=procs, **kw)
        outs += [o.strip() for o in mem.tok.batch_decode(gen[:, width:], skip_special_tokens=True)]
    return outs


# ------------------------------------------------------------------ states and the article-choice head
class States:
    """Hidden states of the frozen model at the input of the final norm: last prompt token for a question, mean
    over a short version of the note for a candidate article."""

    def __init__(self, model, mem):
        self.model, self.mem, self.h, self.keys = model, mem, None, {}
        model.model.embedding_norm.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        self.h = (args[0] if args else kwargs["hidden_states"]).detach()

    def questions(self, questions, batch=256):
        import torch

        dev = next(self.model.parameters()).device
        out = []
        for i in range(0, len(questions), batch):
            seqs = [(p, [1] * len(p)) for p in (self.mem.chat(q) for q in questions[i : i + batch])]
            (ids, att), _ = pad_left(seqs, self.mem.pad, dev)
            with torch.inference_mode():
                self.model.model(input_ids=ids, attention_mask=att)
            out.append(self.h[:, -1].float())
        return torch.cat(out) if out else torch.zeros(0, self.model.config.hidden_size, device=dev)

    def add_keys(self, rows, batch=256):
        import torch

        dev = next(self.model.parameters()).device
        todo = [r for r in {r[0]: r for r in rows}.values() if r[0] not in self.keys]
        for i in range(0, len(todo), batch):
            group = todo[i : i + batch]
            seqs = [([self.mem.bos] + self.mem.enc(f"{r[0]} ({r[1]}). {r[2]}" if r[1] else f"{r[0]}. {r[2]}"))[:72] for r in group]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), self.mem.pad, dtype=torch.long)
            att = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, : len(s)], att[r, : len(s)] = torch.tensor(s), 1
            ids, att = ids.to(dev), att.to(dev)
            with torch.inference_mode():
                self.model.model(input_ids=ids, attention_mask=att)
            m = att[:, :, None].float()
            for r, v in zip(group, ((self.h.float() * m).sum(1) / m.sum(1)).half().cpu()):
                self.keys[r[0]] = v


class ChoiceHead:
    """Scores each candidate article for a question, with a 'none of them' option. Two linear maps on frozen
    states plus the article size as a prior (D11)."""

    def __init__(self, dim_in, device, dim=256):
        import torch

        self.dev = device
        self.q = torch.nn.Linear(dim_in, dim, bias=False).to(device)
        self.k = torch.nn.Linear(dim_in, dim, bias=False).to(device)
        self.w_prior = torch.nn.Parameter(torch.zeros((), device=device))
        self.null = torch.nn.Parameter(torch.zeros((), device=device))
        self.log_t = torch.nn.Parameter(torch.tensor(2.5, device=device))
        self.stats = None

    def parameters(self):
        return list(self.q.parameters()) + list(self.k.parameters()) + [self.w_prior, self.null, self.log_t]

    def scores(self, q, keys, prior, mask):
        import torch
        import torch.nn.functional as F

        s = torch.einsum("bd,bkd->bk", F.normalize(self.q(q), dim=-1), F.normalize(self.k(keys), dim=-1)) * self.log_t.exp() + self.w_prior * prior
        return torch.cat([s.masked_fill(~mask, -1e4), self.null.expand(len(q), 1)], dim=1)

    def tensors(self, q_states, cand_lists, states):
        import torch

        q_mu, q_sd, k_mu, k_sd = self.stats
        q = (q_states - q_mu) / q_sd
        keys = torch.zeros(len(cand_lists), K, q.shape[1], device=self.dev)
        prior = torch.zeros(len(cand_lists), K, device=self.dev)
        mask = torch.zeros(len(cand_lists), K, dtype=torch.bool, device=self.dev)
        for i, cs in enumerate(cand_lists):
            for j, c in enumerate(cs[:K]):
                keys[i, j] = (states.keys[c[0]].float().to(self.dev) - k_mu[0]) / k_sd[0]
                prior[i, j] = math.log(max(c[4], 1)) - 10.0
                mask[i, j] = True
        return q, keys, prior, mask

    def fit(self, q_states, cand_lists, labels, states, epochs=40):
        """labels[i] is the index of the right candidate in cand_lists[i], or K when it is not among them."""
        import torch
        import torch.nn.functional as F

        all_keys = torch.stack(list(states.keys.values())).float().to(self.dev)
        self.stats = (q_states.mean(0, keepdim=True), q_states.std(0, keepdim=True) + 1e-6, all_keys.mean(0, keepdim=True), all_keys.std(0, keepdim=True) + 1e-6)
        del all_keys
        q, keys, prior, mask = self.tensors(q_states, cand_lists, states)
        y = torch.tensor(labels, device=self.dev)
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            perm = torch.randperm(len(y), device=self.dev)
            for i in range(0, len(perm), 512):
                idx = perm[i : i + 512]
                loss = F.cross_entropy(self.scores(q[idx], keys[idx], prior[idx], mask[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def choose(self, q_state, cands, states, top=3):
        """Up to `top` candidates the head prefers to 'none'."""
        import torch

        if not cands:
            return []
        states.add_keys(cands)
        q, keys, prior, mask = self.tensors(q_state[None], [cands], states)
        with torch.no_grad():
            s = self.scores(q, keys, prior, mask)[0]
        return [cands[j] for j in s[:K].topk(min(top, len(cands[:K]))).indices.tolist() if s[j] > s[K]]


# ------------------------------------------------------------------ LoRA on the global-attention layers
class Switch:
    on = True


class LoRALinear:
    """Wraps a frozen nn.Linear: y = W x + scale * B A x. Float32 adapters on a bf16 model. The shared switch turns
    every adapter off, which gives back the original model exactly: used whenever no note is shown."""

    def __new__(cls, base, r, alpha, switch):
        import torch

        class _LoRA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base, self.switch, self.scale = base, switch, alpha / r
                self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * (1 / math.sqrt(base.in_features)))
                self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

            def forward(self, x):
                y = self.base(x)
                if not self.switch.on:
                    return y
                return y + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)

        return _LoRA()


def add_lora(model, r=16, alpha=32, layers=GLOBAL_LAYERS, names=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")):
    """Adds adapters to the attention of the global layers, the only ones that read the notes. Returns the
    trainable parameters and the switch."""
    switch, params = Switch(), []
    for i in layers:
        attn = model.model.layers[i].self_attn
        for name in names:
            wrapped = LoRALinear(getattr(attn, name), r, alpha, switch)
            setattr(attn, name, wrapped)
            params += [wrapped.A, wrapped.B]
    return params, switch
