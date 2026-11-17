"""LFM3 with a note memory, as one model: `from_pretrained`, then `generate`, with nothing around it.

What `generate` does, per prompt:
  1. reads the last user message and finds the names and terms it mentions (triggers.py)
  2. looks them up in the note store (store.py); for a name shared by several articles, a small head picks the
     articles that fit the question, or none
  3. puts the chosen notes in front of the prompt, each followed by `gap` masked tokens:
         [note][masked gap] ... [prompt]
     the gap keeps the notes out of reach of the convolution and sliding-window layers, so only the global
     attention layers read them
  4. generates, then returns the sequences as if the notes had never been there

The weights of LFM3 are loaded from the original repository and are not modified. A prompt that triggers no note
goes through the original `generate` call untouched (except for `suppress_think_after_empty_block`, see below).

    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    out = model.generate(**tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda"))
    model.last_notes          # titles of the notes shown for each prompt of the last call
    model.generate(..., memory=False)   # the model without its memory"""
import collections
import math
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, PreTrainedModel

from .configuration_lfm3_mem import LFM3MemConfig
from .store import LANGS, Store, open_tensors
from .triggers import CAP_STOP, FR_STOP, find_triggers, pick_spans

NOTE_PREFIX = "Reference note about"
NOTE_SUFFIX = ". Use it only if it is relevant, otherwise ignore it.\n"
_FR_ONLY, _EN_ONLY = FR_STOP - CAP_STOP, CAP_STOP - FR_STOP
_PROBE = "QZXPROBEXZQ"
_HUB_KEYS = ("cache_dir", "force_download", "local_files_only", "token", "proxies")


class _Has:
    def __init__(self, fn):
        self.fn = fn

    def __contains__(self, surface):
        return self.fn(surface)


class NoteMemory:
    """Everything that does not need the network: reading the prompt, the lookups, the notes as tokens."""

    def __init__(self, store, tok, config, token_freq=None):
        self.store, self.tok, self.cfg = store, tok, config
        self.bos, self.pad = tok.bos_token_id, tok.pad_token_id
        self.pre, self.post = self.enc(NOTE_PREFIX), self.enc(NOTE_SUFFIX)
        self.token_freq = token_freq  # occurrences per million tokens of general text, per token id; None: unknown
        self._cand, self._ntok = {}, {}
        self.user_open, self.user_close = self._turn_markers()
        think = self.enc("<think>")
        self.think_id = think[0] if len(think) == 1 else None

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ---- reading the prompt
    def _render(self, text):
        try:
            return self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        except Exception:
            return None

    def _turn_markers(self):
        """What the chat template writes before and after a user message, found by rendering a probe."""
        text = self._render(_PROBE)
        if not text or _PROBE not in text:
            return None, None
        before, after = text.split(_PROBE, 1)
        bos = self.tok.bos_token or ""
        if bos and before.startswith(bos):
            before = before[len(bos) :]
        marks = set(self.tok.all_special_tokens) | set(self.tok.get_added_vocab())
        ends = [after.find(s) + len(s) for s in marks if s and after.find(s) >= 0]
        return (before or None), (after[: min(ends)] if ends else after.strip() or None)

    def user_text(self, ids):
        """The last user message of a prompt given as token ids; the whole text when it is not a chat prompt."""
        text = self.tok.decode(ids, skip_special_tokens=False)
        if self.user_open and self.user_open in text:
            text = text[text.rfind(self.user_open) + len(self.user_open) :]
            if self.user_close and self.user_close in text:
                text = text[: text.find(self.user_close)]
            return text
        return self.tok.decode(ids, skip_special_tokens=True)

    def question_ids(self, text):
        """The message alone, as a fresh one-turn prompt: what the choice head reads the question from."""
        rendered = self._render(text)
        ids = self.enc(rendered if rendered is not None else text)
        return ids if ids and ids[0] == self.bos else [self.bos] + ids

    def thinking_closed(self, ids):
        """The prompt ends with an empty thinking block: the user turned thinking off."""
        return self.tok.decode(ids[-8:], skip_special_tokens=False).rstrip().endswith("</think>")

    def language(self, text):
        words = re.findall(r"[^\W\d_]+", text.lower())
        fr, en = sum(w in _FR_ONLY for w in words), sum(w in _EN_ONLY for w in words)
        lang = "fr" if fr > en or (fr == en and re.search(r"[éèêàùçœ]", text.lower())) else "en"
        return lang if lang in self.cfg.languages else self.cfg.languages[0]

    # ---- lookups
    def candidates(self, surface, lang):
        """Article ids answering to this name, most developed first; other languages only if this one has none."""
        key = (surface, lang)
        if key not in self._cand:
            if len(self._cand) > 200_000:
                self._cand.clear()
            rows = []
            for code in [lang] + [x for x in self.cfg.languages if x != lang]:
                rows = self.store.ids_for(surface, code)
                if rows:
                    break
            ids = list(dict.fromkeys(i for i, _ in rows))[: self.cfg.candidates]
            exact = next((i for i, e in rows if e), None)
            self._cand[key] = (ids, exact)
        return self._cand[key][0]

    def exact(self, surface, lang):
        """The most developed article whose title is exactly this name, or None."""
        self.candidates(surface, lang)
        return self._cand[(surface, lang)][1]

    def _word_tokens(self, word):
        if word not in self._ntok:
            if len(self._ntok) > 200_000:
                self._ntok.clear()
            self._ntok[word] = self.enc(" " + word)
        return self._ntok[word]

    def _freq(self, word):
        ids = self._word_tokens(word.lower())
        return float(self.token_freq[ids[0]]) if self.token_freq is not None and len(ids) == 1 else 0.0

    def spans(self, text, lang):
        """Capitalised names of the message that have candidates (homonyms included)."""
        return [s for _, _, s in pick_spans(text[:20000], _Has(lambda s: bool(self.candidates(s, lang))), max_spans=self.cfg.max_spans, general=True)]

    def strict(self, text, lang):
        """Precision-first lookups: exact titles only. Technical terms need word frequencies, which exist for
        English only, so other languages get the name rules alone."""
        out = []
        for surface, kind in find_triggers(text[:20000], lambda s: self.exact(s, lang) is not None, self._freq, lambda w: len(self._word_tokens(w))):
            if kind == "concept" and (lang != "en" or self.token_freq is None):
                continue
            i = self.exact(surface, lang)
            if i is not None and i not in out:
                out.append(i)
        return out[: self.cfg.max_notes]

    # ---- notes as tokens
    def note(self, i):
        head, lead, facts = self.store.note(i)
        return [self.bos] + self.pre + head + self.post + lead[: self.cfg.lead_tokens] + facts[: self.cfg.facts_tokens]

    def key(self, i):
        head, lead, _ = self.store.note(i)
        return ([self.bos] + head + lead)[: self.cfg.key_tokens]

    def prior(self, i):
        """log of the article size, centred: more developed articles are more often the one meant."""
        return int(self.store.size[i]) / 8 * math.log(2) - 10.0

    def title(self, i):
        return self.tok.decode(self.store.note(i)[0]).strip()

    def text(self, i):
        return self.tok.decode(self.note(i)[1:])

    def layout(self, notes, tail):
        """[note][masked gap]...[tail] -> ids, attention mask, mask of the tokens the repetition penalty may count."""
        ids, att, counted = [], [], []
        for n in notes:
            ids += n + [self.pad] * self.cfg.gap
            att += [1] * len(n) + [0] * self.cfg.gap
            counted += [0] * (len(n) + self.cfg.gap)
        return ids + tail, att + [1] * len(tail), counted + [1] * len(tail)


class PenaltyOutsideMemory(LogitsProcessor):
    """The usual repetition penalty, except that the tokens of the notes do not count: the model must stay free
    to copy a name or a date from a note."""

    def __init__(self, counted, penalty, pad):
        self.counted, self.penalty, self.pad = counted, penalty, pad

    def __call__(self, input_ids, scores):
        width = self.counted.shape[1]
        counted = self.counted
        if counted.shape[0] != input_ids.shape[0]:  # several sequences per prompt
            counted = counted.repeat_interleave(input_ids.shape[0] // counted.shape[0], dim=0)
        mask = torch.cat([counted, torch.ones_like(input_ids[:, width:], dtype=torch.bool)], dim=1)
        ids = torch.where(mask, input_ids, torch.full_like(input_ids, self.pad))
        picked = scores.gather(1, ids)
        out = scores.scatter(1, ids, torch.where(picked < 0, picked * self.penalty, picked / self.penalty))
        out[:, self.pad] = scores[:, self.pad]
        return out


def _positions(att):
    """What generate() derives from the attention mask: masked tokens do not advance the position."""
    return (att.long().cumsum(-1) - 1).masked_fill(att == 0, 1)


class MemoryGeneration:
    """Mixed into the LFM3 class: adds the memory to `generate`. Everything else is LFM3's own code."""

    # ---- internal states for the choice head
    def _mem_states(self, seqs, last, batch=64):
        """States at the input of the final norm. last=True: at the last token (sequences are left-padded);
        last=False: mean over the sequence (right-padded)."""
        mem, dev, out = self.memory, self.device, []
        backbone = getattr(self, mem.cfg.backbone_attr)
        for i in range(0, len(seqs), batch):
            group = seqs[i : i + batch]
            width = max(map(len, group))
            ids = torch.full((len(group), width), mem.pad, dtype=torch.long)
            att = torch.zeros((len(group), width), dtype=torch.long)
            for r, s in enumerate(group):
                if last:
                    ids[r, width - len(s) :], att[r, width - len(s) :] = torch.tensor(s), 1
                else:
                    ids[r, : len(s)], att[r, : len(s)] = torch.tensor(s), 1
            ids, att = ids.to(dev), att.to(dev)
            self.__dict__["_mem_capture"] = True
            try:
                with torch.inference_mode():
                    backbone(input_ids=ids, attention_mask=att, position_ids=_positions(att))
            finally:
                self.__dict__["_mem_capture"] = False
            h = self.__dict__.pop("_mem_h").float()
            if last:
                out.append(h[:, -1])
            else:
                m = att[:, :, None].float()
                out.append((h * m).sum(1) / m.sum(1))
        return torch.cat(out) if out else torch.zeros(0, self.config.hidden_size, device=dev)

    def _mem_keys(self, ids):
        cache = self.__dict__["_mem_key_cache"]
        todo = [i for i in dict.fromkeys(ids) if i not in cache]
        if todo:
            for i, v in zip(todo, self._mem_states([self.memory.key(i) for i in todo], last=False).half().cpu()):
                cache[i] = v
            while len(cache) > 50_000:
                cache.popitem(last=False)
        return torch.stack([cache[i] for i in ids]).float().to(self.device)

    def _mem_head(self):
        head, dev = self.__dict__["_mem_head_cpu"], self.device
        if head is None:
            return None
        if self.__dict__.get("_mem_head_dev") != dev:
            self.__dict__["_mem_head_on"] = {k: v.to(dev).float() for k, v in head.items()}
            self.__dict__["_mem_head_dev"] = dev
        return self.__dict__["_mem_head_on"]

    def _mem_choose(self, q_state, cands):
        """Candidates the head prefers to 'none of them', best first. Without a head: the most developed one."""
        mem, head = self.memory, self._mem_head()
        if not cands:
            return []
        if head is None:
            return cands[:1]
        F = torch.nn.functional
        q = (q_state[None] - head["q_mu"]) / head["q_sd"]
        keys = (self._mem_keys(cands) - head["k_mu"]) / head["k_sd"]
        prior = torch.tensor([mem.prior(i) for i in cands], device=q.device)
        s = (F.normalize(q @ head["q"].T, dim=-1) @ F.normalize(keys @ head["k"].T, dim=-1).T)[0] * head["log_t"].exp() + head["w_prior"] * prior
        order = s.topk(min(mem.cfg.top, len(cands))).indices.tolist()
        return [cands[j] for j in order if s[j] > head["null"]]

    def memory_plan(self, prompts):
        """For each prompt (token ids), the article ids of the notes to show."""
        mem = self.memory
        texts = [mem.user_text(p) for p in prompts]
        langs = [mem.language(t) for t in texts]
        spans = [[mem.candidates(s, lang) for s in mem.spans(t, lang)] for t, lang in zip(texts, langs)]
        asked = [k for k, sp in enumerate(spans) if sp]
        q_states = {}
        if asked and self._mem_head() is not None:
            states = self._mem_states([mem.question_ids(texts[k]) for k in asked], last=True)
            q_states = {k: states[n] for n, k in enumerate(asked)}
            self._mem_keys([i for k in asked for cands in spans[k] for i in cands])  # one batch for all candidates
        plans = []
        for k, (t, lang) in enumerate(zip(texts, langs)):
            chosen = [i for cands in spans[k] for i in self._mem_choose(q_states.get(k), cands)] + mem.strict(t, lang)
            plan, entities = [], set()
            for i in chosen:
                qid = int(mem.store.qid[i])
                if i in plan or (qid and qid in entities):  # the same article, or the same thing in another language
                    continue
                plan.append(i)
                entities.add(qid)
            plans.append(plan[: mem.cfg.max_notes])
        return plans

    def memory_notes(self, prompt):
        """The notes that would be shown for a prompt (text of one user message, or token ids), as text."""
        ids = self.memory.question_ids(prompt) if isinstance(prompt, str) else list(prompt)
        return [self.memory.text(i) for i in self.memory_plan([ids])[0]]

    # ---- generation
    def generate(self, inputs=None, *args, memory=True, **kwargs):
        mem = getattr(self, "memory", None)
        ids = kwargs.get("input_ids", inputs)
        usable = mem is not None and torch.is_tensor(ids) and ids.dim() == 2 and not ids.is_floating_point() and kwargs.get("inputs_embeds") is None
        if not usable:
            return super().generate(inputs, *args, **kwargs)
        att = kwargs.get("attention_mask")
        att = torch.ones_like(ids) if att is None else att
        rows = [ids[r][att[r].bool()].tolist() for r in range(ids.shape[0])]
        if mem.cfg.suppress_think_after_empty_block and mem.think_id is not None and all(mem.thinking_closed(r) for r in rows):
            kwargs["suppress_tokens"] = list(kwargs.get("suppress_tokens") or []) + [mem.think_id]
        plans = self.memory_plan(rows) if memory else [[] for _ in rows]
        self.last_notes = [[mem.title(i) for i in p] for p in plans]
        if not any(plans):
            return super().generate(inputs, *args, **kwargs)

        seqs = [mem.layout([mem.note(i) for i in p], row) for row, p in zip(rows, plans)]
        width = max(len(s[0]) for s in seqs)
        new_ids = torch.full((len(seqs), width), mem.pad, dtype=ids.dtype)
        new_att = torch.zeros((len(seqs), width), dtype=att.dtype)
        counted = torch.zeros((len(seqs), width), dtype=torch.bool)
        for r, (i_, a_, c_) in enumerate(seqs):  # left padding
            new_ids[r, width - len(i_) :] = torch.tensor(i_, dtype=ids.dtype)
            new_att[r, width - len(i_) :] = torch.tensor(a_, dtype=att.dtype)
            counted[r, width - len(i_) :] = torch.tensor(c_, dtype=torch.bool)
        kwargs.pop("input_ids", None)
        kwargs["attention_mask"] = new_att.to(ids.device)
        config = kwargs.get("generation_config") or self.generation_config
        penalty = kwargs.get("repetition_penalty", getattr(config, "repetition_penalty", None)) or 1.0
        if penalty != 1.0:
            kwargs["repetition_penalty"] = 1.0
            kwargs["logits_processor"] = LogitsProcessorList([PenaltyOutsideMemory(counted.to(ids.device), penalty, mem.pad)] + list(kwargs.get("logits_processor") or []))
        if "prefill_chunk_size" not in kwargs and len(seqs) * width > 130_000:  # bounds the memory of the first pass
            kwargs["prefill_chunk_size"] = max(64, 130_000 // len(seqs))
        out = super().generate(new_ids.to(ids.device), *args, **kwargs)

        seq = out if torch.is_tensor(out) else out.sequences
        prompt = ids.repeat_interleave(seq.shape[0] // ids.shape[0], dim=0)
        seq = torch.cat([prompt, seq[:, width:]], dim=1)  # as if the notes had never been there
        if torch.is_tensor(out):
            return seq
        out.sequences = seq
        return out


def _file(path, name, optional=False, **hub):
    if os.path.isdir(path):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
        if optional:
            return None
        raise FileNotFoundError(p)
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=path, filename=name, **hub)
    except Exception:
        if optional:
            return None
        raise


def attach_memory(model, path, config=None, tokenizer=None, **hub):
    """Gives a loaded LFM3 model the memory found in `path` (a folder or a Hub repository)."""
    config = config or LFM3MemConfig.from_pretrained(path, **hub)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **hub)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    head_path = _file(path, config.head_file, optional=True, **hub)
    head = None
    if head_path:
        tensors, _ = open_tensors(head_path)
        head = {k: torch.from_numpy(v.copy()) for k, v in tensors.items()}
    token_freq = head.pop("token_freq").numpy() if head and "token_freq" in head else None
    if head is not None and "q" not in head:
        head = None  # the file only carries the word frequencies
    model.__class__ = type("LFM3MemForCausalLM", (MemoryGeneration, model.__class__), {})
    model.memory = NoteMemory(Store(_file(path, config.memory_file, **hub)), tokenizer, config, token_freq)
    model.memory_config = config
    model.last_notes = []
    model.__dict__.update(_mem_head_cpu=head, _mem_key_cache=collections.OrderedDict(), _mem_capture=False)

    def capture(module, args, kwargs):
        if model.__dict__["_mem_capture"]:
            model.__dict__["_mem_h"] = (args[0] if args else kwargs["hidden_states"]).detach()

    getattr(getattr(model, config.backbone_attr), config.final_norm_attr).register_forward_pre_hook(capture, with_kwargs=True)
    return model


class LFM3MemForCausalLM(PreTrainedModel):
    """Entry point for `AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)`. What it returns is the
    original LFM3 model, loaded from `config.base_model`, with the memory attached to it."""

    config_class = LFM3MemConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        hub = {k: kwargs[k] for k in _HUB_KEYS if k in kwargs}
        if kwargs.get("revision"):
            hub["revision"] = kwargs["revision"]
        if not isinstance(config, LFM3MemConfig):
            config = LFM3MemConfig.from_pretrained(pretrained_model_name_or_path, **hub)
        base = {k: v for k, v in kwargs.items() if not k.startswith("_") and k not in ("revision", "subfolder", "code_revision", "config", "trust_remote_code")}
        if config.base_revision:
            base["revision"] = config.base_revision
        model = AutoModelForCausalLM.from_pretrained(config.base_model, *model_args, trust_remote_code=True, **base)
        return attach_memory(model, str(pretrained_model_name_or_path), config, **hub)
