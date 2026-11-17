"""Local test of the integrated model on CPU, with the real LFM3 weights and a small note store.
Needs local/hf_smoke (see lfm3e/export_hf.py). Run: .venv/bin/python tests/test_integrated.py [folder]"""
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "local", "hf_smoke")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.float32).eval()
mem = model.memory
print(f"chargé en {time.time() - t0:.0f} s | {type(model).__name__} | {len(mem.store):,} notes | marqueurs {mem.user_open!r} {mem.user_close!r}")
Base = type(model).__mro__[2]  # the original LFM3 class
assert Base.__name__ == "Lfm3ForCausalLM", Base


def prompt(text, **kw):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True, **kw)


def new_text(out, enc):
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


greedy = dict(max_new_tokens=24, do_sample=False)

# 1. no name in the prompt: no note, and exactly the original model
enc = prompt("what is 17 times 3? answer with the number only.")
a = model.generate(**enc, **greedy)
b = Base.generate(model, **enc, **greedy, suppress_tokens=[mem.think_id])
assert model.last_notes == [[]] and torch.equal(a, b), (model.last_notes, tok.decode(a[0]), tok.decode(b[0]))
print("1. sans note : identique au modèle d'origine |", repr(new_text(a, enc)))

# 2. a name of the store: a note is shown, and the result is what the layout gives through the original generate
q = "Who developed the sitcom Instant Mom?"
enc = prompt(q)
t0 = time.time()
a = model.generate(**enc, **greedy, repetition_penalty=1.0)
dt = time.time() - t0
assert model.last_notes[0] and torch.equal(a[:, : enc["input_ids"].shape[1]], enc["input_ids"]), model.last_notes
i = mem.exact("instant mom", "en")
head, lead, facts = mem.store.note(i)
note = [tok.bos_token_id] + mem.enc("Reference note about") + head + mem.enc(". Use it only if it is relevant, otherwise ignore it.\n") + lead[:96] + facts[:131]
row = enc["input_ids"][0].tolist()
ids = torch.tensor([note + [tok.pad_token_id] * 256 + row])
att = torch.tensor([[1] * len(note) + [0] * 256 + [1] * len(row)])
b = Base.generate(model, input_ids=ids, attention_mask=att, **greedy, repetition_penalty=1.0, suppress_tokens=[mem.think_id])
assert torch.equal(a[:, enc["input_ids"].shape[1] :], b[:, ids.shape[1] :]), (tok.decode(a[0]), tok.decode(b[0, ids.shape[1] :]))
off = model.generate(**enc, **greedy, repetition_penalty=1.0, memory=False)
print(f"2. avec note {model.last_notes if False else [mem.title(i)]} : identique à la mise en page manuelle | {dt:.1f} s")
print("   avec mémoire :", repr(new_text(a, enc)), "| sans :", repr(new_text(off, enc)))
print("   note :", repr(model.memory_notes(q)[0][:230]))

# 3. a batch mixing prompts with and without notes, left-padded, default sampling settings and penalty
tok.padding_side = "left"
qs = ["Who developed the sitcom Instant Mom?", "say hello in one word.", "What was Jim Simons known for?"]
texts = [tok.apply_chat_template([{"role": "user", "content": x}], add_generation_prompt=True, enable_thinking=False, tokenize=False) for x in qs]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
out = model.generate(**enc, max_new_tokens=24)
assert torch.equal(out[:, : enc["input_ids"].shape[1]], enc["input_ids"]) and [bool(n) for n in model.last_notes] == [True, False, True], model.last_notes
for x, o, n in zip(qs, out, model.last_notes):
    print("3.", repr(x), "->", repr(tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True)), "| notes :", n)

# 4. the other ways of calling generate
enc = prompt("What was Jim Simons known for?")
out = model.generate(enc["input_ids"], max_new_tokens=8, do_sample=False, return_dict_in_generate=True, output_scores=True)
assert out.sequences.shape[1] == enc["input_ids"].shape[1] + len(out.scores) and model.last_notes[0]
out = model.generate(**enc, max_new_tokens=6, do_sample=True, num_return_sequences=2)
assert out.shape[0] == 2 and torch.equal(out[:, : enc["input_ids"].shape[1]], enc["input_ids"].repeat(2, 1))
print("4. flux :", end=" ")
model.generate(**enc, max_new_tokens=12, do_sample=False, streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True))

# 5. thinking left on, and French
enc = tok.apply_chat_template([{"role": "user", "content": "What was Jim Simons known for?"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**enc, max_new_tokens=40, do_sample=False)
print("5. réflexion active :", repr(tok.decode(out[0, enc["input_ids"].shape[1] :])[:160]))
import numpy as np

fr = np.flatnonzero(np.asarray(mem.store.lang) == 1)
big = fr[np.argsort(-np.asarray(mem.store.size)[fr])[:400]]
titles = [mem.title(int(i)) for i in big]
pick = next(t for t in titles if 2 <= len(t.split(" (")[0].split()) <= 3 and t.split(" (")[0].replace(" ", "").isalpha())
qf = f"Que peux-tu me dire sur {pick.split(' (')[0]} ?"
enc = prompt(qf)
out = model.generate(**enc, max_new_tokens=40, do_sample=False)
print("   français :", repr(qf), "| langue :", mem.language(qf), "| notes :", model.last_notes[0], "->", repr(new_text(out, enc)[:150]))
assert model.last_notes[0], "aucune note en français"
print("tout est ok")
