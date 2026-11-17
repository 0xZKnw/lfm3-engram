"""D2: how close are LFM3-1.7B-Base and LFM3-1.7B (instruct), and how close is raw text to chat framing?

1. Relative weight change per layer between the two checkpoints.
2. Residual-stream similarity at the input of every layer on the same raw text.
3. Same passage read as raw text vs inside an assistant turn."""
import os
import re


class Tap:
    """Captures the residual stream at the input of every layer, plus the input of the final norm (index n_layers)."""

    def __init__(self, model):
        self.states, self.hooks = {}, []
        layers = list(model.model.layers) + [model.model.embedding_norm]
        for i, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_pre_hook(self._make(i), with_kwargs=True))

    def _make(self, i):
        def hook(module, args, kwargs):
            self.states[i] = (args[0] if args else kwargs["hidden_states"]).detach().float()

        return hook

    def run(self, model, ids):
        self.states = {}
        logits = model(input_ids=ids).logits.float()
        return [self.states[i][0] for i in range(len(self.hooks))], logits[0]


def weight_delta(inst, base):
    import torch

    num, den = {}, {}
    sb = base.state_dict()
    for key, a in inst.state_dict().items():
        if not torch.is_floating_point(a) or key == "lm_head.weight":
            continue
        a, b = a.float(), sb[key].float()
        m = re.match(r"model\.layers\.(\d+)\.(.+?)\.(?:weight|bias)$", key)
        groups = [f"layer_{int(m.group(1)):02d}", f"kind_{m.group(2).split('.')[0]}"] if m else [f"other_{key}"]
        for g in groups + ["all"]:
            num[g] = num.get(g, 0.0) + (a - b).pow(2).sum().item()
            den[g] = den.get(g, 0.0) + b.pow(2).sum().item()
    return {g: round((num[g] / den[g]) ** 0.5, 5) for g in sorted(num)}


def _pair_stats(acc, ha, hb):
    """Accumulate per-layer cosine and relative distance between two lists of [T, D] states."""
    import torch.nn.functional as F

    for i, (a, b) in enumerate(zip(ha, hb)):
        s = acc.setdefault(i, {"cos": 0.0, "rel": 0.0, "norm_a": 0.0, "norm_b": 0.0, "n": 0})
        s["cos"] += F.cosine_similarity(a, b, dim=-1).sum().item()
        s["rel"] += ((a - b).norm(dim=-1) / a.norm(dim=-1)).sum().item()
        s["norm_a"] += a.norm(dim=-1).sum().item()
        s["norm_b"] += b.norm(dim=-1).sum().item()
        s["n"] += a.shape[0]


def _finish(acc):
    return {
        f"{i:02d}": {k: round(v[k] / v["n"], 4) for k in ("cos", "rel", "norm_a", "norm_b")} for i, v in sorted(acc.items())
    }


def _ce(logits, ids):
    import torch.nn.functional as F

    return F.cross_entropy(logits[:-1], ids[1:], reduction="sum").item(), ids.shape[0] - 1


def run(data: str, n_docs: int = 200, max_len: int = 1024, corpora=("wiki_en", "web", "chat"), n_chat_docs: int = 150):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset

    from .common import load_model, load_tokenizer

    tok = load_tokenizer()
    inst, base = load_model("instruct"), load_model("base")
    out = {"weight_delta_rel": weight_delta(inst, base)}
    tap_i, tap_b = Tap(inst), Tap(base)
    dev = "cuda"

    # 2. same raw text through both models
    out["raw_text"] = {}
    with torch.inference_mode():
        for name in corpora:
            tokens = np.load(f"{data}/{name}/test_tokens.npy")
            offsets = np.load(f"{data}/{name}/test_offsets.npy")
            acc, li, lb, kl, n, used = {}, 0.0, 0.0, 0.0, 0, 0
            # consistency check against the batched, padded D1 scorer on the same positions
            d1_path, d1_sum = f"{data}/{name}/scores_instruct.npz", 0.0
            d1_loss = np.load(d1_path)["loss"] if os.path.exists(d1_path) else None
            for d in range(len(offsets) - 1):
                s, e = int(offsets[d]), int(offsets[d + 1])
                if e - s < 128:
                    continue
                ids = torch.from_numpy(tokens[s : min(e, s + max_len)].astype(np.int64)).to(dev)
                hi, logit_i = tap_i.run(inst, ids[None])
                hb, logit_b = tap_b.run(base, ids[None])
                _pair_stats(acc, [h[1:] for h in hi], [h[1:] for h in hb])  # skip BOS
                a, k = _ce(logit_i, ids)
                li, lb, n = li + a, lb + _ce(logit_b, ids)[0], n + k
                if d1_loss is not None:
                    d1_sum += float(d1_loss[s + 1 : s + ids.shape[0]].sum())
                lp_i, lp_b = F.log_softmax(logit_i[:-1], -1), F.log_softmax(logit_b[:-1], -1)
                kl += (lp_i.exp() * (lp_i - lp_b)).sum().item()
                used += 1
                if used >= n_docs:
                    break
            out["raw_text"][name] = {
                "docs": used, "tokens": n, "loss_instruct": round(li / n, 4), "loss_base": round(lb / n, 4),
                "kl_instruct_to_base": round(kl / n, 4), "instruct_vs_base_by_layer": _finish(acc),
                "loss_instruct_from_d1_scorer": round(d1_sum / n, 4) if d1_loss is not None else None,
            }

        # 3. the same Wikipedia passage as raw text and inside an assistant turn
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        pairs = {"instruct_raw_vs_instruct_chat": {}, "base_raw_vs_instruct_chat": {}, "base_raw_vs_base_chat": {}}
        losses = {k: [0.0, 0] for k in ("instruct_raw", "instruct_chat", "base_raw", "base_chat")}
        used, skip = 0, 32
        bos = tok.bos_token_id
        for ex in ds:
            if len(ex["text"]) < 4000:
                continue
            passage = tok(ex["text"], add_special_tokens=False)["input_ids"][:512]
            head = tok(ex["title"] + "\n\n", add_special_tokens=False)["input_ids"]
            prefix = tok.apply_chat_template(
                [{"role": "user", "content": f"Write the Wikipedia article about {ex['title']}."}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False,
            )
            prefix = tok(prefix, add_special_tokens=False)["input_ids"]
            if prefix[0] != bos:
                prefix = [bos] + prefix
            raw = torch.tensor([bos] + head + passage, device=dev)
            chat = torch.tensor(prefix + passage, device=dev)
            r0, c0 = 1 + len(head) + skip, len(prefix) + skip
            states = {}
            for key, tap, model, ids, start in (
                ("instruct_raw", tap_i, inst, raw, r0), ("instruct_chat", tap_i, inst, chat, c0),
                ("base_raw", tap_b, base, raw, r0), ("base_chat", tap_b, base, chat, c0),
            ):
                hs, logits = tap.run(model, ids[None])
                states[key] = [h[start:] for h in hs]
                # loss on the passage tokens only (positions start..end, predicted from start-1..end-1)
                losses[key][0] += F.cross_entropy(logits[start - 1 : -1], ids[start:], reduction="sum").item()
                losses[key][1] += ids.shape[0] - start
            _pair_stats(pairs["instruct_raw_vs_instruct_chat"], states["instruct_raw"], states["instruct_chat"])
            _pair_stats(pairs["base_raw_vs_instruct_chat"], states["base_raw"], states["instruct_chat"])
            _pair_stats(pairs["base_raw_vs_base_chat"], states["base_raw"], states["base_chat"])
            used += 1
            if used == 1:
                out["chat_prefix_example"] = tok.decode(prefix, skip_special_tokens=False)
            if used >= n_chat_docs:
                break
        out["raw_vs_chat"] = {
            "docs": used,
            "passage_loss": {k: round(v[0] / v[1], 4) for k, v in losses.items()},
            **{k: _finish(v) for k, v in pairs.items()},
        }
    return out
