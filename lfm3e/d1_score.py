"""D1 step 2 (GPU): per-token loss of a frozen model over the test stream of each corpus."""
import os
import time


def make_windows(offsets, max_len=2048, overlap=256):
    """Split each document [s, e) into windows of at most `max_len` tokens. Later windows re-read `overlap` tokens of
    context that are not scored again. Returns (start, end, score_from): positions score_from..end-1 are scored."""
    wins = []
    for d in range(len(offsets) - 1):
        s, e = int(offsets[d]), int(offsets[d + 1])
        a, score_from = s, s + 1
        while True:
            b = min(a + max_len, e)
            if b > score_from:
                wins.append((a, b, score_from))
            if b >= e:
                break
            score_from, a = b, b - overlap
    return wins


def score_stream(model, tokens, offsets, pad_id, batch=16, max_len=2048, overlap=256):
    """Returns loss (float32, NaN where unscored), ptop (model's top-1 probability) and correct (top-1 == target),
    all aligned with `tokens`."""
    import numpy as np
    import torch

    n = len(tokens)
    loss = np.full(n, np.nan, dtype=np.float32)
    ptop = np.zeros(n, dtype=np.float32)
    correct = np.zeros(n, dtype=bool)
    wins = sorted(make_windows(offsets, max_len, overlap), key=lambda w: w[0] - w[1])
    weight = model.lm_head.weight.float()
    dev = weight.device
    tokens_t = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    with torch.inference_mode():
        for i in range(0, len(wins), batch):
            group = wins[i : i + batch]
            width = group[0][1] - group[0][0]
            ids = torch.full((len(group), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(group), width), dtype=torch.long)
            for r, (a, b, _) in enumerate(group):
                ids[r, : b - a] = tokens_t[a:b]
                mask[r, : b - a] = 1
            ids, mask = ids.to(dev), mask.to(dev)
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            for r, (a, b, sf) in enumerate(group):
                j0, length = sf - a - 1, b - a
                logits = hidden[r, j0 : length - 1].float() @ weight.T
                target = ids[r, j0 + 1 : length]
                lse = torch.logsumexp(logits, dim=-1)
                top = logits.max(dim=-1)
                loss[sf:b] = (lse - logits.gather(-1, target[:, None])[:, 0]).cpu().numpy()
                ptop[sf:b] = torch.exp(top.values - lse).cpu().numpy()
                correct[sf:b] = (top.indices == target).cpu().numpy()
    return loss, ptop, correct


def run(model_key: str, data: str, limit_docs: int = 0, corpora=None):
    import numpy as np

    from .common import load_model, load_tokenizer

    tok = load_tokenizer(model_key)
    model = load_model(model_key)
    out = {}
    for name in sorted(os.listdir(data)):
        if corpora and name not in corpora:
            continue
        path = f"{data}/{name}"
        if not os.path.exists(f"{path}/test_tokens.npy"):
            continue
        tokens = np.load(f"{path}/test_tokens.npy")
        offsets = np.load(f"{path}/test_offsets.npy")
        if limit_docs:
            offsets = offsets[: limit_docs + 1]
            tokens = tokens[: offsets[-1]]
        t = time.time()
        loss, ptop, correct = score_stream(model, tokens, offsets, tok.pad_token_id)
        dt = time.time() - t
        suffix = f"_{limit_docs}" if limit_docs else ""
        np.savez(f"{path}/scores_{model_key}{suffix}.npz", loss=loss, ptop=ptop, correct=correct)
        scored = ~np.isnan(loss)
        out[name] = {
            "tokens": int(len(tokens)), "scored": int(scored.sum()), "loss_nats": round(float(loss[scored].mean()), 4),
            "top1_acc": round(float(correct[scored].mean()), 4), "tok_per_s": round(len(tokens) / dt),
        }
    return out
