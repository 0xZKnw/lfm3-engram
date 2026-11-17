"""D1 step 3 (CPU, numpy only): how much loss could an n-gram memory remove?

For each test token we look up the count-based next-token estimate given the previous 1..4 tokens, and mix it into
the model's probability one context length at a time (shortest first). Each stage has one mixing weight per bucket
(context count, how peaked the count distribution is, model confidence), fitted on half of the test documents; the
loss reduction is reported on the other half. It is a lower bound on what an n-gram memory can add: a real Engram
injects into the hidden states instead of mixing output probabilities.

Convention: context length c corresponds to an Engram n-gram of order c (the key ends at the current token and
serves the prediction of the next one). c = 1 is a single-token key."""
import numpy as np

U = np.uint64
_GOLD, _GOLD2 = U(0x9E3779B97F4A7C15), U(0xD6E8FEB86659FD93)
LAMBDAS = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
N_COUNT_B, N_PEAK_B, N_CONF_B = 15, 5, 5
MIN_BUCKET = 200
THRESHOLDS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]


def _mix(x):
    x = (x ^ (x >> U(30))) * U(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> U(27))) * U(0x94D049BB133111EB)
    return x ^ (x >> U(31))


def ctx_hash(tc, c):
    """out[j] = hash of tc[j : j + c], i.e. the context of position j + c. Length len(tc) - c."""
    n = len(tc) - c
    h = np.zeros(n, dtype=U)
    for k in range(c):
        h = _mix(h ^ ((tc[k : n + k].astype(U) + U(1)) * _GOLD))
    return h


def pair_hash(h, nxt):
    return _mix(h ^ ((nxt.astype(U) + U(1)) * _GOLD2))


def build_table(tc, raw, c):
    """Counts for context length c over a train stream. Returns sorted unique context hashes with their total count
    and the count of their most frequent continuation, and sorted unique (context, next) hashes with counts."""
    ctx = ctx_hash(tc, c)
    pair = pair_hash(ctx, raw[c:])
    up, first, cp = np.unique(pair, return_index=True, return_counts=True)
    ctx_of_pair = ctx[first]
    del pair, first
    uc, cc = np.unique(ctx, return_counts=True)
    del ctx
    order = np.argsort(ctx_of_pair, kind="stable")
    cs = ctx_of_pair[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    mx = np.maximum.reduceat(cp[order], starts)
    assert len(mx) == len(uc)
    return uc, cc, mx, up, cp


def _lookup(keys, vals, q):
    if len(keys) == 0:
        return np.zeros(len(q), dtype=np.int64)
    idx = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return np.where(keys[idx] == q, vals[idx], 0).astype(np.int64)


def test_counts(table, tc_te, raw_te, doc_start, c):
    """Per test position: count of its context, of (context, true next token), and of the context's top
    continuation. Zero where the context would cross a document start or was never seen."""
    uc, cc, mx, up, cp = table
    n = len(tc_te)
    n_ctx, n_pair, n_max = (np.zeros(n, dtype=np.int64) for _ in range(3))
    if n <= c:
        return n_ctx, n_pair, n_max
    ctx = ctx_hash(tc_te, c)
    valid = (np.arange(c, n) - c) >= doc_start[c:]
    n_ctx[c:] = np.where(valid, _lookup(uc, cc, ctx), 0)
    n_max[c:] = np.where(valid, _lookup(uc, mx, ctx), 0)
    n_pair[c:] = np.where(valid, _lookup(up, cp, pair_hash(ctx, raw_te[c:])), 0)
    return n_ctx, n_pair, n_max


def cascade(loss, ptop, counts, orders, dev, ev, use_conf=True, min_count=1):
    """Mix the count estimates into the model probability, one context length at a time. Contexts seen fewer than
    `min_count` times are treated as unseen. Returns the per-token gain on `ev` (nats, zero elsewhere) and the gain
    added by each stage, summed over `ev`."""
    p = np.exp(-np.nan_to_num(loss, nan=0.0).astype(np.float64))
    p0 = p.copy()
    bm = np.clip((ptop * N_CONF_B).astype(np.int64), 0, N_CONF_B - 1) if use_conf else np.zeros(len(loss), dtype=np.int64)
    nb = 1 + N_COUNT_B * N_PEAK_B * N_CONF_B
    by_order = {}
    for c in orders:
        n_ctx, n_pair, n_max = counts[c]
        has = n_ctx >= min_count
        denom = np.maximum(n_ctx, 1)
        p_ng = np.where(has, n_pair / denom, 0.0)
        bc = np.clip(np.floor(np.log2(denom)), 0, N_COUNT_B - 1).astype(np.int64)
        bp = np.clip((n_max / denom * N_PEAK_B).astype(np.int64), 0, N_PEAK_B - 1)
        bucket = np.where(has, 1 + (bc * N_PEAK_B + bp) * N_CONF_B + bm, 0)
        sums = np.stack(
            [np.bincount(bucket[dev], weights=-np.log((1 - lam) * p[dev] + lam * p_ng[dev]), minlength=nb) for lam in LAMBDAS]
        )
        best = LAMBDAS[np.argmin(sums, axis=0)]
        best[np.bincount(bucket[dev], minlength=nb) < MIN_BUCKET] = 0.0
        best[0] = 0.0  # unseen context: nothing to mix
        lam = best[bucket]
        p_new = (1 - lam) * p + lam * p_ng
        by_order[c] = float((np.log(p_new[ev]) - np.log(p[ev])).sum())
        p = p_new
    gain = np.zeros(len(loss))
    gain[ev] = np.log(p[ev]) - np.log(p0[ev])
    return gain, by_order


def analyse(train_raw, test_raw, offsets, loss, ptop, correct, compress, cats, cat_names, fractions=(0.25, 1.0), cmax=4):
    """Full D1 analysis for one corpus. All arrays are numpy; `compress` maps raw token id -> canonical id."""
    doc_id = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    doc_start = np.repeat(offsets[:-1], np.diff(offsets))
    scored = ~np.isnan(loss)
    dev, ev = scored & (doc_id % 2 == 0), scored & (doc_id % 2 == 1)
    n_ev = int(ev.sum())
    tc_te = compress[test_raw]
    target_cat = cats[test_raw]
    per_tok = lambda x: round(float(x) / n_ev, 5)  # noqa: E731

    out = {
        "test_tokens_scored": int(scored.sum()), "eval_tokens": n_ev, "train_tokens": int(len(train_raw)),
        "model_loss_nats": round(float(loss[scored].mean()), 4), "model_top1_acc": round(float(correct[scored].mean()), 4),
        "loss_share_by_category": {
            cat_names[k]: round(float(loss[scored & (target_cat == k)].sum() / loss[scored].sum()), 4) for k in range(len(cat_names))
        },
        "token_share_by_category": {
            cat_names[k]: round(float((scored & (target_cat == k)).sum() / scored.sum()), 4) for k in range(len(cat_names))
        },
        "by_train_fraction": {},
    }
    all_orders, engram_orders = list(range(1, cmax + 1)), list(range(2, cmax + 1))
    for frac in fractions:
        tr = train_raw[: int(len(train_raw) * frac)]
        tc_tr = compress[tr]
        counts, table_sizes = {}, {}
        for c in all_orders:
            table = build_table(tc_tr, tr, c)
            counts[c] = test_counts(table, tc_te, test_raw, doc_start, c)
            table_sizes[c] = {t: int((table[1] >= t).sum()) for t in THRESHOLDS}
            del table
        gain, by_order = cascade(loss, ptop, counts, all_orders, dev, ev)
        total = float(gain.sum())
        cum = np.cumsum([by_order[c] for c in all_orders])
        g_eng, by_order_eng = cascade(loss, ptop, counts, engram_orders, dev, ev)
        res = {
            "train_tokens": int(len(tr)),
            "coverage_by_context_len": {str(c): round(float((counts[c][0][scored] > 0).mean()), 4) for c in counts},
            # single-token keys (length 1) included, cumulative up to each length
            "gain_nats_up_to_context_len": {str(c): per_tok(cum[i]) for i, c in enumerate(all_orders)},
            # Engram-like table: keys of 2 to cmax tokens only
            "gain_nats_keys_2_to_4": per_tok(g_eng.sum()),
            "gain_nats_keys_2_to_4_by_len": {str(c): per_tok(v) for c, v in by_order_eng.items()},
            "gain_nats_keys_2_to_4_no_confidence_feature": per_tok(cascade(loss, ptop, counts, engram_orders, dev, ev, use_conf=False)[0].sum()),
            "gain_share_by_category_keys_2_to_4": {
                cat_names[k]: round(float(g_eng[target_cat == k].sum() / g_eng.sum()), 4) if g_eng.sum() else 0.0 for k in range(len(cat_names))
            },
            "gain_nats_all_lens": per_tok(total),
            # table sizing: keep only keys (2 to cmax tokens) seen at least T times in the train stream
            "by_min_count": {
                str(t): {
                    "gain_nats": per_tok(cascade(loss, ptop, counts, engram_orders, dev, ev, min_count=t)[0].sum()),
                    "keys": int(sum(table_sizes[c][t] for c in engram_orders)),
                }
                for t in THRESHOLDS
            },
        }
        out["by_train_fraction"][str(frac)] = res
    return out
