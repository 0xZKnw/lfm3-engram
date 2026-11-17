"""Local check of the D1 n-gram logic on a synthetic 2nd-order Markov stream (no GPU, no model)."""
import collections
import numpy as np
from lfm3e import d1_ngram as ng
from lfm3e.d1_score import make_windows

rng = np.random.default_rng(0)
V = 50
trans = rng.dirichlet(np.full(V, 0.05), size=(V, V))  # peaked next-token distribution given 2 previous tokens


def stream(n_docs, doc_len):
    toks, offs = [], [0]
    for _ in range(n_docs):
        d = [V, int(rng.integers(V)), int(rng.integers(V))]  # V plays BOS
        for _ in range(doc_len - 4):
            d.append(int(rng.choice(V, p=trans[d[-2], d[-1]])))
        d.append(V + 1)  # EOS
        toks += d
        offs.append(len(toks))
    return np.array(toks, dtype=np.int32), np.array(offs, dtype=np.int64)


train, _ = stream(400, 500)
test, offs = stream(60, 500)
uni = np.bincount(train, minlength=V + 2) / len(train)
loss = -np.log(uni[test]).astype(np.float32)   # the "model" is a unigram LM
loss[offs[:-1]] = np.nan                       # BOS positions are not scored
ptop = np.full(len(test), uni.max(), dtype=np.float32)
correct = test == uni.argmax()
compress = np.arange(V + 2, dtype=np.int32)
cats = np.zeros(V + 2, dtype=np.int8)

# 1. counts match a brute-force dictionary
doc_start = np.repeat(offs[:-1], np.diff(offs))
for c in (1, 2, 3):
    brute_ctx, brute_pair = collections.Counter(), collections.Counter()
    for i in range(c, len(train)):
        k = tuple(train[i - c:i])
        brute_ctx[k] += 1
        brute_pair[k + (int(train[i]),)] += 1
    n_ctx, n_pair, n_max = ng.test_counts(ng.build_table(train, train, c), test, test, doc_start, c)
    for i in rng.integers(0, len(test), 400):
        k = tuple(test[i - c:i]) if i - c >= doc_start[i] else None
        assert n_ctx[i] == (brute_ctx[k] if k else 0), (c, i)
        assert n_pair[i] == (brute_pair[k + (int(test[i]),)] if k else 0), (c, i)
        if k and brute_ctx[k]:
            assert n_max[i] == max(v for kk, v in brute_pair.items() if kk[:-1] == k), (c, i)
print("counts OK")

# 2. the mixture recovers most of the gap between the unigram model and the true 2nd-order process
out = ng.analyse(train, test, offs, loss, ptop, correct, compress, cats, ["all"], fractions=(0.25, 1.0), cmax=4)
full = out["by_train_fraction"]["1.0"]
true_loss = float(np.mean([-np.log(trans[test[i - 2], test[i - 1], test[i]]) for i in range(len(test)) if i - 3 >= doc_start[i] and test[i] < V]))
print("model loss", out["model_loss_nats"], "| true process loss", round(true_loss, 3))
print("gain up to ctx len", full["gain_nats_up_to_context_len"])
print("keys 2-4:", full["gain_nats_keys_2_to_4"], full["gain_nats_keys_2_to_4_by_len"])
print("quarter of train, keys 2-4:", out["by_train_fraction"]["0.25"]["gain_nats_keys_2_to_4"])
print("by min count", {k: v for k, v in full["by_min_count"].items() if k in ("1", "8", "64")})
g = full["gain_nats_up_to_context_len"]
gap = out["model_loss_nats"] - true_loss
assert g["4"] >= g["3"] >= g["2"] > g["1"] > 0 and g["2"] > 0.8 * gap
assert full["gain_nats_keys_2_to_4"] > 0.8 * gap
assert full["gain_nats_keys_2_to_4"] >= out["by_train_fraction"]["0.25"]["gain_nats_keys_2_to_4"]
assert full["by_min_count"]["1"]["gain_nats"] >= full["by_min_count"]["64"]["gain_nats"]
assert full["by_min_count"]["1"]["keys"] > full["by_min_count"]["64"]["keys"]

# 3. windows: every non-BOS token is scored exactly once
seen = np.zeros(int(offs[-1]), dtype=int)
for a, b, sf in make_windows(offs, max_len=128, overlap=32):
    assert b - a <= 128 and a <= sf - 1
    seen[sf:b] += 1
expect = np.ones_like(seen); expect[offs[:-1]] = 0
assert (seen == expect).all()
print("windows OK")
