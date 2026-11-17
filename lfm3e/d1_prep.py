"""D1 step 1 (CPU): stream a corpus, tokenize it, write a held-out test stream and a large train stream.

Both streams are flat int32 arrays of documents laid out as [BOS] tokens [EOS]. The test stream also has
`test_offsets` (document starts, plus the final end)."""
import json
import os
import time

CORPORA = {
    "wiki_en": dict(path="wikimedia/wikipedia", name="20231101.en", split="train"),
    "wiki_fr": dict(path="wikimedia/wikipedia", name="20231101.fr", split="train"),
    "arxiv": dict(path="gfissore/arxiv-abstracts-2021", name=None, split="train"),
    "web": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"),
    "code": dict(path="code-search-net/code_search_net", name="python", split="train"),
    "chat": dict(path="HuggingFaceH4/ultrachat_200k", name=None, split="train_sft"),
}
TEST_DOC_CAP = 8192


def _text(name, ex, tok):
    """Returns (text, already_templated) or None to skip the example."""
    if name.startswith("wiki"):
        if len(ex["text"]) < 2000:  # stubs are mostly lists and templates
            return None
        return f"{ex['title']}\n\n{ex['text']}", False
    if name == "arxiv":
        return f"{' '.join(ex['title'].split())}\n\n{' '.join(ex['abstract'].split())}", False
    if name == "web":
        return ex["text"], False
    if name == "code":
        return ex["whole_func_string"], False
    if name == "chat":
        return tok.apply_chat_template(ex["messages"], tokenize=False), True
    raise KeyError(name)


def run(name: str, out_dir: str, n_test: int = 3_000_000, n_train: int = 200_000_000, max_minutes: float = 25.0):
    import numpy as np
    from datasets import load_dataset

    from .common import load_tokenizer

    t0 = time.time()
    tok = load_tokenizer()
    bos, eos = tok.bos_token_id, tok.eos_token_id
    cfg = CORPORA[name]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True).shuffle(seed=0, buffer_size=10_000)

    test, test_offsets, train = [], [0], []
    n_te = n_tr = docs_te = docs_tr = 0
    batch, flags, stopped = [], [], "exhausted"

    def flush():
        nonlocal n_te, n_tr, docs_te, docs_tr
        if not batch:
            return
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids, templated in zip(enc, flags):
            if templated:
                if not ids or ids[0] != bos:
                    ids = [bos] + ids
            else:
                ids = [bos] + ids + [eos]
            if len(ids) < 32:
                continue
            if n_te < n_test:
                ids = ids[:TEST_DOC_CAP]
                test.append(np.asarray(ids, dtype=np.int32))
                n_te += len(ids)
                test_offsets.append(n_te)
                docs_te += 1
            else:
                train.append(np.asarray(ids, dtype=np.int32))
                n_tr += len(ids)
                docs_tr += 1
        batch.clear()
        flags.clear()

    for ex in ds:
        item = _text(name, ex, tok)
        if item is None:
            continue
        batch.append(item[0])
        flags.append(item[1])
        if len(batch) >= 512:
            flush()
            if n_tr >= n_train:
                stopped = "target"
                break
            if time.time() - t0 > max_minutes * 60:
                stopped = "time"
                break
    flush()

    os.makedirs(f"{out_dir}/{name}", exist_ok=True)
    np.save(f"{out_dir}/{name}/test_tokens.npy", np.concatenate(test))
    np.save(f"{out_dir}/{name}/test_offsets.npy", np.asarray(test_offsets, dtype=np.int64))
    np.save(f"{out_dir}/{name}/train_tokens.npy", np.concatenate(train) if train else np.zeros(0, np.int32))
    meta = {
        "corpus": name, "test_tokens": n_te, "test_docs": docs_te, "train_tokens": n_tr, "train_docs": docs_tr,
        "stopped": stopped, "seconds": round(time.time() - t0),
        "sample": tok.decode(test[0][:60], skip_special_tokens=False),
    }
    with open(f"{out_dir}/{name}/meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta
