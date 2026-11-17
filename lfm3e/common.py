"""Shared constants and loaders. Heavy imports stay inside functions so the Modal CLI can import this locally."""

MODELS = {"instruct": "LiquidAI/LFM3-1.7B", "base": "LiquidAI/LFM3-1.7B-Base"}
VOL = "/vol"


def data_dir(tag: str = "full") -> str:
    return f"{VOL}/data_{tag}"


def load_tokenizer(key: str = "instruct"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[key], trust_remote_code=True)


def load_model(key: str = "instruct", device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODELS[key], dtype=torch.bfloat16, trust_remote_code=True)
    return model.to(device).eval()


def build_compress_table(tok):
    """Engram's tokenizer compression (deepseek-ai/Engram demo): raw token id -> canonical id after
    NFKC / accent stripping / lowercasing / whitespace folding. Returns (int32 lookup, n_canonical)."""
    import numpy as np
    from tokenizers import Regex, normalizers

    sentinel = ""
    norm = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab = len(tok)
    key2new, lookup = {}, np.empty(vocab, dtype=np.int32)
    for tid in range(vocab):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.convert_ids_to_tokens(tid)
        else:
            key = norm.normalize_str(text) or text
        lookup[tid] = key2new.setdefault(key, len(key2new))
    return lookup, len(key2new)


CATEGORY_NAMES = ["suite_de_mot", "debut_minuscule", "debut_majuscule", "chiffre", "ponctuation_autre", "special"]


def token_categories(tok):
    """Rough per-token category (index into CATEGORY_NAMES). 'suite_de_mot' = alphabetic token with no leading
    whitespace, which also catches words glued to punctuation or CJK text: an approximation."""
    import numpy as np

    vocab = len(tok)
    special = set(tok.all_special_ids)
    cats = np.zeros(vocab, dtype=np.int8)
    for tid in range(vocab):
        if tid in special:
            cats[tid] = 5
            continue
        s = tok.decode([tid], skip_special_tokens=False)
        st = s.lstrip(" \n\t\r")
        if not st or not (st[0].isalpha() or st[0].isdigit()):
            cats[tid] = 4
        elif st[0].isdigit():
            cats[tid] = 3
        elif len(st) == len(s):
            cats[tid] = 0
        elif st[0].isupper():
            cats[tid] = 2
        else:
            cats[tid] = 1
    return cats
