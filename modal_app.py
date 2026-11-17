"""Modal entry points for the diagnostics.

    modal run modal_app.py::d0                       # speed on A100-40GB (+ --h100 for the 5-minute comparison)
    modal run modal_app.py::prep --tag full          # tokenize the six corpora (CPU)
    modal run modal_app.py::score --tag full         # per-token loss of the frozen model (GPU)
    modal run modal_app.py::ngram --tag full         # n-gram headroom (CPU)
    modal run modal_app.py::d2 --tag full            # Base vs instruct (GPU)

Results are written to results/ next to this file."""
import json
import pathlib

import modal

from lfm3e.common import VOL, data_dir

app = modal.App("lfm3-engram")
vol = modal.Volume.from_name("lfm3-engram", create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1", "transformers==5.20.2", "datasets==5.1.0", "tokenizers==0.23.3", "huggingface-hub==2.3.0",
        "accelerate==1.15.0", "safetensors==0.8.0", "numpy",
    )
    .env({"TOKENIZERS_PARALLELISM": "true"})
)
image = base_image.add_local_python_source("lfm3e")
sw_image = base_image.pip_install("orjson").add_local_python_source("lfm3e")
RESULTS = pathlib.Path(__file__).parent / "results"
GPU = "A100-40GB"


def _plain(obj):
    """Round-trip through JSON so nothing torch- or numpy-typed is pickled back to the local machine."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(f"-> {path}")


@app.function(image=image, gpu=GPU, timeout=25 * 60)
def d0_a100(mode: str = "main"):
    from lfm3e import d0_speed

    return _plain(d0_speed.run(mode))


@app.function(image=image, gpu="H100", timeout=10 * 60)
def d0_h100(mode: str = "quick"):
    from lfm3e import d0_speed

    return _plain(d0_speed.run(mode))


@app.function(image=image, cpu=8, memory=24 * 1024, timeout=45 * 60, volumes={VOL: vol})
def prep_corpus(name: str, tag: str, n_test: int, n_train: int):
    from lfm3e import d1_prep

    meta = d1_prep.run(name, data_dir(tag), n_test=n_test, n_train=n_train)
    vol.commit()
    return _plain(meta)


@app.function(image=image, gpu=GPU, timeout=45 * 60, volumes={VOL: vol})
def score_corpora(model_key: str, tag: str, limit_docs: int = 0):
    from lfm3e import d1_score

    out = d1_score.run(model_key, data_dir(tag), limit_docs=limit_docs)
    vol.commit()
    return _plain(out)


@app.function(image=image, cpu=8, memory=32 * 1024, timeout=60 * 60, volumes={VOL: vol})
def ngram_corpus(name: str, tag: str, model_key: str = "instruct"):
    import numpy as np

    from lfm3e import d1_ngram
    from lfm3e.common import CATEGORY_NAMES, build_compress_table, load_tokenizer, token_categories

    tok = load_tokenizer(model_key)
    compress, n_canonical = build_compress_table(tok)
    path = f"{data_dir(tag)}/{name}"
    scores = np.load(f"{path}/scores_{model_key}.npz")
    out = d1_ngram.analyse(
        np.load(f"{path}/train_tokens.npy"), np.load(f"{path}/test_tokens.npy"), np.load(f"{path}/test_offsets.npy"),
        scores["loss"], scores["ptop"], scores["correct"], compress, token_categories(tok), CATEGORY_NAMES,
    )
    out["corpus"], out["canonical_vocab"], out["raw_vocab"] = name, n_canonical, len(tok)
    return _plain(out)


@app.function(image=image, gpu=GPU, timeout=30 * 60, volumes={VOL: vol})
def d2_similarity(tag: str, n_docs: int = 200, n_chat_docs: int = 150):
    from lfm3e import d2_similarity

    return _plain(d2_similarity.run(data_dir(tag), n_docs=n_docs, n_chat_docs=n_chat_docs))


@app.function(image=image, cpu=8, memory=32 * 1024, timeout=30 * 60, volumes={VOL: vol})
def d3_fetch():
    from lfm3e import d3_entity

    out = d3_entity.fetch_articles(f"{VOL}/popqa")
    vol.commit()
    return _plain(out)


@app.function(image=image, gpu="H100", timeout=45 * 60, volumes={VOL: vol})
def d3_run(layers: list, n_baseline: int, n_fail: int, n_ok: int, epochs: int, n_think: int = 400):
    from lfm3e import d3_entity

    return _plain(d3_entity.run(f"{VOL}/popqa", layers=tuple(layers), n_baseline=n_baseline, n_think=n_think, n_fail=n_fail, n_ok=n_ok, epochs=epochs))


@app.function(image=image, gpu="H100", timeout=30 * 60, volumes={VOL: vol})
def d4_run(n_baseline: int, n_fail: int, n_ok: int):
    from lfm3e import d4_kvmem

    return _plain(d4_kvmem.run(f"{VOL}/popqa", n_baseline=n_baseline, n_fail=n_fail, n_ok=n_ok))


@app.function(image=image, cpu=8, memory=32 * 1024, timeout=30 * 60, volumes={VOL: vol})
def d5_build():
    from lfm3e import d5_trigger

    out = d5_trigger.build_candidates(f"{VOL}/popqa")
    vol.commit()
    return _plain(out)


@app.function(image=image, gpu="H100", timeout=45 * 60, volumes={VOL: vol})
def d5_run(n_baseline: int, n_fail: int, n_ok: int, n_e2e: int, biases: list):
    from lfm3e import d5_trigger

    return _plain(d5_trigger.run(f"{VOL}/popqa", n_baseline=n_baseline, n_fail=n_fail, n_ok=n_ok, n_e2e=n_e2e, biases=tuple(biases)))


@app.function(image=image, cpu=8, memory=32 * 1024, timeout=40 * 60, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf-token")])
def d6_build():
    from lfm3e import d6_reason

    out = d6_reason.build(f"{VOL}/d6")
    vol.commit()
    return _plain(out)


@app.function(image=image, timeout=10 * 60, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf-token")])
def d6_gpqa():
    from lfm3e import d6_reason

    out = d6_reason.add_gpqa(f"{VOL}/d6")
    vol.commit()
    return _plain(out)


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={VOL: vol})
def d6_run(n_hotpot: int, n_gsm: int, n_gpqa: int):
    from lfm3e import d6_reason

    return _plain(d6_reason.run(f"{VOL}/d6", f"{data_dir('full')}/web/train_tokens.npy", n_hotpot=n_hotpot, n_gsm=n_gsm, n_gpqa=n_gpqa))


@app.function(image=image, gpu="H100", timeout=20 * 60, volumes={VOL: vol})
def d6_calibrate():
    from lfm3e import d6_reason

    return _plain(d6_reason.calibrate(f"{VOL}/d6", f"{VOL}/popqa", f"{data_dir('full')}/web/train_tokens.npy"))


CORPUS_NAMES = ["wiki_en", "wiki_fr", "arxiv", "web", "code", "chat"]


@app.local_entrypoint()
def d0(h100: bool = False, compiled: bool = False):
    if h100:
        # same quick benchmark on both cards, in parallel
        a100, h = d0_a100.spawn("quick"), d0_h100.spawn("quick")
        _save("d0_quick_a100", a100.get())
        _save("d0_quick_h100", h.get())
    elif compiled:
        _save("d0_a100_compile", d0_a100.remote("compile"))
    else:
        _save("d0_a100", d0_a100.remote("main"))


@app.local_entrypoint()
def prep(tag: str = "full", n_test: int = 3_000_000, n_train: int = 200_000_000, corpora: str = ""):
    args = [(name, tag, n_test, n_train) for name in (corpora.split(",") if corpora else CORPUS_NAMES)]
    _save(f"d1_prep_{tag}", list(prep_corpus.starmap(args)))


@app.local_entrypoint()
def score(tag: str = "full", model_key: str = "instruct", limit_docs: int = 0):
    _save(f"d1_score_{model_key}_{tag}", score_corpora.remote(model_key, tag, limit_docs))


@app.local_entrypoint()
def ngram(tag: str = "full", model_key: str = "instruct", corpora: str = ""):
    names = corpora.split(",") if corpora else CORPUS_NAMES
    results = list(ngram_corpus.starmap([(name, tag, model_key) for name in names], return_exceptions=True))
    _save(f"d1_ngram_{model_key}_{tag}", [r if isinstance(r, dict) else {"error": repr(r)} for r in results])


@app.local_entrypoint()
def d2(tag: str = "full", n_docs: int = 200, n_chat_docs: int = 150):
    _save(f"d2_{tag}", d2_similarity.remote(tag, n_docs, n_chat_docs))


@app.local_entrypoint()
def d3(fetch: bool = False, layers: str = "3,9,15,21", n_baseline: int = 5000, n_fail: int = 240, n_ok: int = 80, epochs: int = 40, n_think: int = 400, name: str = "d3"):
    if fetch:
        _save("d3_fetch", d3_fetch.remote())
    else:
        _save(name, d3_run.remote([int(x) for x in layers.split(",")], n_baseline, n_fail, n_ok, epochs, n_think))


@app.local_entrypoint()
def d4(n_baseline: int = 5000, n_fail: int = 400, n_ok: int = 150, name: str = "d4"):
    _save(name, d4_run.remote(n_baseline, n_fail, n_ok))


@app.local_entrypoint()
def d5(build: bool = False, n_baseline: int = 2500, n_fail: int = 400, n_ok: int = 150, n_e2e: int = 1000, biases: str = "0.25,0.5,1,2", name: str = "d5"):
    if build:
        _save("d5_build", d5_build.remote())
    else:
        _save(name, d5_run.remote(n_baseline, n_fail, n_ok, n_e2e, [float(x) for x in biases.split(",")]))


@app.local_entrypoint()
def d6(build: bool = False, gpqa: bool = False, calibrate: bool = False, n_hotpot: int = 250, n_gsm: int = 150, n_gpqa: int = 198, name: str = "d6"):
    if build:
        _save("d6_build", d6_build.remote())
    elif gpqa:
        _save("d6_gpqa", d6_gpqa.remote())
    elif calibrate:
        _save("d6_calibrate", d6_calibrate.remote())
    else:
        _save(name, d6_run.remote(n_hotpot, n_gsm, n_gpqa))


@app.function(image=sw_image, cpu=16, memory=64 * 1024, timeout=60 * 60, ephemeral_disk=600 * 1024, volumes={VOL: vol})
def d6b_notes(lang: str = "en"):
    from lfm3e import d6b_reason

    out = d6b_reason.build_notes(f"{VOL}/d6", lang)
    if lang == "fr":
        out["word_freq"] = d6b_reason.build_word_freq(f"{data_dir('full')}/wiki_fr/train_tokens.npy", f"{VOL}/d6/word_freq_fr.json")
    vol.commit()
    return _plain(out)


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={VOL: vol})
def d6b_run(n_hotpot: int, n_gsm: int, n_gpqa: int, min_novelty: float, budget_scale: float = 1.0):
    from lfm3e import d6b_reason

    out = d6b_reason.run(f"{VOL}/d6", f"{data_dir('full')}/web/train_tokens.npy", n_hotpot=n_hotpot, n_gsm=n_gsm, n_gpqa=n_gpqa, min_novelty=min_novelty, budget_scale=budget_scale)
    vol.commit()
    return _plain(out)


@app.local_entrypoint()
def d6b(notes: bool = False, lang: str = "en", n_hotpot: int = 250, n_gsm: int = 150, n_gpqa: int = 100, min_novelty: float = 2.0, budget_scale: float = 1.0, name: str = "d6b"):
    if notes:
        _save(f"d6b_notes_{lang}", d6b_notes.remote(lang))
    else:
        _save(name, d6b_run.remote(n_hotpot, n_gsm, n_gpqa, min_novelty, budget_scale))


@app.function(image=image, gpu="H100", timeout=45 * 60, volumes={VOL: vol})
def d7_run(n: int):
    from lfm3e import d7_fr

    return _plain(d7_fr.run(f"{VOL}/d6", f"{data_dir('full')}/web/train_tokens.npy", n=n))


@app.local_entrypoint()
def d7(n: int = 1000, name: str = "d7_fr"):
    _save(name, d7_run.remote(n))


@app.function(image=image, gpu="H100", memory=64 * 1024, timeout=45 * 60, volumes={VOL: vol})
def d8_run(n_train: int, n_test: int, pool: int, n_gate_train: int, n_gate_test: int):
    from lfm3e import d8_probe

    return _plain(d8_probe.run(f"{VOL}/d6", n_train=n_train, n_test=n_test, pool=pool, n_gate_train=n_gate_train, n_gate_test=n_gate_test))


@app.local_entrypoint()
def d8(n_train: int = 40000, n_test: int = 3000, pool: int = 200000, n_gate_train: int = 5000, n_gate_test: int = 2000, name: str = "d8"):
    _save(name, d8_run.remote(n_train, n_test, pool, n_gate_train, n_gate_test))


@app.function(image=image, gpu="H100", memory=48 * 1024, timeout=45 * 60, volumes={VOL: vol})
def d9_run(n_probe: int, n_eval: int):
    from lfm3e import d9_gate

    return _plain(d9_gate.run(f"{VOL}/d6", f"{VOL}/popqa", f"{data_dir('full')}/web/train_tokens.npy", n_probe=n_probe, n_eval=n_eval))


@app.local_entrypoint()
def d9(n_probe: int = 5000, n_eval: int = 1000, name: str = "d9"):
    _save(name, d9_run.remote(n_probe, n_eval))


@app.function(image=sw_image, cpu=16, memory=64 * 1024, timeout=60 * 60, ephemeral_disk=600 * 1024, volumes={VOL: vol})
def d10_run():
    from lfm3e import d10_coverage

    return _plain(d10_coverage.run(f"{VOL}/d6", f"{VOL}/popqa", f"{data_dir('full')}/web/train_tokens.npy"))


@app.local_entrypoint()
def d10():
    _save("d10_coverage", d10_run.remote())


@app.function(image=image, gpu="H100", memory=64 * 1024, timeout=60 * 60, volumes={VOL: vol})
def d11_run(n_eval: int, n_train: int):
    from lfm3e import d11_choose

    return _plain(d11_choose.run(f"{VOL}/d6", f"{VOL}/popqa", f"{data_dir('full')}/web/train_tokens.npy", n_eval=n_eval, n_train=n_train))


@app.local_entrypoint()
def d11(n_eval: int = 1000, n_train: int = 13000, name: str = "d11"):
    _save(name, d11_run.remote(n_eval, n_train))


@app.function(image=image, gpu="H100", timeout=30 * 60, volumes={VOL: vol})
def d12_run(n: int):
    from lfm3e import d12_audit

    return _plain(d12_audit.run(f"{VOL}/d6", f"{VOL}/popqa", n=n))


@app.local_entrypoint()
def d12(n: int = 600, name: str = "d12_audit"):
    _save(name, d12_run.remote(n))


@app.function(image=image, gpu="H100", timeout=60 * 60)
def d13_run(bench: str, n_per_type: int, budget_scale: float):
    from lfm3e import d13_methods

    return _plain(d13_methods.run(bench, n_per_type=n_per_type, budget_scale=budget_scale))


@app.local_entrypoint()
def d13(benches: str = "bbh,math,gsm8k", n_per_type: int = 40, budget_scale: float = 1.0, name: str = "d13"):
    calls = {b: d13_run.spawn(b, n_per_type, budget_scale) for b in benches.split(",")}
    for b, call in calls.items():
        _save(f"{name}_{b}", call.get())


@app.function(image=image, cpu=4, memory=16 * 1024, timeout=30 * 60, volumes={VOL: vol}, secrets=[modal.Secret.from_name("hf-token")])
def light_data():
    """Question sets only (no Wikipedia scan): PopQA, and the HotpotQA / GSM8K / GPQA samples used since D6."""
    import os
    import random

    from datasets import load_dataset

    from lfm3e import d6_reason

    os.makedirs(f"{VOL}/popqa", exist_ok=True)
    os.makedirs(f"{VOL}/d6", exist_ok=True)
    qa = [dict(r) for r in load_dataset("akariasai/PopQA", split="test")]
    with open(f"{VOL}/popqa/popqa.json", "w") as f:
        json.dump(qa, f)
    rng = random.Random(0)
    hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = rng.sample(range(len(hp)), 500)
    hotpot = [{"q": hp[i]["question"], "answer": hp[i]["answer"], "type": hp[i]["type"], "titles": sorted(set(hp[i]["supporting_facts"]["title"]))} for i in idx]
    gs = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(gs)), 300)
    gsm = [{"q": gs[i]["question"], "answer": gs[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]
    gpqa = d6_reason.fetch_gpqa()
    with open(f"{VOL}/d6/benchmarks.json", "w") as f:
        json.dump({"hotpot": hotpot, "gsm8k": gsm, "gpqa": gpqa}, f)
    vol.commit()
    return {"popqa": len(qa), "hotpot": len(hotpot), "gsm8k": len(gsm), "gpqa": len(gpqa)}


@app.local_entrypoint()
def lightdata():
    _save("light_data", light_data.remote())


@app.function(image=image, gpu="H100", memory=64 * 1024, timeout=120 * 60, volumes={VOL: vol})
def d14_run(n_train: int, n_eval: int, n_hotpot_eval: int, epochs: int, tag: str, head_cap: int = 0):
    import os

    from lfm3e import d14_lora

    os.makedirs(f"{VOL}/d14", exist_ok=True)
    out = d14_lora.run(f"{VOL}/d6", f"{VOL}/popqa", f"{data_dir('full')}/web/train_tokens.npy", save_path=f"{VOL}/d14/{tag}.pt",
                       n_train=n_train, n_eval=n_eval, n_hotpot_eval=n_hotpot_eval, epochs=epochs, head_cap=head_cap)
    vol.commit()
    return _plain(out)


@app.local_entrypoint()
def d14(n_train: int = 2500, n_eval: int = 1000, n_hotpot_eval: int = 250, epochs: int = 3, head_cap: int = 0, name: str = "d14"):
    _save(name, d14_run.remote(n_train, n_eval, n_hotpot_eval, epochs, name, head_cap))
