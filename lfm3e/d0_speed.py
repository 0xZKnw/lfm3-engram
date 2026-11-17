"""D0: does the model load and answer, and how many tokens per second do we get?"""
import time

SEQ = 2048
SANITY_TEXT = (
    "The mitochondrion is an organelle found in the cells of most eukaryotes, such as animals, plants and fungi. "
    "Mitochondria have a double membrane structure and use aerobic respiration to generate adenosine triphosphate, "
    "which is used throughout the cell as a source of chemical energy. They were discovered by Albert von Kolliker "
    "in 1857 in the voluntary muscles of insects."
)


def _bench(fn, tokens_per_call, warmup=2, iters=5):
    import torch

    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = time.time() - t
        return {"tok_per_s": round(tokens_per_call * iters / dt), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}


def chunked_ce_backward(model, hidden, labels, rows_per_chunk=2):
    """Cross-entropy over a 128K vocab without materialising all logits: backward per chunk into a detached copy of
    the hidden states, then one backward through the backbone. Returns the mean loss (float)."""
    import torch
    import torch.nn.functional as F

    h = hidden.detach().requires_grad_(True)
    n = labels[:, 1:].numel()
    total = 0.0
    for r in range(0, h.shape[0], rows_per_chunk):
        logits = model.lm_head(h[r : r + rows_per_chunk, :-1]).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[r : r + rows_per_chunk, 1:].reshape(-1), reduction="sum") / n
        loss.backward()
        total += loss.item()
    hidden.backward(h.grad)
    return total


def run(mode: str = "main"):
    import torch

    from .common import load_model, load_tokenizer

    out = {"mode": mode, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t = time.time()
    tok = load_tokenizer()
    model = load_model()
    out["load_s"] = round(time.time() - t, 1)
    out["params"] = sum(p.numel() for p in model.parameters())
    dev = "cuda"

    def rand_ids(b):
        return torch.randint(0, 120_000, (b, SEQ), device=dev)

    if mode == "main":
        ids = tok(SANITY_TEXT, return_tensors="pt").input_ids.to(dev)
        out["sanity_starts_with_bos"] = bool(ids[0, 0].item() == tok.bos_token_id)
        with torch.inference_mode():
            out["sanity_loss_nats"] = round(model(input_ids=ids, labels=ids).loss.item(), 3)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": "What is the capital of Burkina Faso? Answer in one short sentence."}],
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True,
            ).to(dev)
            gen = model.generate(**chat, max_new_tokens=48, do_sample=False)
        out["sanity_prompt"] = tok.decode(chat["input_ids"][0], skip_special_tokens=False)
        out["sanity_answer"] = tok.decode(gen[0, chat["input_ids"].shape[1] :], skip_special_tokens=False)

        # Inference: backbone only (what the scorer uses, logits are computed per row afterwards)
        out["inference_backbone"] = {}
        for b in (8, 16, 32):
            ids = rand_ids(b)

            def f():
                with torch.inference_mode():
                    model.model(input_ids=ids).last_hidden_state

            out["inference_backbone"][f"batch_{b}"] = _bench(f, b * SEQ)

    for p in model.parameters():
        p.requires_grad_(False)
    delta = torch.nn.Parameter(torch.zeros(1, 1, model.config.hidden_size, device=dev, dtype=torch.bfloat16))

    def add_delta(module, args, kwargs):
        return (args[0] + delta,) + args[1:], kwargs

    hook = model.model.layers[3].register_forward_pre_hook(add_delta, with_kwargs=True)

    def train_step(b, backbone):
        ids = rand_ids(b)
        hidden = backbone(input_ids=ids).last_hidden_state
        chunked_ce_backward(model, hidden, ids)
        delta.grad = None

    if mode == "main":
        # Training with the backbone frozen and one trainable vector injected at layer 3 (the Engram case)
        out["train_frozen"] = {}
        for b in (8, 16, 24):
            out["train_frozen"][f"batch_{b}"] = _bench(lambda: train_step(b, model.model), b * SEQ, warmup=1, iters=4)

        # Same with every weight trainable (upper bound on the cost of a LoRA / full fine-tune step)
        for p in model.parameters():
            p.requires_grad_(True)

        def full_step():
            train_step(8, model.model)
            model.zero_grad(set_to_none=True)

        out["train_all_weights"] = {"batch_8": _bench(full_step, 8 * SEQ, warmup=1, iters=4)}
        for p in model.parameters():
            p.requires_grad_(False)

    if mode == "quick":
        out["train_frozen"] = {}
        for b in (4, 8):
            out["train_frozen"][f"batch_{b}"] = _bench(lambda: train_step(b, model.model), b * SEQ, warmup=1, iters=4)
        ids = rand_ids(16)

        def f():
            with torch.inference_mode():
                model.model(input_ids=ids).last_hidden_state

        out["inference_backbone"] = {"batch_16": _bench(f, 16 * SEQ)}

    if mode == "compile":
        t = time.time()
        compiled = torch.compile(model.model)
        try:
            out["train_frozen_compiled"] = {"batch_16": _bench(lambda: train_step(16, compiled), 16 * SEQ, warmup=3, iters=4)}
            out["compile_s"] = round(time.time() - t, 1)
        except Exception as e:  # noqa: BLE001 - we want the message, compile failures are informative
            out["compile_error"] = f"{type(e).__name__}: {str(e)[:500]}"

    hook.remove()
    return out
