# lfm3-engram

Experiments on giving LFM3-1.7B (Liquid AI) a memory, without touching its weights.

- [PLAN.md](PLAN.md): the initial plan (an Engram-style n-gram memory). Largely superseded by what the diagnostics showed.
- [RESULTS.md](RESULTS.md): every measurement so far, the architecture that came out of them, and the open reservations.

## Layout

| Path | Content |
|---|---|
| `lfm3e/` | One module per experiment (`d0_speed` … `d12_audit`) and shared helpers |
| `modal_app.py` | Modal entry points: `modal run modal_app.py::d9`, etc. |
| `results/` | Raw JSON output of each run |
| `tests/` | Local test of the n-gram counting logic (numpy only) |

## Status

Research code, written to answer questions quickly. Experiments run on Modal (H100 / CPU); nothing here is a
packaged model. Results that involve a trained component say so in RESULTS.md.

The model is used under the LFM Open License v1.0. Notes come from Wikipedia (CC BY-SA 4.0) through Wikimedia's
structured-contents export.
