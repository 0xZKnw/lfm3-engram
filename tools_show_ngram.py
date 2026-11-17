"""Print a compact view of the D1 n-gram results."""
import json, os, sys
for key in ("base", "instruct"):
    p = f"results/d1_ngram_{key}_full.json"
    if not os.path.exists(p):
        print(key, "not finished"); continue
    print(f"===== {key}")
    for r in json.load(open(p)):
        if "error" in r:
            print("ERROR", r["error"][:300]); continue
        f, q = r["by_train_fraction"]["1.0"], r["by_train_fraction"]["0.25"]
        print(f"{r['corpus']:8s} loss {r['model_loss_nats']:.3f} | train {f['train_tokens']/1e6:.0f}M | keys2-4 gain {f['gain_nats_keys_2_to_4']:.4f} (quarter data {q['gain_nats_keys_2_to_4']:.4f}, no-conf {f['gain_nats_keys_2_to_4_no_confidence_feature']:.4f}) | +1-token keys {f['gain_nats_all_lens']:.4f}")
        print("         by len", f["gain_nats_keys_2_to_4_by_len"], "| coverage", f["coverage_by_context_len"])
        print("         gain share by cat", f["gain_share_by_category_keys_2_to_4"])
        print("         loss share by cat", r["loss_share_by_category"])
        print("         min count:", {t: (v["gain_nats"], f"{v['keys']/1e6:.1f}M") for t, v in f["by_min_count"].items() if t in ("1", "4", "16", "64", "256")})
