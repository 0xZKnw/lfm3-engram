"""Assembles the folder published on Hugging Face: our code, the configuration, the tokenizer, the note store and
the choice head. LFM3's weights are not copied: the model loads them from Liquid's repository."""
import os
import shutil

CODE = ("configuration_lfm3_mem.py", "modeling_lfm3_mem.py", "store.py", "triggers.py")


def export(out_dir: str, store_path: str, head_path: str = "", base_model: str = "LiquidAI/LFM3-1.7B", link: bool = False, **config):
    from transformers import AutoTokenizer

    from .configuration_lfm3_mem import LFM3MemConfig

    os.makedirs(out_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in CODE:
        shutil.copy(os.path.join(here, name), os.path.join(out_dir, name))
    cfg = LFM3MemConfig(base_model=base_model, **config)
    cfg.auto_map = {"AutoConfig": "configuration_lfm3_mem.LFM3MemConfig", "AutoModelForCausalLM": "modeling_lfm3_mem.LFM3MemForCausalLM"}
    cfg.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(out_dir)
    for src, name in ((store_path, cfg.memory_file), (head_path, cfg.head_file)):
        dst = os.path.join(out_dir, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if src:
            os.symlink(os.path.abspath(src), dst) if link else shutil.copy(src, dst)
    return sorted(os.listdir(out_dir))
