"""Configuration of LFM3 with a note memory. The weights of LFM3 itself are not part of this model: they are
loaded from `base_model`, unchanged."""
from transformers import PretrainedConfig


class LFM3MemConfig(PretrainedConfig):
    model_type = "lfm3_mem"

    def __init__(
        self,
        base_model="LiquidAI/LFM3-1.7B",
        base_revision=None,
        memory_file="memory.safetensors",
        head_file="head.safetensors",
        gap=256,  # masked tokens between a note and what follows: hides the note from the local layers
        lead_tokens=96,  # tokens of the summary shown to the model
        facts_tokens=131,  # tokens of the facts block shown ("\nFacts:\n" + 128)
        key_tokens=72,  # tokens of an article read to compute its key for the choice head
        candidates=12,  # articles considered per name
        top=3,  # notes the head may keep per name
        max_notes=3,  # notes shown per prompt
        max_spans=2,  # names looked up per prompt
        languages=("en", "fr"),
        suppress_think_after_empty_block=True,
        backbone_attr="model",
        final_norm_attr="embedding_norm",
        **kwargs,
    ):
        self.base_model, self.base_revision = base_model, base_revision
        self.memory_file, self.head_file = memory_file, head_file
        self.gap, self.lead_tokens, self.facts_tokens, self.key_tokens = gap, lead_tokens, facts_tokens, key_tokens
        self.candidates, self.top, self.max_notes, self.max_spans = candidates, top, max_notes, max_spans
        self.languages = list(languages)
        self.suppress_think_after_empty_block = suppress_think_after_empty_block
        self.backbone_attr, self.final_norm_attr = backbone_attr, final_norm_attr
        super().__init__(**kwargs)
