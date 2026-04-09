"""Model loading + attention-capture hooks for the LVLM relevancy pipeline.

    model.model.language_model.layers          # decoder layers (no LM head here)
    model.model.vision_tower.vision_model.encoder.layers  # CLIP ViT layers
    model.config.image_seq_length              # number of image tokens (e.g. 576)
    model.config.image_token_id                # <image> placeholder token id
    model.config.text_config.model_type        # 'llama', 'gemma', ...
"""

import logging

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)


_orig_sample = LlavaForConditionalGeneration._sample
if not getattr(_orig_sample, "_enable_grad_patched", False):
    _patched = torch.enable_grad()(_orig_sample)
    _patched._enable_grad_patched = True  # type: ignore[attr-defined]
    LlavaForConditionalGeneration._sample = _patched


def _make_attn_hook(model, attr_name: str, src_name: str, is_warning: bool = False):
    """Forward hook that retains attention weights for gradient backprop.

    The storage list is looked up on the model by name on every call (not
    captured in the closure) so callers can safely reset the buffer with
    ``model.enc_attn_weights = []`` between runs without orphaning the hook.
    """

    def hook(module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            (logger.warning if is_warning else logger.error)(
                "Attention weights were not returned for %s. "
                "Ensure `output_attentions=True` and attn_implementation='eager'.",
                src_name,
            )
            return output
        attn = output[1]
        attn.requires_grad_(True)
        attn.retain_grad()
        getattr(model, attr_name).append(attn)
        return output

    return hook


def get_processor_model(args):
    """Load the processor + LLaVA model and register attention-capture hooks."""
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    quant_config = None
    if getattr(args, "load_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif getattr(args, "load_8bit", False):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map=getattr(args, "device_map", "auto"),
        attn_implementation="eager",
    )

    model.model.vision_tower.config.output_attentions = True

    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    for layer in model.model.language_model.layers:
        layer.self_attn.register_forward_hook(
            _make_attn_hook(model, "enc_attn_weights", "language_model", is_warning=False)
        )

    for layer in model.model.vision_tower.vision_model.encoder.layers:
        layer.self_attn.register_forward_hook(
            _make_attn_hook(model, "enc_attn_weights_vit", "vision_tower", is_warning=True)
        )

    return processor, model
