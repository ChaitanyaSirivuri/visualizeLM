#!/usr/bin/env python3
"""CLI to generate relevancy-map heatmap overlays for a LLaVA model.

Targets `transformers >= 5.0`.

Usage:
    python test_interpret.py --image path/to/image.jpg --prompt "What is in this image?"
"""

import argparse
import os
import re
import sys

# Support importing src from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.visualization import draw_heatmap_on_image
from src.model import get_processor_model
from src.relevancy import construct_relevancy_map

SEPARATORS = set(
    [
        ".", ",", "?", "!", ":", ";", "</s>", "/", "(", ")", "[", "]", "{", "}",
        "<", ">", "|", "\\", "-", "_", "+", "=", "*", "&", "^", "%", "$", "#",
        "@", "~", "`", " ", "\t", "\n", "\r", "\x0b", "\x0c",
    ]
)


def sanitize_filename(s: str) -> str:
    s = s.strip("▁").strip("_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    return s or "token"


def save_relevancy_overlays(
    word_rel_maps,
    rel_type,
    img_idx,
    image_seq_length,
    recovered_image,
    output_dir,
    p_low: float = 60.0,
    p_high: float = 99.0,
    blur_radius: float = 2.0,
    max_alpha: int = 200,
    alpha_gamma: float = 1.2,
):
    """Render per-token relevancy maps as heatmap overlays on the image."""
    if rel_type not in word_rel_maps:
        raise ValueError(
            f"Relevancy type '{rel_type}' not found. Available: {list(word_rel_maps.keys())}"
        )

    rel_map_dict = word_rel_maps[rel_type]
    os.makedirs(output_dir, exist_ok=True)

    # ViT-side maps carry an extra CLS token, so their side length is image_seq_length + 1.
    vit_side_len = image_seq_length + 1
    grid_side = int(round(image_seq_length**0.5))
    if grid_side * grid_side != image_seq_length:
        raise ValueError(
            f"image_seq_length={image_seq_length} is not a perfect square; "
            "relevancy overlays require a square patch grid."
        )

    saved = []
    for i, (rel_key, rel_map) in enumerate(rel_map_dict.items()):
        if rel_key is None or rel_key in SEPARATORS:
            continue

        if rel_map.shape[-1] != vit_side_len:
            image_slice = rel_map[-1, :][img_idx : img_idx + image_seq_length]
        else:
            image_slice = rel_map[0, 1:]

        heat = image_slice.reshape(grid_side, grid_side).float().cpu().numpy()
        overlay = draw_heatmap_on_image(
            heat,
            recovered_image,
            normalize=True,
            p_low=p_low,
            p_high=p_high,
            blur_radius=blur_radius,
            max_alpha=max_alpha,
            alpha_gamma=alpha_gamma,
        )

        fname = f"{i:03d}_{sanitize_filename(rel_key)}.png"
        out_path = os.path.join(output_dir, fname)
        overlay.save(out_path)
        saved.append(out_path)
        print(f"  Saved: {out_path}")

    return saved


def build_prompt(processor, text_prompt: str) -> str:
    """Render the user prompt via the processor's chat template."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def resolve_eos_token_id(processor, model):
    """Pick the EOS id matching the underlying text model (Gemma uses <end_of_turn>)."""
    if model.config.text_config.model_type == "gemma":
        ids = processor.tokenizer("<end_of_turn>", add_special_tokens=False).input_ids
        if ids:
            return ids[0]
    return processor.tokenizer.eos_token_id


def main():
    parser = argparse.ArgumentParser(
        description="Generate relevancy-map heatmap overlays for a LLaVA model."
    )
    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="What is in this image?")
    parser.add_argument("--output_dir", type=str, default="./output_relevancy")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--rel_type",
        type=str,
        default="all_v2",
        choices=["llama", "llama_token", "vit", "all", "all_v2"],
        help=(
            "Which relevancy map variant to render. 'all_v2' combines LLM- and "
            "ViT-space relevancy and is usually the cleanest; 'vit' is pure "
            "ViT-space attention; 'llama' is the raw LLM-space rollout (noisier)."
        ),
    )
    parser.add_argument("--p_low", type=float, default=60.0,
                        help="Lower percentile for heatmap normalization. Default 60.")
    parser.add_argument("--p_high", type=float, default=99.0,
                        help="Upper percentile for heatmap normalization. Default 99.")
    parser.add_argument("--blur_radius", type=float, default=2.0,
                        help="Gaussian blur radius (pixels) on the upsampled heatmap. Default 2.")
    parser.add_argument("--max_alpha", type=int, default=200,
                        help="Peak overlay opacity (0-255). Default 200.")
    parser.add_argument("--alpha_gamma", type=float, default=1.2,
                        help="Overlay alpha falloff; >1 makes low-heat areas more transparent.")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name_or_path}")
    processor, model = get_processor_model(args)
    image_seq_length = model.config.image_seq_length
    print(f"Model loaded. image_seq_length={image_seq_length}")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    prompt = build_prompt(processor, args.prompt)
    print(f"Prompt: {prompt}")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    img_token_id = model.config.image_token_id
    img_pos = torch.where(input_ids == img_token_id)[1]
    if img_pos.numel() == 0:
        raise RuntimeError(
            "Could not locate <image> token in input_ids. "
            "Ensure the chat template correctly inserts the image slot."
        )
    img_idx = img_pos[0].item()
    print(f"Image token index: {img_idx}")

    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    eos_token_id = resolve_eos_token_id(processor, model)

    print("Generating response...")
    outputs = model.generate(
        **inputs,
        do_sample=args.temperature > 0.001,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_attentions=True,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=eos_token_id,
    )

    output_ids = outputs.sequences.reshape(-1)[input_ids.shape[-1] :].tolist()
    generated_text = processor.tokenizer.decode(output_ids)
    generated_text_tokenized = processor.tokenizer.tokenize(generated_text)

    print("\n" + "=" * 50)
    print("GENERATED TEXT:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)

    img_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    img_recover = inputs.pixel_values[0].cpu().float() * img_std + img_mean
    img_recover = to_pil_image(img_recover.clamp(0, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    img_recover.save(os.path.join(args.output_dir, "preprocessed_image.png"))

    print("\nBuilding relevancy maps...")
    word_rel_maps = construct_relevancy_map(
        tokenizer=processor.tokenizer,
        model=model,
        input_ids=input_ids,
        tokens=generated_text_tokenized,
        outputs=outputs,
        output_ids=output_ids,
        img_idx=img_idx,
        image_seq_length=image_seq_length,
    )

    print(f"\nRendering '{args.rel_type}' relevancy heatmaps...")
    saved = save_relevancy_overlays(
        word_rel_maps,
        args.rel_type,
        img_idx,
        image_seq_length,
        img_recover,
        args.output_dir,
        p_low=args.p_low,
        p_high=args.p_high,
        blur_radius=args.blur_radius,
        max_alpha=args.max_alpha,
        alpha_gamma=args.alpha_gamma,
    )

    print(f"\nDone. Saved {len(saved)} relevancy overlays to: {args.output_dir}")


if __name__ == "__main__":
    main()
