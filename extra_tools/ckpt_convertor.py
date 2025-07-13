# This script mainly deletes the frozen LLM pretrained weights from the checkpoint.
# Optionally, it can also:
# 1) change parameters' key names to match the PEFT LoRA format.
# 2) delete the optimizer state dict to save space.
import argparse
import os
import torch
from pathlib import Path
from typing import Dict
import copy
from collections import OrderedDict
import re

def args_parser():
    parser = argparse.ArgumentParser(description="Delete frozen LLM weights and optionally convert for PEFT LoRA.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Output checkpoint file path. If not set, save as *_copy.pth in the same directory.")
    parser.add_argument("--delete_optimizer", action="store_true", help="Delete optimizer state dict to save space.")
    parser.add_argument("--convert_to_peft_format", action="store_true",
                        help="Convert parameter key names to match PEFT LoRA format.")
    return parser.parse_args()


def should_delete_llm_weight(key: str) -> bool:
    """Identify whether a key is a frozen LLM pretrained weight."""
    pattern = re.compile(
        r"(llm\.llm\.model\..*\.base_layer\.weight|"
        r"attention\.(wqkv|wo)\.weight|"
        r"attention\.rotary_emb|"
        r"feed_forward\.(w1|w2|w3)\.weight)|"
        r"attention_norm\.weight|"
        r"ffn_norm\.weight"
    )
    return bool(pattern.search(key)) and "lora" not in key
def convert_lora_key_to_peft_format(key: str) -> str:
    """
    Convert lora_A/lora_B key to PEFT format by inserting '.base_model' after 'llm.llm'
    Example:
      pts_bbox_head.llm.llm.model.layers.0.feed_forward.w1.lora_A.default.weight
      -> pts_bbox_head.llm.llm.base_model.model.model.layers.0.feed_forward.w1.lora_A.default.weight
    """
    parts = key.split('.')
    try:
        idx = parts.index('llm')  # find the first 'llm'
        if idx + 1 < len(parts) and parts[idx + 1] == 'llm':
            new_parts = parts[:idx + 2] + ['base_model.model'] + parts[idx + 2:]
            return '.'.join(new_parts)
    except ValueError:
        pass
    return key


def convert_state_dict(state_dict: Dict[str, torch.Tensor], convert_to_peft_format: bool) -> Dict[str, torch.Tensor]:
    """
    Remove frozen LLM pretrained weights and optionally convert lora key names.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if should_delete_llm_weight(k):
            continue
        new_k = k
        if convert_to_peft_format:
            if "lora_A." in k or "lora_B." in k or "tok_embeddings" in k or "norm.weight" in k or "output.weight" in k:
                new_k = convert_lora_key_to_peft_format(k)
        new_state_dict[new_k] = v
    return new_state_dict


def main():
    args = args_parser()
    checkpoint_path = Path(args.checkpoint_path)
    save_path = args.save_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    # Default save path: same as input but with _copy.pth
    if not save_path:
        save_path = str(checkpoint_path.with_name(checkpoint_path.stem + "_copy.pth"))
    else:
        import os
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Copy meta if exists
    meta = copy.deepcopy(checkpoint['meta']) if 'meta' in checkpoint else {}

    # Convert the state_dict
    print("Processing state_dict...")
    state_dict = checkpoint['state_dict']
    new_state_dict = convert_state_dict(state_dict, args.convert_to_peft_format)

    # Build new checkpoint
    new_checkpoint = {
        'meta': meta,
        'state_dict': new_state_dict
    }

    # Optionally copy optimizer
    if not args.delete_optimizer and 'optimizer' in checkpoint:
        new_checkpoint['optimizer'] = copy.deepcopy(checkpoint['optimizer'])
    else:
        print("Optimizer state dict is removed.")

    print(f"Saving processed checkpoint to {save_path}")
    torch.save(new_checkpoint, save_path)
    print("Checkpoint conversion completed successfully.")


if __name__ == "__main__":
    main()
