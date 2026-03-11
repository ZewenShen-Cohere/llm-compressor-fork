"""
Verify SmoothQuant scale export for the C5 (Cohere2Vision + Cohere2MoE) model.

Runs a small calibration pass with export-only mode (apply_smoothing=False),
then loads the exported safetensors and prints per-layer statistics.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

DEFAULT_MODEL_PATH = "/root/repos/ckpts/c5_wonbag_bf16"
DEFAULT_CALIB_PATH = "/root/repos/llm-compressor-fork/agentic_reasoning-calibration-dataset-783-v4.2.jsonl"
DEFAULT_EXPORT_PATH = "/tmp/smooth_quant_scales.safetensors"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify SmoothQuant scale export")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--calib-set-path", type=str, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--export-path", type=str, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--num-calibration-samples", type=int, default=100)
    parser.add_argument("--max-sequence-length", type=int, default=64000)
    parser.add_argument("--smoothing-strength", type=float, default=0.5)
    return parser.parse_args()


def inspect_scales(path: str):
    """Load exported scales and print diagnostics."""
    scales = load_file(path)
    print(f"\n{'=' * 70}")
    print(f"Exported scales: {path}")
    print(f"Number of layers with scales: {len(scales)}")
    print(f"{'=' * 70}\n")

    print(
        f"{'Layer':<50} {'Shape':>12} {'Min':>10} {'Max':>10} "
        f"{'Mean':>10} {'Std':>10}"
    )
    print("-" * 102)

    all_mins, all_maxs = [], []
    for name in sorted(scales.keys()):
        t = scales[name]
        mn = t.min().item()
        mx = t.max().item()
        mean = t.mean().item()
        std = t.std().item()
        all_mins.append(mn)
        all_maxs.append(mx)
        print(
            f"{name:<50} {str(tuple(t.shape)):>12} {mn:>10.4f} {mx:>10.4f} "
            f"{mean:>10.4f} {std:>10.4f}"
        )

    print("-" * 102)
    print(f"{'Global':.<50} {'':>12} {min(all_mins):>10.4f} {max(all_maxs):>10.4f}")
    print()

    near_zero = sum(1 for s in scales.values() for v in s if v < 1e-4)
    very_large = sum(1 for s in scales.values() for v in s if v > 100)
    total_elements = sum(s.numel() for s in scales.values())
    print(f"Elements near zero (<1e-4): {near_zero}/{total_elements}")
    print(f"Elements very large (>100): {very_large}/{total_elements}")

    if near_zero > 0.01 * total_elements:
        print("WARNING: Many near-zero scales detected — may cause numerical issues")
    if very_large > 0.01 * total_elements:
        print("WARNING: Many very large scales detected — may cause numerical issues")

    print()


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model from {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    print(f"Model class: {model.__class__.__name__}")

    print(f"\nLoading calibration dataset from {args.calib_set_path} ...")
    ds = pd.read_json(args.calib_set_path, lines=True, orient="records")
    ds = ds.sample(n=args.num_calibration_samples, random_state=42)
    ds = Dataset.from_pandas(ds)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = [
        SmoothQuantModifier(
            smoothing_strength=args.smoothing_strength,
            export_scales_path=args.export_path,
            apply_smoothing=False,
        ),
    ]

    print(f"\nRunning oneshot calibration (export-only, no weight modification) ...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
        shuffle_calibration_samples=False,
    )

    if not Path(args.export_path).exists():
        print(f"ERROR: Expected export file not found at {args.export_path}")
        sys.exit(1)

    inspect_scales(args.export_path)
    print("Verification complete.")


if __name__ == "__main__":
    main()
