"""
RTN quantization for MoE expert modules.

Quantizes only the MoE expert layers (gate_proj, up_proj, down_proj) of a
given model using one of three formats: nvfp4a16, mxfp4a16, or int4a16.
All other components (attention, embeddings, router gate) are left untouched.

RTN (Round-To-Nearest) requires no calibration data.

Usage:
    python rtn_moe_quantize.py --quant-type nvfp4a16
    python rtn_moe_quantize.py --quant-type mxfp4a16
    python rtn_moe_quantize.py --quant-type int4a16
    python rtn_moe_quantize.py --quant-type int4a16 --model-id Qwen/Qwen3-30B-A3B
"""

import argparse

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# ── Quantization configurations ──────────────────────────────────────────────

QUANT_CONFIGS = {
    "nvfp4a16": dict(
        num_bits=4,
        type="float",
        strategy="tensor_group",
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    ),
    "mxfp4a16": dict(
        num_bits=4,
        type="float",
        strategy="group",
        symmetric=True,
        dynamic=False,
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
    ),
    "int4a16": dict(
        num_bits=4,
        type="int",
        strategy="group",
        symmetric=True,
        dynamic=False,
        group_size=32,
    ),
}

# ── MoE-only targeting ───────────────────────────────────────────────────────

MOE_TARGETS = [
    "re:.*mlp\\.experts\\.[0-9]+\\.gate_proj$",
    "re:.*mlp\\.experts\\.[0-9]+\\.up_proj$",
    "re:.*mlp\\.experts\\.[0-9]+\\.down_proj$",
]

IGNORE = [
    "lm_head",
    "re:.*mlp.gate$",
    "re:.*q_proj$",
    "re:.*k_proj$",
    "re:.*v_proj$",
    "re:.*o_proj$",
]

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTN quantization (MoE experts only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        required=True,
        choices=list(QUANT_CONFIGS.keys()),
        help="Quantization format to apply",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-235B-A22B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if omitted)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    model_name = args.model_id.rstrip("/").split("/")[-1]
    save_dir = args.save_dir or f"{model_name}-{args.quant_type}-rtn-moe"

    print(f"Model       : {args.model_id}")
    print(f"Quant type  : {args.quant_type}")
    print(f"Save dir    : {save_dir}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build quantization recipe
    weights_arg = QuantizationArgs(**QUANT_CONFIGS[args.quant_type])

    recipe = [
        QuantizationModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=MOE_TARGETS, weights=weights_arg
                )
            },
            ignore=IGNORE,
        ),
    ]

    # Apply RTN quantization (no calibration data needed)
    oneshot(model=model, recipe=recipe)

    # Save compressed model
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
