"""
Verify that fusing SmoothQuant scales into model weights produces
numerically equivalent outputs (no quantization, just the math identity:
    layernorm(x)/s @ (s*W)^T == layernorm(x) @ W^T
).

Steps:
  1. Load model, run forward pass on sample input, capture logits.
  2. Fuse scales into weights in-place.
  3. Run same forward pass, capture logits.
  4. Compare: should match to floating-point tolerance.

Use --dtype float32 to isolate logic correctness from bf16 rounding.
"""

import argparse
import re

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "/root/repos/ckpts/c5_wonbag_bf16"
DEFAULT_SCALES_PATH = "/tmp/smooth_quant_scales.safetensors"

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify SmoothQuant fusion equivalence")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scales-path", type=str, default=DEFAULT_SCALES_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=DTYPE_MAP.keys(),
        help="Model dtype. Use float32 to verify logic correctness without rounding noise.",
    )
    return parser.parse_args()


def get_reference_logits(model, input_ids, device):
    """Run a forward pass and return logits on CPU."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids.to(device))
    return out.logits.cpu()


LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def fuse_scales_into_model(model, scales: dict[str, torch.Tensor]):
    """
    Fuse SmoothQuant scales into model weights in-place.

    For each scale entry (keyed by layernorm name like
    "model.language_model.layers.3.input_layernorm"):
      - LayerNorm: weight /= s  (and bias /= s if present)
      - Projections fed by that layernorm (q/k/v/gate/up_proj + mlp.gate): weight *= s
        along the input dimension (dim=1).

    Arithmetic is done in float32 to minimize rounding during fusion,
    then cast back to the parameter's original dtype.
    """
    state_dict = dict(model.named_parameters())
    projection_suffixes = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "mlp.gate"]

    fused_count = 0
    for scale_key, scale in scales.items():
        m = LAYER_IDX_RE.search(scale_key)
        if m is None:
            print(f"  WARNING: could not parse layer index from {scale_key}, skipping")
            continue
        layer_idx = int(m.group(1))

        ln_prefix = scale_key

        # Fuse layernorm: weight /= s, bias /= s
        ln_weight_key = f"{ln_prefix}.weight"
        ln_bias_key = f"{ln_prefix}.bias"

        if ln_weight_key in state_dict:
            orig_dtype = state_dict[ln_weight_key].dtype
            s = scale.float().to(state_dict[ln_weight_key].device)
            state_dict[ln_weight_key].data.copy_(
                (state_dict[ln_weight_key].data.float() / s).to(orig_dtype)
            )
            if ln_bias_key in state_dict:
                state_dict[ln_bias_key].data.copy_(
                    (state_dict[ln_bias_key].data.float() / s).to(orig_dtype)
                )
        else:
            print(f"  WARNING: {ln_weight_key} not found in model")
            continue

        # Find and fuse all projection weights in this layer
        layer_prefix = scale_key[: scale_key.rindex(".input_layernorm")] + "."
        proj_count = 0
        for param_name, param in state_dict.items():
            if not param_name.startswith(layer_prefix):
                continue
            if not any(param_name.endswith(f"{sf}.weight") for sf in projection_suffixes):
                continue
            # weight shape is [out_features, in_features], scale along in_features (dim=1)
            orig_dtype = param.dtype
            s = scale.float().to(param.device)
            param.data.copy_(
                (param.data.float() * s.unsqueeze(0)).to(orig_dtype)
            )
            proj_count += 1

        fused_count += 1
        print(f"  Layer {layer_idx}: fused layernorm + {proj_count} projections")

    print(f"\nFused {fused_count} layers total.")


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print(f"Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    attn_impl = "eager" if dtype == torch.float32 else "flash_attention_2"
    print(f"Loading model from {args.model_id} (dtype={args.dtype}, attn={attn_impl}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # Prepare a sample input
    sample_text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids

    # Step 1: get reference logits before fusion
    print("\nRunning forward pass BEFORE fusion ...")
    logits_before = get_reference_logits(model, input_ids, device)
    print(f"  Logits shape: {logits_before.shape}")
    print(f"  Logits sample (first 5 of last token): {logits_before[0, -1, :5]}")

    # Step 2: fuse scales
    print(f"\nLoading scales from {args.scales_path} ...")
    scales = load_file(args.scales_path)
    print(f"  {len(scales)} layer scales loaded")

    print("\nFusing scales into model weights ...")
    fuse_scales_into_model(model, scales)

    # Step 3: get logits after fusion
    print("\nRunning forward pass AFTER fusion ...")
    logits_after = get_reference_logits(model, input_ids, device)
    print(f"  Logits shape: {logits_after.shape}")
    print(f"  Logits sample (first 5 of last token): {logits_after[0, -1, :5]}")

    # Step 4: compare
    print("\n" + "=" * 60)
    print(f"COMPARISON (model dtype: {args.dtype})")
    print("=" * 60)

    abs_diff = (logits_before.float() - logits_after.float()).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    rel_diff = abs_diff / (logits_before.float().abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"  Max  absolute diff: {max_abs_diff:.6e}")
    print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
    print(f"  Max  relative diff: {max_rel_diff:.6e}")
    print(f"  Mean relative diff: {mean_rel_diff:.6e}")

    if dtype == torch.float32:
        tolerance = 1e-3
        label = "fp32"
    else:
        tolerance = 8.0
        label = "bf16 (rounding over 48 layers)"

    if max_abs_diff < tolerance:
        print(f"\n  PASS: max diff {max_abs_diff:.6e} < {tolerance} ({label} tolerance)")
    else:
        print(f"\n  FAIL: max diff {max_abs_diff:.6e} >= {tolerance}")
        if dtype != torch.float32:
            print(f"  Try --dtype float32 to check if this is a rounding issue vs logic bug.")

    # Top-1 token agreement
    top1_before = logits_before[0].argmax(dim=-1)
    top1_after = logits_after[0].argmax(dim=-1)
    agreement = (top1_before == top1_after).float().mean().item()
    print(f"\n  Top-1 token agreement: {agreement * 100:.1f}% "
          f"({(top1_before == top1_after).sum()}/{len(top1_before)} positions)")


if __name__ == "__main__":
    main()
