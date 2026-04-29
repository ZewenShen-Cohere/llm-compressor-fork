import pytest
import torch
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationScheme,
)

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers.min_max import (
    MemorylessMinMaxObserver,
    StaticMinMaxObserver,
)
from llmcompressor.recipe import Recipe


def _mxfp8_args(rounding: str = "nearest", dynamic: bool = False):
    return QuantizationArgs(
        num_bits=8,
        type="float",
        strategy="group",
        group_size=32,
        scale_dtype=torch.uint8,
        zp_dtype=torch.uint8,
        symmetric=True,
        dynamic=dynamic,
        mxfp_scale_rounding=rounding,
    )


def test_explicit_mxfp8_scale_rounding_survives_config_resolution():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=_mxfp8_args("ceil"),
        input_activations=_mxfp8_args("mse", dynamic=True),
    )
    modifier = QuantizationModifier(config_groups={"group_0": scheme})

    resolved_scheme = modifier.resolve_quantization_config().config_groups["group_0"]

    assert resolved_scheme.weights.mxfp_scale_rounding == "ceil"
    assert resolved_scheme.input_activations.mxfp_scale_rounding == "mse"


def test_mxfp8_preset_uses_default_scale_rounding():
    modifier = QuantizationModifier(scheme="MXFP8")

    resolved_scheme = modifier.resolve_quantization_config().config_groups["group_0"]

    assert resolved_scheme.weights.mxfp_scale_rounding == "nearest"
    assert resolved_scheme.input_activations.mxfp_scale_rounding == "nearest"


def test_recipe_parses_nested_mxfp8_scale_rounding():
    recipe = Recipe.create_instance(
        """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    config_groups:
                        group_0:
                            targets: ["Linear"]
                            weights:
                                num_bits: 8
                                type: float
                                strategy: group
                                group_size: 32
                                scale_dtype: torch.uint8
                                zp_dtype: torch.uint8
                                symmetric: true
                                mxfp_scale_rounding: ceil
        """
    )

    modifier = recipe.modifiers[0]
    weights = modifier.config_groups["group_0"].weights

    assert weights.mxfp_scale_rounding == "ceil"


def test_mxfp8_ceil_scale_rounding_observer_prevents_overflow():
    weight = torch.tensor([[449.0] + [0.25] * 31], dtype=torch.float32)
    observer = MemorylessMinMaxObserver(base_name="weight", args=_mxfp8_args("ceil"))

    scale, _ = observer(weight)

    assert torch.all(weight.abs().amax(dim=-1) / scale.squeeze(-1) <= FP8_E4M3_DATA.max)
    assert torch.equal(scale, torch.tensor([[2.0]]))


def test_mxfp8_mse_scale_rounding_observer_selects_lower_error_candidate():
    lower_wins = torch.tensor([449.0] + [0.002] * 31, dtype=torch.float32)
    upper_wins = torch.full((32,), 700.0, dtype=torch.float32)
    weight = torch.stack((lower_wins, upper_wins))
    observer = MemorylessMinMaxObserver(base_name="weight", args=_mxfp8_args("mse"))

    scale, _ = observer(weight)

    assert torch.equal(scale, torch.tensor([[1.0], [2.0]]))


def test_mxfp8_mse_scale_rounding_recompute_requires_observed_values():
    observer = StaticMinMaxObserver(base_name="input", args=_mxfp8_args("mse"))
    observer.past_min_vals = torch.tensor([0.0])
    observer.past_max_vals = torch.tensor([700.0])

    with pytest.raises(ValueError, match="without observed grouped values"):
        observer.recompute_qparams()
