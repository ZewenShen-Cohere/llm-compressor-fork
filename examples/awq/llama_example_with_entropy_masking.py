"""
AWQ Quantization Example with Entropy-Based Token Masking

This example demonstrates AWQ quantization where the loss_mask is derived from
prediction entropy rather than chat template roles. High-entropy tokens represent
positions where the model is most uncertain, making them more informative for AWQ
calibration.

The approach:
1. Tokenize a calibration dataset
2. Run a forward pass to compute per-token prediction entropy
3. Select the top N% highest-entropy tokens as the loss_mask
4. Run AWQ with use_loss_mask=True so calibration focuses on these tokens
"""

import torch
import torch.nn.functional as F
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Fewer samples for fast iteration.
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 512

# Entropy masking config: tokens with entropy >= this quantile are selected.
# 0.75 means the top 25% highest-entropy tokens are masked in.
ENTROPY_QUANTILE = 0.75

# Number of samples to visualize.
NUM_DISPLAY_SAMPLES = 3

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)


def compute_token_entropies(model, dataset, device):
    """
    Compute per-token prediction entropy for each sample.

    For each position i, the entropy reflects the model's uncertainty about
    token i+1. This aligns with AWQ's per-position loss mask semantics.

    Processes one sample at a time to manage memory (logits for one sample
    with vocab size 128256 at seq_len 512 is ~250MB in float32).

    Returns a list of 1D CPU tensors, one per sample, each of shape [seq_len].
    """
    model.eval()
    all_entropies = []

    for idx in range(len(dataset)):
        input_ids = torch.tensor(dataset[idx]["input_ids"], device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            # logits shape: [1, seq_len, vocab_size]
            logits = outputs.logits.float()  # float32 for numerical stability

        # Compute entropy: H = -sum(p * log(p))
        # Use log_softmax for numerical stability (log-sum-exp trick internally)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # [1, seq_len]

        all_entropies.append(entropy.squeeze(0).cpu())

        # Free GPU memory
        del logits, log_probs, probs, entropy, outputs
        torch.cuda.empty_cache()

        if (idx + 1) % 10 == 0:
            print(f"  Computed entropy for {idx + 1}/{len(dataset)} samples")

    return all_entropies


# Compute entropies via forward pass.
print("Computing per-token prediction entropies...")
device = next(model.parameters()).device
all_entropies = compute_token_entropies(model, ds, device)

# Compute threshold from the specified quantile across all tokens.
all_entropy_values = torch.cat(all_entropies)
threshold = torch.quantile(all_entropy_values, ENTROPY_QUANTILE).item()

# Print entropy statistics.
print(f"\nEntropy Statistics:")
print(f"  Min:       {all_entropy_values.min().item():.4f}")
print(f"  Max:       {all_entropy_values.max().item():.4f}")
print(f"  Mean:      {all_entropy_values.mean().item():.4f}")
print(f"  Threshold: {threshold:.4f} (quantile={ENTROPY_QUANTILE})")
num_selected = (all_entropy_values >= threshold).sum().item()
print(
    f"  Selected:  {num_selected}/{len(all_entropy_values)} tokens "
    f"({100 * num_selected / len(all_entropy_values):.1f}%)"
)

# Visualize high-entropy tokens for a few samples.
print(f"\n{'='*60}")
print("HIGH-ENTROPY TOKEN VISUALIZATION")
print(f"{'='*60}")

for sample_idx in range(min(NUM_DISPLAY_SAMPLES, len(ds))):
    input_ids = ds[sample_idx]["input_ids"]
    entropies = all_entropies[sample_idx]
    mask = (entropies >= threshold).long()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    num_masked = mask.sum().item()

    print(f"\n--- Sample {sample_idx} ({num_masked}/{len(tokens)} tokens selected) ---")
    for i, (tok, ent, m) in enumerate(zip(tokens, entropies.tolist(), mask.tolist())):
        marker = "[*]" if m else "   "
        print(f"  {marker} {i:4d} | {ent:6.3f} | {tok}")

# Add loss_mask to each sample based on entropy threshold.
entropy_lookup = {i: ent for i, ent in enumerate(all_entropies)}


def add_entropy_mask(sample, idx):
    entropies = entropy_lookup[idx]
    mask = (entropies >= threshold).long()
    sample["loss_mask"] = mask
    return sample


ds = ds.map(add_entropy_mask, with_indices=True)

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
        duo_scaling="both",
    ),
]

# Apply algorithms with entropy-based token masking.
print("\nRunning AWQ with entropy-based masking...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    use_loss_mask=True,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-asym-entropy-masked"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
