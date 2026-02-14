"""
Online serving benchmark for RTN-quantized MoE checkpoints via vLLM in Docker.

For each quantized checkpoint (nvfp4a16, mxfp4a16, int4a16), the script:
  1. Starts a vLLM server inside a Docker container (v0.15.0)
  2. Waits for the server to become healthy
  3. Runs `vllm bench serve` on the host for every (input_len, output_len, concurrency)
  4. Tears down the container and moves to the next checkpoint

Usage:
    python bench_rtn_moe.py --ckpt-dir /path/to/checkpoints
    python bench_rtn_moe.py --ckpt-dir /ckpts --backend b200 --tp 4 \
        --concurrencies 1 8 32 --input-lens 1000 10000 --output-lens 100 1000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONTAINER_PREFIX = "vllm-bench"

# ── Quantization tags to benchmark ────────────────────────────────────────────

QUANT_TAGS = ["nvfp4a16", "mxfp4a16", "int4a16"]

# ── Backend presets ───────────────────────────────────────────────────────────
# Each preset defines:
#   env          – environment variables passed to the Docker container
#   engine_flags – extra flags for `vllm serve` (underscores → hyphens)
#                  bool True  → bare flag (--enable-chunked-prefill)
#                  bool False → omitted
#                  other      → flag + value (--max-num-batched-tokens 8192)
#   setup_cmds   – shell commands to run inside the container before vllm serve
#                  (e.g. installing CUDA compat libraries)

BACKEND_PRESETS: dict[str, dict] = {
    "b200": {
        "image": "vllm/vllm-openai:v0.15.0",
        "gpu_mode": "nvidia_cdi",          # --device=nvidia.com/gpu=N per GPU
        "docker_flags": ["--runtime", "nvidia", "--ipc=host"],
        "env": {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        },
        "engine_flags": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
        },
        "setup_cmds": [
            "apt-get update -qq && apt-get install -y -qq cuda-compat-13-0 > /dev/null 2>&1",
            "export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}",
        ],
    },
    "h100": {
        "image": "vllm/vllm-openai:v0.15.0",
        "gpu_mode": "nvidia_cdi",
        "docker_flags": ["--ipc=host"],
        "env": {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        },
        "engine_flags": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
        },
        "setup_cmds": [],
    },
    "mi300x": {
        "image": "vllm/vllm-openai:v0.15.0-rocm",
        "gpu_mode": "rocm",                # GPUs exposed via /dev/kfd + /dev/dri
        "docker_flags": [
            "--privileged",
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--shm-size=500g",
            "--security-opt", "seccomp=unconfined",
        ],
        "env": {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MHA": "1",
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0",
        },
        "engine_flags": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
        },
        "setup_cmds": [],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> str:
    """Run a command and optionally return its stdout."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def engine_flags_to_args(flags: dict) -> list[str]:
    """Convert an engine_flags dict to CLI arguments for `vllm serve`."""
    args: list[str] = []
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


# ── Server lifecycle ──────────────────────────────────────────────────────────


def build_server_cmd(
    preset: dict,
    ckpt_dir: str,
    ckpt_name: str,
    tp: int,
    port: int,
    max_model_len: int | None,
    gpus: str,
    container_name: str,
) -> list[str]:
    """Build the `docker run` command for the vLLM server."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    image = preset["image"]

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{ckpt_dir}:/models:ro",
        "-v", f"{hf_home}:/root/.cache/huggingface",
        "-p", f"{port}:{port}",
    ]

    # Backend-specific Docker flags (runtime, device passthrough, shm, etc.)
    cmd.extend(preset.get("docker_flags", []))

    # GPU device passthrough — only needed for NVIDIA CDI mode;
    # ROCm exposes all GPUs via /dev/kfd + /dev/dri (already in docker_flags).
    if preset.get("gpu_mode") == "nvidia_cdi":
        for gpu_id in gpus.split(","):
            cmd.extend(["--device", f"nvidia.com/gpu={gpu_id.strip()}"])

    # Inject env vars from the backend preset
    for key, value in preset.get("env", {}).items():
        cmd.extend(["-e", f"{key}={value}"])

    # Build the vllm serve arguments
    serve_args = [
        f"/models/{ckpt_name}",
        "-tp", str(tp),
        "--port", str(port),
    ]
    serve_args.extend(engine_flags_to_args(preset.get("engine_flags", {})))
    if max_model_len is not None:
        serve_args.extend(["--max-model-len", str(max_model_len)])

    setup_cmds = preset.get("setup_cmds", [])
    if setup_cmds:
        # Override entrypoint to run setup commands before vllm serve
        cmd.extend(["--entrypoint", "bash"])
        cmd.append(image)
        setup_script = " && ".join(setup_cmds)
        serve_cmd = "vllm serve " + " ".join(serve_args)
        cmd.extend(["-c", f"{setup_script} && {serve_cmd}"])
    else:
        # The vllm/vllm-openai image has ENTRYPOINT ["vllm", "serve"],
        # so we only pass the model path and flags.
        cmd.append(image)
        cmd.extend(serve_args)

    return cmd


def start_server(cmd: list[str], container_name: str) -> None:
    """Start the vLLM server container."""
    # Make sure no stale container exists
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"  ERROR: docker run failed (exit {result.returncode})")
        if result.stdout.strip():
            print(f"  stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()}")
        sys.exit(1)
    print(f"  Container started: {result.stdout.strip()[:12]}")


def wait_for_health(
    port: int, container_name: str, timeout: int = 900, interval: int = 10,
) -> None:
    """Poll the /health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    print(f"  Waiting for server at {url} (timeout {timeout}s) ...")

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print("  Server is healthy.")
                    return
        except Exception:
            pass
        time.sleep(interval)

    # If we get here, the server never became healthy — dump logs for debugging
    print("  ERROR: Server did not become healthy. Container logs:")
    subprocess.run(["docker", "logs", "--tail", "80", container_name], check=False)
    stop_server(container_name)
    sys.exit(1)


def stop_server(container_name: str) -> None:
    """Stop and remove the container."""
    print(f"  Stopping container {container_name} ...")
    subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
    subprocess.run(["docker", "rm", container_name], capture_output=True, check=False)


# ── Benchmark runner ──────────────────────────────────────────────────────────


def run_bench(
    served_model_name: str,
    tokenizer_path: str,
    port: int,
    in_len: int,
    out_len: int,
    concurrency: int,
    num_prompts: int,
    result_dir: str,
    result_filename: str,
) -> None:
    """Run a single `vllm bench serve` invocation on the host."""
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        "vllm", "bench", "serve",
        "--model", served_model_name,
        "--tokenizer", tokenizer_path,
        "--backend", "openai",
        "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--ignore-eos",
        "--max-concurrency", str(concurrency),
        "--num-prompts", str(num_prompts),
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
        "--gpu-memory-utilization", "0.95",
    ]

    print(f"\n{'=' * 70}")
    print(f"  Bench: in={in_len} out={out_len} concurrency={concurrency}")
    print(f"  Results → {result_dir}/{result_filename}")
    print(f"{'=' * 70}")
    run(cmd, check=False)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online vLLM benchmark for RTN-quantized MoE checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Parent directory containing the quantized checkpoint folders",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="b200",
        choices=list(BACKEND_PRESETS.keys()),
        help="Hardware backend preset (selects env vars and engine flags)",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 4, 32],
        help="List of --max-concurrency values to sweep",
    )
    parser.add_argument(
        "--input-lens",
        type=int,
        nargs="+",
        default=[1000, 10000, 80000],
        help="List of input lengths (paired 1:1 with --output-lens)",
    )
    parser.add_argument(
        "--output-lens",
        type=int,
        nargs="+",
        default=[100, 1000, 8000],
        help="List of output lengths (paired 1:1 with --input-lens)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bench_results",
        help="Directory to store JSON result files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3-235B-A22B",
        help="Base model name (used to construct checkpoint folder names)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the vLLM server",
    )
    parser.add_argument(
        "--num-prompts-factor",
        type=int,
        default=10,
        help="num_prompts = concurrency * this factor",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the model's default max context length",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=(
            "Comma-separated GPU IDs to expose to the Docker container "
            "via --device=nvidia.com/gpu=N. Examples: '0', '0,1', '2,3'"
        ),
    )
    parser.add_argument(
        "--quant-tags",
        type=str,
        nargs="+",
        default=QUANT_TAGS,
        help="Quantization tags to benchmark",
    )

    args = parser.parse_args()

    if len(args.input_lens) != len(args.output_lens):
        parser.error(
            f"--input-lens ({len(args.input_lens)} items) and "
            f"--output-lens ({len(args.output_lens)} items) must have the same length"
        )

    return args


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    preset = BACKEND_PRESETS[args.backend]
    ckpt_dir = os.path.abspath(args.ckpt_dir)

    # Print configuration
    print("=" * 70)
    print("  vLLM Online Benchmark")
    print("=" * 70)
    print(f"  Backend       : {args.backend}")
    print(f"  GPUs          : {args.gpus}")
    print(f"  TP            : {args.tp}")
    print(f"  Checkpoints   : {ckpt_dir}")
    print(f"  Quant tags    : {args.quant_tags}")
    print(f"  Concurrencies : {args.concurrencies}")
    print(f"  Length pairs  : {list(zip(args.input_lens, args.output_lens))}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Docker image  : {preset['image']}")
    print(f"  Env vars      : {preset.get('env', {})}")
    print(f"  Engine flags  : {preset.get('engine_flags', {})}")
    print("=" * 70)

    # Validate checkpoint directories
    for tag in args.quant_tags:
        ckpt_name = f"{args.model_name}-{tag}-rtn-moe"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.isdir(ckpt_path):
            print(f"  WARNING: checkpoint not found: {ckpt_path}")

    # Pull Docker image
    print(f"\nPulling {preset['image']} ...")
    run(["docker", "pull", preset["image"]], check=False)

    # Run benchmarks for each checkpoint
    results: list[str] = []

    for tag in args.quant_tags:
        ckpt_name = f"{args.model_name}-{tag}-rtn-moe"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        result_subdir = os.path.join(args.output_dir, f"{tag}-rtn-moe-tp{args.tp}")
        container_name = f"{CONTAINER_PREFIX}-{tag}"
        os.makedirs(result_subdir, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"  Checkpoint: {ckpt_name}")
        print(f"  Container : {container_name}")
        print(f"{'#' * 70}")

        if not os.path.isdir(ckpt_path):
            print(f"  SKIP: {ckpt_path} does not exist")
            continue

        # Start server
        server_cmd = build_server_cmd(
            preset=preset,
            ckpt_dir=ckpt_dir,
            ckpt_name=ckpt_name,
            tp=args.tp,
            port=args.port,
            max_model_len=args.max_model_len,
            gpus=args.gpus,
            container_name=container_name,
        )
        start_server(server_cmd, container_name)
        wait_for_health(args.port, container_name)

        try:
            for in_len, out_len in zip(args.input_lens, args.output_lens):
                for conc in args.concurrencies:
                    num_prompts = conc * args.num_prompts_factor
                    filename = f"in{in_len}_out{out_len}_c{conc}.json"

                    run_bench(
                        served_model_name=f"/models/{ckpt_name}",
                        tokenizer_path=ckpt_path,
                        port=args.port,
                        in_len=in_len,
                        out_len=out_len,
                        concurrency=conc,
                        num_prompts=num_prompts,
                        result_dir=result_subdir,
                        result_filename=filename,
                    )
                    results.append(os.path.join(result_subdir, filename))
        finally:
            stop_server(container_name)

    # Summary
    print(f"\n{'=' * 70}")
    print("  Benchmark complete. Result files:")
    print(f"{'=' * 70}")
    for path in results:
        exists = "OK" if os.path.isfile(path) else "MISSING"
        print(f"  [{exists}] {path}")


if __name__ == "__main__":
    main()
