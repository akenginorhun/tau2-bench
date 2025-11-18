#!/usr/bin/env python3
"""
Utility script for initializing (and keeping alive) a vLLM OpenAI-compatible
server that can be shared across multiple tau2 benchmark runs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from loguru import logger


def wait_for_vllm(
    base_url: str,
    process: subprocess.Popen,
    timeout: int,
    api_key: Optional[str] = None,
) -> None:
    """
    Wait until the vLLM OpenAI-compatible server is ready.
    """
    deadline = time.time() + timeout
    status_url = f"{base_url.rstrip('/')}/models"
    last_error: Optional[str] = None

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "vLLM server process exited prematurely with return code "
                f"{process.returncode}."
            )
        try:
            request = urllib.request.Request(status_url)
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request) as response:
                if response.status == 200:
                    logger.info("vLLM server is ready.")
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 404}:
                logger.info("vLLM server responded with HTTP %s, assuming ready.", exc.code)
                return
            last_error = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        time.sleep(1)

    error_suffix = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for vLLM server to start{error_suffix}.")


def parse_env_overrides(overrides: Optional[list[str]]) -> dict[str, str]:
    env_updates: dict[str, str] = {}
    if not overrides:
        return env_updates
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Invalid --vllm-env entry '{entry}'. Expected KEY=value.")
        key, value = entry.split("=", 1)
        env_updates[key.strip()] = value
    return env_updates


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.vllm_host,
        "--port",
        str(args.vllm_port),
    ]

    if args.download_dir is not None:
        command.extend(["--download-dir", args.download_dir])
    if args.tensor_parallel_size is not None:
        command.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.dtype is not None:
        command.extend(["--dtype", args.dtype])
    if args.vllm_extra_arg:
        for extra in args.vllm_extra_arg:
            command.extend(shlex.split(extra))

    return command


def launch_vllm(args: argparse.Namespace) -> tuple[subprocess.Popen, str]:
    api_base = args.api_base or f"http://{args.client_host}:{args.vllm_port}/v1"
    command = build_vllm_command(args)

    env = os.environ.copy()
    env.update(parse_env_overrides(args.vllm_env))

    logger.info("Starting vLLM server:\n%s", " ".join(shlex.quote(c) for c in command))
    process = subprocess.Popen(command, env=env)

    try:
        wait_for_vllm(
            api_base,
            process=process,
            timeout=args.startup_timeout,
            api_key=args.api_key,
        )
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise

    return process, api_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a reusable vLLM OpenAI-compatible server for tau2 benchmarks."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name to load with vLLM.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key forwarded to readiness checks and exposed endpoint (default: EMPTY).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Explicit API base for the vLLM server (e.g. http://127.0.0.1:8008/v1).",
    )
    parser.add_argument(
        "--client-host",
        default="127.0.0.1",
        help="Host used by clients when constructing the API base (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--vllm-host",
        default="0.0.0.0",
        help="Host interface for serving the vLLM OpenAI API.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8008,
        help="Port for the vLLM OpenAI API server.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional model download/cache directory for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of tensor parallel partitions.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max_model_len.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory to allocate for the model.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="dtype flag forwarded to vLLM (e.g. auto, float16, bfloat16).",
    )
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        help="Extra argument string forwarded to vLLM (quotes required).",
    )
    parser.add_argument(
        "--vllm-env",
        action="append",
        help="Environment variable overrides for the vLLM process (KEY=value).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180,
        help="Seconds to wait for vLLM server readiness.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vllm_process: Optional[subprocess.Popen] = None

    def _cleanup(signum=None, frame=None) -> None:  # type: ignore[override]
        if vllm_process is None:
            return
        logger.info("Shutting down vLLM server...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing vLLM server after timeout.")
            vllm_process.kill()

    def _handle_signal(signum, frame):
        logger.info("Received signal %s. Cleaning up vLLM server...", signum)
        _cleanup(signum, frame)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    vllm_process, api_base = launch_vllm(args)
    logger.success("vLLM server is ready at %s", api_base)
    logger.info(
        "Keep this process running to reuse the server. "
        "In another terminal run tau2 via: python run_tau2.py --model %s --api-base %s ...",
        args.model,
        api_base,
    )

    try:
        vllm_process.wait()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()

