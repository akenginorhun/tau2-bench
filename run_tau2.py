#!/usr/bin/env python3
"""
Utility script for running tau2 simulations against a locally hosted vLLM model.

Example:
    python run_tau2.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --domains airline retail \
        --num-trials 1 \
        --num-tasks 5
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
from copy import deepcopy
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.config import (
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain


def parse_kv_pairs(pairs: list[str]) -> dict:
    """
    Parse command-line key=value pairs into a dictionary with basic type coercion.
    """

    def _convert(value: str):
        value = value.strip()
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    parsed: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid argument '{pair}'. Expected key=value format.")
        key, value = pair.split("=", 1)
        parsed[key.strip()] = _convert(value)
    return parsed


def wait_for_vllm(
    base_url: str,
    process: Optional[subprocess.Popen],
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
        if process is not None and process.poll() is not None:
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
            # 401 / 403 indicate server is up but requires auth.
            if exc.code in {401, 403, 404}:
                logger.info("vLLM server is ready (received HTTP %s).", exc.code)
                return
            last_error = f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        time.sleep(1)

    error_suffix = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for vLLM server to start{error_suffix}.")


def launch_vllm(args: argparse.Namespace) -> tuple[Optional[subprocess.Popen], str]:
    """
    Launch (or validate) the vLLM OpenAI-compatible server.
    Returns the process handle (if we started one) and the api_base URL.
    """
    api_base = args.api_base
    if api_base is None:
        host_for_client = args.client_host or args.vllm_host
        api_base = f"http://{host_for_client}:{args.vllm_port}/v1"

    if args.reuse_vllm:
        logger.info("Reusing existing vLLM server at %s", api_base)
        wait_for_vllm(
            api_base,
            process=None,
            timeout=args.startup_timeout,
            api_key=args.api_key,
        )
        return None, api_base

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

    env = os.environ.copy()
    if args.vllm_env:
        for pair in args.vllm_env:
            if "=" not in pair:
                raise ValueError(
                    f"Invalid --vllm-env entry '{pair}'. Expected KEY=value."
                )
            key, value = pair.split("=", 1)
            env[key.strip()] = value

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


def build_llm_args(
    defaults: dict,
    api_base: str,
    api_key: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra_args: Optional[list[str]],
) -> dict:
    """
    Prepare LLM argument dictionaries for tau2.
    """
    llm_args = deepcopy(defaults)
    llm_args.update({"api_base": api_base, "api_key": api_key})
    if temperature is not None:
        llm_args["temperature"] = temperature
    if max_tokens is not None:
        llm_args["max_tokens"] = max_tokens
    if extra_args:
        llm_args.update(parse_kv_pairs(extra_args))
    return llm_args


def run_tau2_simulations(args: argparse.Namespace, api_base: str) -> None:
    """
    Execute tau2 simulations for the requested domains using the local vLLM model.
    """
    # Build agent LLM args and apply safe defaults for provider/tool choice if not set
    agent_llm_args = build_llm_args(
        defaults=DEFAULT_LLM_ARGS_AGENT,
        api_base=api_base,
        api_key=args.api_key,
        temperature=args.agent_temperature,
        max_tokens=args.agent_max_tokens,
        extra_args=args.agent_arg,
    )
    if "custom_llm_provider" not in agent_llm_args:
        agent_llm_args["custom_llm_provider"] = "openai"
    if "tool_choice" not in agent_llm_args:
        agent_llm_args["tool_choice"] = "required"

    if args.user_llm is None:
        user_llm = args.model
        # Build user LLM args and apply defaults (no tools requested by user)
        user_llm_args = build_llm_args(
            defaults=DEFAULT_LLM_ARGS_USER,
            api_base=api_base,
            api_key=args.api_key,
            temperature=args.user_temperature,
            max_tokens=args.user_max_tokens,
            extra_args=args.user_arg,
        )
        if "custom_llm_provider" not in user_llm_args:
            user_llm_args["custom_llm_provider"] = "openai"
        if "tool_choice" not in user_llm_args:
            user_llm_args["tool_choice"] = "none"
    else:
        user_llm = args.user_llm
        user_llm_args = deepcopy(DEFAULT_LLM_ARGS_USER)
        if args.user_arg:
            user_llm_args.update(parse_kv_pairs(args.user_arg))
        if "custom_llm_provider" not in user_llm_args:
            user_llm_args["custom_llm_provider"] = "openai"
        if "tool_choice" not in user_llm_args:
            user_llm_args["tool_choice"] = "none"

    for domain in args.domains:
        logger.info("Running tau2 domain '%s' with model '%s'", domain, args.model)
        config = RunConfig(
            domain=domain,
            task_set_name=args.task_set,
            task_split_name=args.task_split,
            task_ids=args.task_ids if args.task_ids else None,
            num_tasks=args.num_tasks,
            agent=args.agent,
            llm_agent=args.model,
            llm_args_agent=agent_llm_args,
            user=args.user,
            llm_user=user_llm,
            llm_args_user=user_llm_args,
            num_trials=args.num_trials,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            save_to=args.save_to,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
            log_level=args.log_level,
            enforce_communication_protocol=args.enforce_protocol,
        )
        run_domain(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a vLLM server and route tau2 simulations to it."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name to load with vLLM (used as the tau2 agent LLM).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["airline"],
        help="Tau2 domains to evaluate.",
    )
    parser.add_argument(
        "--task-set",
        default=None,
        help="Optional override for the task set name. Defaults to the domain name.",
    )
    parser.add_argument(
        "--task-split",
        default="base",
        help="Task split to evaluate (default: base).",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Specific task ID to run. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Limit the number of tasks per domain.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Number of trials per task.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum turns per simulation.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help="Maximum consecutive tool errors allowed.",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="Optional base name for the simulation output JSON.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum number of concurrent simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for sampling tasks.",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        help="Override log level (default from tau2 config).",
    )
    parser.add_argument(
        "--agent",
        default="llm_agent",
        help="Tau2 agent implementation.",
    )
    parser.add_argument(
        "--user",
        default="user_simulator",
        help="Tau2 user implementation.",
    )
    parser.add_argument(
        "--user-llm",
        default=None,
        help="Use a different LLM for the user simulator. Defaults to the vLLM model.",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=None,
        help="Optional temperature override for the agent.",
    )
    parser.add_argument(
        "--user-temperature",
        type=float,
        default=None,
        help="Optional temperature override for the user.",
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override for the agent.",
    )
    parser.add_argument(
        "--user-max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override for the user.",
    )
    parser.add_argument(
        "--agent-arg",
        action="append",
        help="Additional key=value overrides for agent llm_args.",
    )
    parser.add_argument(
        "--user-arg",
        action="append",
        help="Additional key=value overrides for user llm_args.",
    )
    parser.add_argument(
        "--enforce-protocol",
        action="store_true",
        help="Enable communication protocol enforcement.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key passed to the vLLM endpoint (default: EMPTY).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Explicit API base for the vLLM server (e.g. http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--client-host",
        default="127.0.0.1",
        help="Host used by the client when constructing the API base (defaults to 127.0.0.1).",
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
        "--reuse-vllm",
        action="store_true",
        help="Skip launching vLLM and reuse an already running server.",
    )
    parser.add_argument(
        "--keep-vllm",
        action="store_true",
        help="Keep the vLLM server running after simulations finish.",
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
        if args.keep_vllm:
            logger.info("Keeping vLLM server alive (per --keep-vllm).")
            return
        logger.info("Shutting down vLLM server...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing vLLM server after timeout.")
            vllm_process.kill()

    def _handle_signal(signum, frame):
        _cleanup(signum, frame)
        sys.exit(0)

    try:
        vllm_process, api_base = launch_vllm(args)
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        run_tau2_simulations(args, api_base=api_base)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()

