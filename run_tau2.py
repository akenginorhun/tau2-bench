#!/usr/bin/env python3
"""
Utility script for running tau2 simulations against a vLLM server.

You can now pre-initialize the vLLM server once via ``initialize_vllm.py`` and
reuse its API endpoint for multiple benchmark runs by pointing ``run_tau2.py``
at the desired ``--api-base`` (or ``--client-host``/``--vllm-port`` combo).

Example:
    python initialize_vllm.py --model meta-llama/Llama-3.1-8B-Instruct
    python run_tau2.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --domains airline retail \
        --num-trials 1 \
        --num-tasks 5 \
        --api-base http://127.0.0.1:8008/v1
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    
    # vLLM hack: force tool_choice=None to avoid "auto" errors if server doesn't support it.
    # The model (Qwen + hermes parser) will still produce tool calls naturally.
    agent_llm_args["tool_choice"] = None

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
        # if "tool_choice" not in user_llm_args:
        #     user_llm_args["tool_choice"] = "auto"
    else:
        user_llm = args.user_llm
        user_llm_args = deepcopy(DEFAULT_LLM_ARGS_USER)
        if args.user_arg:
            user_llm_args.update(parse_kv_pairs(args.user_arg))
        if "custom_llm_provider" not in user_llm_args:
            user_llm_args["custom_llm_provider"] = "openai"
        # if "tool_choice" not in user_llm_args:
        #     user_llm_args["tool_choice"] = "auto"

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
        "--vllm-port",
        type=int,
        default=8008,
        help="Port for the vLLM OpenAI API server.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed agent behavior tracing (sets log level to DEBUG).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Suppress LiteLLM cost mapping warnings - must be set before any imports
    import os
    os.environ["LITELLM_LOG"] = "CRITICAL"
    
    # Also suppress standard logging
    import logging
    logging.getLogger("litellm").setLevel(logging.CRITICAL)
    
    api_base = args.api_base or f"http://{args.client_host}:{args.vllm_port}/v1"
    
    # Override log level for detailed tracing if requested
    if args.verbose:
        args.log_level = "DEBUG"
        
    # Reconfigure logger to filter out the specific cost mapping error
    # This is required because tau2.utils.llm_utils catches the exception and explicitly logs it
    def filter_cost_errors(record):
        return "This model isn't mapped yet" not in record["message"]
        
    logger.remove()
    # We use sys.stderr by default like standard loguru, but with our filter
    import sys
    logger.add(sys.stderr, level=args.log_level, filter=filter_cost_errors)
    
    logger.info("Using vLLM endpoint at %s", api_base)
    
    run_tau2_simulations(args, api_base=api_base)


if __name__ == "__main__":
    main()

