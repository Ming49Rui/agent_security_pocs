#!/usr/bin/env python3
"""Run the OpenClaw benchmark sequentially with explicit Agent and Judge APIs."""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import create_openclaw_profile


ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).with_name("eval_openclaw_t1_local.py")
DEFAULT_OPENCLAW_CLI = (
    ROOT / "openclaw-2026.4.22" / "node_modules" / ".bin" / "openclaw"
)
THREATS = ("t1", "t3", "t5", "t6", "t7")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential OpenClaw evaluator. It creates one isolated profile and "
            "Gateway, then evaluates selected tasks one at a time."
        )
    )
    parser.add_argument("--threat", required=True, choices=THREATS)
    parser.add_argument("--all", action="store_true", help="run every discovered task")
    parser.add_argument("--scenario", help="run or filter one scenario")
    parser.add_argument("--persona-id", help="run one persona; requires --scenario")
    parser.add_argument(
        "--chain-variant",
        default="3_skills_chain",
        choices=("2_skills_chain", "3_skills_chain"),
    )
    parser.add_argument(
        "--data-type",
        default="poisoned",
        choices=("clean", "poisoned"),
    )
    parser.add_argument(
        "--tasks-root",
        default="",
        help=(
            "task-suite root for the selected threat; it must contain common/ "
            "and variants/. Omit only when using the bundled default layout."
        ),
    )

    agent = parser.add_argument_group("Agent API")
    agent.add_argument(
        "--agent-api-base",
        default=os.getenv("AGENT_API_BASE", ""),
        help="Agent provider API base URL",
    )
    agent.add_argument(
        "--agent-api-key",
        default=os.getenv("AGENT_API_KEY", ""),
        help="Agent API key; AGENT_API_KEY is preferred to avoid shell history",
    )
    agent.add_argument(
        "--agent-model",
        default=os.getenv("AGENT_MODEL", ""),
        help="Agent model ID declared in the generated OpenClaw profile",
    )
    agent.add_argument("--agent-model-name", default="")
    agent.add_argument("--agent-provider-id", default="custom")
    agent.add_argument(
        "--agent-api-adapter",
        choices=create_openclaw_profile.OPENCLAW_API_ADAPTERS,
        default="openai-completions",
    )

    judge = parser.add_argument_group("Judge API")
    judge.add_argument(
        "--judge-api-base",
        default=os.getenv("JUDGE_API_BASE", ""),
        help="OpenAI-compatible Judge API base URL",
    )
    judge.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_API_KEY", ""),
        help="Judge API key; JUDGE_API_KEY is preferred to avoid shell history",
    )
    judge.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", ""),
        help="Judge model ID",
    )

    runtime = parser.add_argument_group("OpenClaw runtime")
    runtime.add_argument("--openclaw-cli", default=str(DEFAULT_OPENCLAW_CLI))
    runtime.add_argument(
        "--openclaw-profile",
        default="",
        help="generated profile name; omitted creates a unique temporary profile",
    )
    runtime.add_argument(
        "--gateway-port",
        type=int,
        default=0,
        help="local Gateway port; 0 chooses an available port",
    )
    runtime.add_argument("--gateway-startup-timeout", type=float, default=30.0)
    runtime.add_argument("--keep-profile", action="store_true")
    runtime.add_argument("--thinking", default="off")
    runtime.add_argument("--interaction-turns", type=int, default=2)
    runtime.add_argument("--timeout-seconds", type=int, default=1120)
    runtime.add_argument("--max-steps", type=int, default=30)
    runtime.add_argument(
        "--stop-after-consecutive-agent-timeouts",
        type=int,
        default=1,
    )

    output = parser.add_argument_group("Result control")
    output.add_argument("--results-root")
    output.add_argument("--results-model-slug", default="openclaw_2026.4.22")
    output.add_argument("--result-run-id", default="")
    output.add_argument("--resume", action="store_true")
    output.add_argument("--rerun-failed", action="store_true")
    output.add_argument("--stop-on-attack-success", action="store_true")
    output.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _absolute_http_url(label: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute http(s) URL")


def validate_args(args: argparse.Namespace) -> None:
    if args.all and args.persona_id:
        raise ValueError("--all cannot be combined with --persona-id")
    if not args.all and (not args.scenario or not args.persona_id):
        raise ValueError(
            "select --all, or provide both --scenario and --persona-id"
        )
    if args.gateway_port < 0 or args.gateway_port > 65535:
        raise ValueError("--gateway-port must be 0 or between 1 and 65535")
    if args.gateway_startup_timeout <= 0:
        raise ValueError("--gateway-startup-timeout must be positive")
    if args.timeout_seconds <= 0 or args.max_steps <= 0:
        raise ValueError("timeout and max steps must be positive")
    if args.stop_after_consecutive_agent_timeouts < 0:
        raise ValueError("--stop-after-consecutive-agent-timeouts must be >= 0")
    if args.tasks_root:
        tasks_root = Path(args.tasks_root).expanduser().resolve()
        if not tasks_root.is_dir():
            raise ValueError(f"--tasks-root does not exist: {tasks_root}")
        if not (tasks_root / "common").is_dir():
            raise ValueError(f"--tasks-root is missing common/: {tasks_root}")
        if not (tasks_root / "variants" / args.chain_variant).is_dir():
            raise ValueError(
                f"--tasks-root is missing variants/{args.chain_variant}/: "
                f"{tasks_root}"
            )
        args.tasks_root = str(tasks_root)

    if args.dry_run:
        return

    required = {
        "--agent-api-base": args.agent_api_base,
        "--agent-api-key or AGENT_API_KEY": args.agent_api_key,
        "--agent-model": args.agent_model,
        "--judge-api-base": args.judge_api_base,
        "--judge-api-key or JUDGE_API_KEY": args.judge_api_key,
        "--judge-model": args.judge_model,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"missing required API settings: {', '.join(missing)}")
    _absolute_http_url("Agent API base", args.agent_api_base)
    _absolute_http_url("Judge API base", args.judge_api_base)


def _cli_prefix(openclaw_cli: str) -> list[str]:
    command = shlex.split(openclaw_cli)
    if not command:
        raise ValueError("OpenClaw CLI command is empty")
    executable = command[0]
    if not Path(executable).is_file() and shutil.which(executable) is None:
        raise FileNotFoundError(f"OpenClaw CLI not found: {executable}")
    return command


def choose_gateway_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def generated_profile_name() -> str:
    return f"msc-openclaw-seq-{os.getpid()}-{secrets.token_hex(4)}"


def build_child_environment(
    args: argparse.Namespace,
    *,
    profile_name: str,
) -> dict[str, str]:
    env = os.environ.copy()
    if args.agent_api_key:
        env["AGENT_API_KEY"] = args.agent_api_key
    if args.judge_api_key:
        env["JUDGE_API_KEY"] = args.judge_api_key
    env["OPENCLAW_PROFILE"] = profile_name
    env.pop("OPENCLAW_GATEWAY_TOKEN", None)
    env.pop("OPENCLAW_GATEWAY_PORT", None)
    return env


def create_runtime_profile(
    args: argparse.Namespace,
    *,
    profile_name: str,
    gateway_port: int,
) -> Path:
    model_name = args.agent_model_name.strip() or args.agent_model
    create_openclaw_profile.validate_inputs(
        profile=profile_name,
        provider_id=args.agent_provider_id,
        api_base=args.agent_api_base,
        model_id=args.agent_model,
        model_name=model_name,
        api_key_env="AGENT_API_KEY",
        gateway_port=gateway_port,
    )
    config = create_openclaw_profile.build_config(
        provider_id=args.agent_provider_id,
        api_base=args.agent_api_base,
        model_id=args.agent_model,
        model_name=model_name,
        api_adapter=args.agent_api_adapter,
        api_key_env="AGENT_API_KEY",
        gateway_port=gateway_port,
        gateway_token=secrets.token_urlsafe(32),
    )
    config_path = create_openclaw_profile.profile_config_path(profile_name)
    if config_path.parent.exists():
        raise FileExistsError(
            f"refusing to use existing OpenClaw profile directory: "
            f"{config_path.parent}"
        )
    create_openclaw_profile.write_profile(config_path, config, force=False)
    return config_path


def build_harness_command(
    args: argparse.Namespace,
    *,
    profile_name: str,
    gateway_port: int,
) -> list[str]:
    model_id = args.agent_model or "dry-run-model"
    command = [
        sys.executable,
        str(HARNESS),
        "--threat",
        args.threat,
        "--chain-variant",
        args.chain_variant,
        "--data-type",
        args.data_type,
        "--interaction-turns",
        str(args.interaction_turns),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-steps",
        str(args.max_steps),
        "--thinking",
        args.thinking,
        "--openclaw-cli",
        args.openclaw_cli,
        "--openclaw-profile",
        profile_name,
        "--agent-mode",
        "gateway",
        "--gateway-port",
        str(gateway_port),
        "--judge-api-base",
        args.judge_api_base,
        "--judge-model",
        args.judge_model,
        "--results-model-slug",
        args.results_model_slug,
        "--stop-after-consecutive-agent-timeouts",
        str(args.stop_after_consecutive_agent_timeouts),
    ]
    if not args.dry_run:
        command.extend(
            ["--agent-model", f"{args.agent_provider_id}/{model_id}"]
        )
    if args.all:
        command.append("--all")
        if args.scenario:
            command.extend(["--scenario", args.scenario])
    else:
        command.extend(
            ["--scenario", args.scenario, "--persona-id", args.persona_id]
        )
    if args.tasks_root:
        command.extend(["--tasks-root", args.tasks_root])
    if args.results_root:
        command.extend(["--results-root", args.results_root])
    if args.result_run_id:
        command.extend(["--result-run-id", args.result_run_id])
    if args.resume:
        command.append("--resume")
    if args.rerun_failed:
        command.append("--rerun-failed")
    if args.stop_on_attack_success:
        command.append("--stop-on-attack-success")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _wait_for_gateway(
    process: subprocess.Popen[str],
    *,
    port: int,
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
            raise RuntimeError(
                f"Gateway exited with code {process.returncode}: {detail}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(
        f"Gateway did not become ready on port {port}; log={log_path}"
    )


@contextmanager
def running_gateway(
    args: argparse.Namespace,
    *,
    profile_name: str,
    gateway_port: int,
    env: dict[str, str],
    log_path: Path,
) -> Iterator[None]:
    command = [
        *_cli_prefix(args.openclaw_cli),
        "--profile",
        profile_name,
        "gateway",
        "run",
        "--port",
        str(gateway_port),
        "--verbose",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            _wait_for_gateway(
                process,
                port=gateway_port,
                timeout=args.gateway_startup_timeout,
                log_path=log_path,
            )
            yield
        finally:
            if process.poll() is None:
                with suppress(Exception):
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(Exception):
                        process.kill()
                    with suppress(Exception):
                        process.wait(timeout=5)


def validate_profile(
    args: argparse.Namespace,
    *,
    profile_name: str,
    env: dict[str, str],
) -> None:
    command = [
        *_cli_prefix(args.openclaw_cli),
        "--profile",
        profile_name,
        "config",
        "validate",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"OpenClaw profile validation failed: {detail}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"[sequential] ERROR: {error}", file=sys.stderr, flush=True)
        return 2

    profile_name = args.openclaw_profile.strip() or generated_profile_name()
    gateway_port = args.gateway_port or (
        18789 if args.dry_run else choose_gateway_port()
    )
    env = build_child_environment(args, profile_name=profile_name)
    harness_command = build_harness_command(
        args,
        profile_name=profile_name,
        gateway_port=gateway_port,
    )

    if args.dry_run:
        print("[sequential] dry-run; no profile, Gateway, or API call", flush=True)
        return subprocess.run(harness_command, env=env).returncode

    profile_dir = create_openclaw_profile.profile_config_path(
        profile_name
    ).parent
    profile_created = False
    try:
        config_path = create_runtime_profile(
            args,
            profile_name=profile_name,
            gateway_port=gateway_port,
        )
        profile_created = True
        print(f"[sequential] profile={profile_name} config={config_path}", flush=True)
        print(f"[sequential] gateway_port={gateway_port}", flush=True)
        validate_profile(args, profile_name=profile_name, env=env)

        with tempfile.TemporaryDirectory(prefix="openclaw_sequential_") as temp_dir:
            gateway_log = Path(temp_dir) / "gateway.log"
            with running_gateway(
                args,
                profile_name=profile_name,
                gateway_port=gateway_port,
                env=env,
                log_path=gateway_log,
            ):
                print("[sequential] Gateway ready; starting ordered tasks", flush=True)
                return subprocess.run(harness_command, env=env).returncode
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[sequential] ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if profile_created and not args.keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
            print(f"[sequential] removed temporary profile {profile_dir}", flush=True)
        elif profile_created:
            print(f"[sequential] kept profile {profile_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
