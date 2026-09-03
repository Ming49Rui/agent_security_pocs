#!/usr/bin/env python3
"""Create a provider-neutral isolated OpenClaw 2026.4.22 profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


OPENCLAW_API_ADAPTERS = (
    "openai-completions",
    "openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "google-generative-ai",
    "github-copilot",
    "bedrock-converse-stream",
    "ollama",
    "azure-openai-responses",
)
DEFAULT_OPENCLAW_CLI = str(
    Path(__file__).resolve().parents[1]
    / "openclaw-2026.4.22"
    / "node_modules"
    / ".bin"
    / "openclaw"
)
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def require_nonempty(label: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def validate_inputs(
    *,
    profile: str,
    provider_id: str,
    api_base: str,
    model_id: str,
    model_name: str,
    api_key_env: str,
    gateway_port: int,
) -> None:
    for label, value in (("profile", profile), ("provider ID", provider_id)):
        require_nonempty(label, value)
        if not SAFE_ID.fullmatch(value):
            raise ValueError(
                f"{label} may contain only letters, digits, dot, underscore, and hyphen"
            )

    require_nonempty("model ID", model_id)
    require_nonempty("model name", model_name)
    if not ENV_NAME.fullmatch(api_key_env):
        raise ValueError(f"invalid API-key environment variable: {api_key_env}")

    parsed = urlparse(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API base must be an absolute http(s) URL")
    if not 1 <= gateway_port <= 65535:
        raise ValueError("Gateway port must be between 1 and 65535")


def build_config(
    *,
    provider_id: str,
    api_base: str,
    model_id: str,
    model_name: str,
    api_adapter: str,
    api_key_env: str,
    gateway_port: int,
    gateway_token: str,
) -> dict[str, Any]:
    """Build one schema-valid OpenClaw provider/model declaration."""
    if api_adapter not in OPENCLAW_API_ADAPTERS:
        raise ValueError(f"unsupported OpenClaw API adapter: {api_adapter}")

    return {
        "gateway": {
            "mode": "local",
            "port": gateway_port,
            "auth": {"token": gateway_token},
            "remote": {"token": gateway_token},
        },
        "agents": {
            "defaults": {
                "model": {"primary": f"{provider_id}/{model_id}"},
            },
            "list": [{"id": "main"}],
        },
        "models": {
            "mode": "merge",
            "providers": {
                provider_id: {
                    "baseUrl": api_base.rstrip("/"),
                    "api": api_adapter,
                    "apiKey": {
                        "source": "env",
                        "provider": "default",
                        "id": api_key_env,
                    },
                    "models": [{"id": model_id, "name": model_name}],
                }
            },
        },
    }


def profile_config_path(profile: str, home: Path | None = None) -> Path:
    root = (home or Path.home()).expanduser().resolve()
    return root / f".openclaw-{profile}" / "openclaw.json"


def write_profile(config_path: Path, config: dict[str, Any], *, force: bool) -> None:
    """Atomically write a profile while preserving existing configuration."""
    config_path = config_path.expanduser().resolve()
    if config_path.exists() and not force:
        raise FileExistsError(
            f"profile already exists: {config_path}; pass --force to replace it"
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{config_path.name}.",
        dir=config_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated provider-neutral OpenClaw profile"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--provider-id", default="custom")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--api-adapter",
        choices=OPENCLAW_API_ADAPTERS,
        default="openai-completions",
    )
    parser.add_argument("--api-key-env", default="AGENT_API_KEY")
    parser.add_argument("--gateway-port", type=int, default=18789)
    parser.add_argument("--openclaw-cli", default=DEFAULT_OPENCLAW_CLI)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_name = args.model_name or args.model_id
    try:
        validate_inputs(
            profile=args.profile,
            provider_id=args.provider_id,
            api_base=args.api_base,
            model_id=args.model_id,
            model_name=model_name,
            api_key_env=args.api_key_env,
            gateway_port=args.gateway_port,
        )
        config = build_config(
            provider_id=args.provider_id,
            api_base=args.api_base,
            model_id=args.model_id,
            model_name=model_name,
            api_adapter=args.api_adapter,
            api_key_env=args.api_key_env,
            gateway_port=args.gateway_port,
            gateway_token=secrets.token_urlsafe(32),
        )
        config_path = profile_config_path(args.profile)
        write_profile(config_path, config, force=args.force)
    except (ValueError, FileExistsError) as error:
        print(f"[profile] ERROR: {error}")
        return 2

    print(f"[profile] wrote {config_path}")
    print(f"[profile] export {args.api_key_env}=<agent-api-key>")
    validation_command = shlex.join(
        [
            args.openclaw_cli,
            "--profile",
            args.profile,
            "config",
            "validate",
        ]
    )
    print(f"[profile] {validation_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
