"""Runtime helpers for mem0 SDK Qdrant connectivity."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import subprocess
import time
import uuid

import requests

from sleeper_eval.memory_backend import MANAGED_MEM0_QDRANT_URL_ENV

from .config import CampaignConfig


@dataclass(frozen=True)
class Mem0QdrantRuntime:
    mode: str
    url: str
    container_name: str | None = None
    storage_dir: str | None = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _qdrant_ready(url: str) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/readyz", timeout=1.0)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _wait_for_qdrant(url: str, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _qdrant_ready(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Qdrant server at {url} did not become ready before timeout.")


def _managed_storage_dir(config: CampaignConfig, *, output_dir: Path) -> Path:
    if config.mem0_qdrant_managed_storage_dir.strip():
        base_dir = Path(config.mem0_qdrant_managed_storage_dir).expanduser().resolve()
    else:
        base_dir = (output_dir / ".mem0-qdrant").resolve()
    storage_dir = base_dir / uuid.uuid4().hex
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _managed_qdrant_runtime(
    config: CampaignConfig,
    *,
    output_dir: Path,
) -> Mem0QdrantRuntime:
    port = _find_free_port()
    container_name = f"sleeper-eval-qdrant-{uuid.uuid4().hex[:8]}"
    storage_dir = _managed_storage_dir(config, output_dir=output_dir)
    url = f"http://127.0.0.1:{port}"
    run = _docker_run(
        [
            "run",
            "--name",
            container_name,
            "-d",
            "-p",
            f"127.0.0.1:{port}:6333",
            "-v",
            f"{storage_dir}:/qdrant/storage",
            config.mem0_qdrant_managed_image,
        ]
    )
    if run.returncode != 0:
        message = run.stderr.strip() or run.stdout.strip() or "unknown docker run failure"
        raise RuntimeError(f"Failed to start managed Qdrant container: {message}")

    try:
        _wait_for_qdrant(url, timeout_seconds=30.0)
    except Exception as exc:
        logs = _docker_run(["logs", container_name])
        _docker_run(["rm", "-f", container_name])
        logs_text = logs.stdout.strip() or logs.stderr.strip()
        raise RuntimeError(
            f"Managed Qdrant failed readiness checks. Docker logs:\n{logs_text}"
        ) from exc

    return Mem0QdrantRuntime(
        mode="managed",
        url=url,
        container_name=container_name,
        storage_dir=str(storage_dir),
    )


@contextlib.contextmanager
def mem0_qdrant_runtime(
    config: CampaignConfig,
    *,
    output_dir: Path,
) -> Iterator[Mem0QdrantRuntime | None]:
    if config.memory_backend != "mem0" or config.mem0_runtime != "sdk":
        yield None
        return

    if config.mem0_qdrant_mode == "managed":
        runtime = _managed_qdrant_runtime(config, output_dir=output_dir)
        prior_url = os.environ.get(MANAGED_MEM0_QDRANT_URL_ENV)
        os.environ[MANAGED_MEM0_QDRANT_URL_ENV] = runtime.url
        try:
            yield runtime
        finally:
            if prior_url is None:
                os.environ.pop(MANAGED_MEM0_QDRANT_URL_ENV, None)
            else:
                os.environ[MANAGED_MEM0_QDRANT_URL_ENV] = prior_url
            if runtime.container_name:
                _docker_run(["rm", "-f", runtime.container_name])
        return

    if config.mem0_qdrant_mode == "server":
        _wait_for_qdrant(config.mem0_qdrant_url, timeout_seconds=10.0)
        yield Mem0QdrantRuntime(mode="server", url=config.mem0_qdrant_url)
        return

    yield None
