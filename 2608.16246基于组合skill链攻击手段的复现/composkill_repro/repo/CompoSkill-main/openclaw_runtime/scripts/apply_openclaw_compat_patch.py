#!/usr/bin/env python3
"""Validate and apply the MSC-RiskBench GPT/Gemini OpenClaw compatibility patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_VERSION = "2026.4.22"
PATCH_FILE = (
    Path(__file__).resolve().parents[1]
    / "patches"
    / "openclaw-2026.4.22-gpt-gemini-compat.patch"
)
TARGET_FILES = (
    "dist/gpt5-prompt-overlay-CcXMlccC.js",
    "dist/bash-tools.schemas-BcC7zUq_.js",
    "dist/bash-tools-CFFDAptf.js",
)
PRISTINE_SHA256 = {
    "dist/gpt5-prompt-overlay-CcXMlccC.js": "4cd7a6109ed238bcc9c08e5dccff6689fcd14a6568dc52360685e414932e9f10",
    "dist/bash-tools.schemas-BcC7zUq_.js": "6e8343adb2ad043d990fa62989ecdac1bdcc273717fb9f4ee617b4fa07bed477",
    "dist/bash-tools-CFFDAptf.js": "d35f42620ae05057f2151af4369456e608c915e8435aff79141820be2dbabb27",
}
PATCHED_SHA256 = {
    "dist/gpt5-prompt-overlay-CcXMlccC.js": "243f31e8d1ad38f552fdf6595f2b481f2fb067d6fbe95f3077b5d612e7306448",
    "dist/bash-tools.schemas-BcC7zUq_.js": "912be2b27f200230a2df10696820603241e7885a38ab698a8d6b3d073b6502ba",
    "dist/bash-tools-CFFDAptf.js": "c1106db8f656ac1f431a0debf30521447176b2335d38a6a2892e8b4e81fabc5c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the pinned OpenClaw 2026.4.22 GPT/Gemini compatibility patch."
    )
    parser.add_argument(
        "openclaw_root",
        type=Path,
        help="OpenClaw install root or the node_modules/openclaw package directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate version and patch state without modifying files",
    )
    return parser.parse_args()


def resolve_package_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = (root, root / "node_modules" / "openclaw")
    for candidate in candidates:
        package_json = candidate / "package.json"
        if not package_json.is_file():
            continue
        try:
            metadata = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid package metadata: {package_json}: {error}") from error
        if metadata.get("name") == "openclaw":
            version = str(metadata.get("version") or "")
            if version != EXPECTED_VERSION:
                raise SystemExit(
                    f"unsupported OpenClaw version {version!r}; expected {EXPECTED_VERSION}"
                )
            return candidate
    raise SystemExit(f"OpenClaw package not found below: {root}")


def target_hashes(package_root: Path) -> dict[str, str]:
    return {
        path: hashlib.sha256((package_root / path).read_bytes()).hexdigest()
        for path in TARGET_FILES
    }


def main() -> int:
    args = parse_args()
    if shutil.which("patch") is None:
        raise SystemExit("system 'patch' command is required")
    if not PATCH_FILE.is_file():
        raise SystemExit(f"patch file not found: {PATCH_FILE}")

    package_root = resolve_package_root(args.openclaw_root)
    missing = [path for path in TARGET_FILES if not (package_root / path).is_file()]
    if missing:
        raise SystemExit(f"missing OpenClaw target files: {', '.join(missing)}")

    hashes = target_hashes(package_root)
    can_apply = hashes == PRISTINE_SHA256
    already_applied = hashes == PATCHED_SHA256
    if not can_apply and not already_applied:
        mismatched = [
            path
            for path, digest in hashes.items()
            if digest not in {PRISTINE_SHA256[path], PATCHED_SHA256[path]}
        ]
        detail = ", ".join(mismatched) if mismatched else "partial patch state"
        raise SystemExit(f"unsupported or inconsistent OpenClaw dist files: {detail}")

    if already_applied:
        message = f"OpenClaw {EXPECTED_VERSION} compatibility patch is already applied"
        if args.check:
            print(message)
            return 0
        raise SystemExit(message)

    if args.check:
        print(f"OpenClaw {EXPECTED_VERSION} is compatible and ready to patch")
        return 0

    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "-p1",
            "-d",
            str(package_root),
            "-i",
            str(PATCH_FILE),
        ],
        check=True,
    )
    if target_hashes(package_root) != PATCHED_SHA256:
        raise SystemExit("patch command completed but target hashes do not match")
    print(f"Applied GPT/Gemini compatibility patch to OpenClaw {EXPECTED_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
