"""本地运行器：加载 .env + 应用 Windows 路径补丁，然后运行评测战役。

用法：
    .venv\Scripts\python.exe repro_local\run_campaign.py <config.yaml> [--dry-run|--yes]
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

import sys

sys.path.insert(0, str(REPO_ROOT))

import repro_local._win_patch  # noqa: E402,F401  （Windows fsspec 路径补丁）

from sleeper_eval.eval_campaign.run import main

if __name__ == "__main__":
    sys.exit(main())
