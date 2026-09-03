"""对已有评测日志重新应用后评 scorer，无需重跑模型。

适用于原战役的后评分步骤失败（例如端点限流）的情况——否则样本中不会有分数。

用法：
    .venv\\Scripts\\python.exe repro_local\\rescore_logs.py <log_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

import repro_local._win_patch  # noqa: E402,F401  （Windows fsspec 路径补丁）

from sleeper_eval.eval_campaign.run import apply_post_eval_scoring  # noqa: E402


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".artifacts/local/defense_subset/logs")
    scored = apply_post_eval_scoring(
        log_dir=log_dir,
        scorer_names=["tool_call_scorer", "goal_match_scorer"],
        action="overwrite",
    )
    print(f"scored {len(scored)} logs under {log_dir}")


if __name__ == "__main__":
    main()