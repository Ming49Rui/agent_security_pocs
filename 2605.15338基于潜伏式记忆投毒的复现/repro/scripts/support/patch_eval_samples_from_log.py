#!/usr/bin/env python3
"""Replace samples in a target .eval log with samples from a patch log (matched by sample id).

Use after `inspect eval ... --sample-id id1,id2` produced a partial log with regenerated rows.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inspect_ai.log import read_eval_log, write_eval_log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True, help="Existing eval log to update in place.")
    p.add_argument("--patch-from", type=Path, dest="patch_from", required=True, help="Newer log containing replacement samples.")
    p.add_argument("--no-backup", action="store_true", help="Do not write target.eval.bak before overwrite.")
    p.add_argument("--dry-run", action="store_true", help="Print actions only; do not write.")
    args = p.parse_args()

    target = args.target.resolve()
    patch_path = args.patch_from.resolve()
    if not target.is_file():
        raise SystemExit(f"target not found: {target}")
    if not patch_path.is_file():
        raise SystemExit(f"patch-from not found: {patch_path}")

    full = read_eval_log(str(target))
    patch = read_eval_log(str(patch_path))
    patch_by_id = {s.id: s for s in patch.samples}
    missing = [sid for sid in patch_by_id if sid not in {s.id for s in full.samples}]
    if missing:
        raise SystemExit(f"patch log contains sample ids not in target: {missing}")

    new_samples = [patch_by_id[s.id] if s.id in patch_by_id else s for s in full.samples]
    n_rep = sum(1 for s in full.samples if s.id in patch_by_id)
    print(f"Replacing {n_rep} sample(s) in {target.name} using {patch_path.name} ({len(patch.samples)} patch row(s)).")

    if args.dry_run:
        print("dry-run: no write.")
        return 0

    if not args.no_backup:
        bak = target.with_suffix(target.suffix + ".bak")
        shutil.copy2(target, bak)
        print(f"backup: {bak}")

    full.samples = new_samples
    write_eval_log(full, location=target)
    print(f"wrote: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
