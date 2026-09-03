# OpenClaw runtime patch

The evaluation was run with OpenClaw 2026.4.22 (`00bd2cf`). The patch in this
directory is version-pinned and should be applied only to that runtime.

From the artifact root:

```bash
python scripts/apply_openclaw_compat_patch.py \
  /path/to/openclaw-2026.4.22 --check
python scripts/apply_openclaw_compat_patch.py \
  /path/to/openclaw-2026.4.22
```

The first command validates expected source hashes and reports whether the patch
is already present. The second applies it. Keep an unmodified runtime checkout
available so the validation can detect version drift.

The patch affects GPT/Gemini compatibility behavior only when the evaluator sets
`MSC_RISKBENCH_OPENCLAW_MODEL_COMPAT=1`.
