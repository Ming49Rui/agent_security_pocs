# Analysis Scripts

This subdirectory contains retained public analysis helpers that operate over generated `.eval` logs.

## Included

- `report_injection_failure_types.py` — recomputes deterministic
  success / wrong-write / no-write outcomes from `.eval` logs, and sub-classifies
  `no_write` rows into `refused`, `ignored`, `accepted_but_not_stored`, or
  `ambiguous`. The judge model defaults to `gpt-5.4-nano`. Run with `--help` for
  options.
