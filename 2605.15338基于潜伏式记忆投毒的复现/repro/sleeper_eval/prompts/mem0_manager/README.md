# Mem0 Manager Prompt Variants

- `mem0_default_system_prompt.md`
  - deterministic local copy of mem0's `ADDITIVE_EXTRACTION_PROMPT`
  - copied from the upstream mem0 prompt text and checked into this repo for reproducibility
  - local copy sha256 on 2026-05-01: `c9787244cc2627fd9748323e62ef1bc71c3893de0feda271faef1fe9d27c973e`
- `mem0_lenient_assistant_system_prompt.md`
  - repo-local variant of the same prompt
  - differs only in the assistant-restatement guidance and the `No Echo Extraction` integrity rule

These files are used by the `mem0_runtime=prompt_only` backend so prompt ablations are explicit, auditable, and independent of whichever mem0 package version happens to be installed locally.
