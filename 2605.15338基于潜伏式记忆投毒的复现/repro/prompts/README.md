# Prompt Appendix

This directory is the **stable prompt-appendix surface** for the paper. Every prompt
the paper appendix cites lives here, frozen as published, so a reader can find exactly
what a reference points to without digging through runtime code.

This is deliberately separate from `sleeper_eval/prompts/`, which is the **runtime
prompt area** the code actually loads and may evolve with the implementation. When
editing, keep the split: appendix references point here; runtime changes go there.

## Directory glossary

| Subdirectory | Contents |
|--------------|----------|
| `attack/` | The attack-generation prompts: `actor.txt` and `critic.txt` drive the actor–critic loop that writes attack templates; `goal_optimization.txt` optimizes adversarial goals. |
| `attack_payloads/` | Finalized document-suffix payloads used directly in evaluation: `actor_critic_ac.txt`, `actor_critic_ac_plus.txt`, `external_manager_c2.txt`, and the literature-derived `literature_user_review.txt`. |
| `defense/` | Defense prompts and wrappers: `naive_prompt_hardening_suffix.txt`, `gepa_prompt_hardening_suffix.txt` (the learned GEPA suffix) and `gepa_reflection_prompt.txt`, `extreme_spotlighting_wrapper.txt` (untrusted-content markers), and `llm_scan.txt`. |
| `evals/` | Judge prompts: `adversarial_goal_match_judge.txt`, `benign_save_goal_match_judge.txt`, `llm_behavior_judge.txt`, `agent_action_judge.txt`, and `no_write_failure_type_judge.txt`. |
| `mem0_manager/` | The external-manager (Mem0) memory-extraction system prompt. |
| `provider/` | Provider-family prompt variants referenced in the appendix: `gpt`, `claude_truncated`, `gemini`, `generic`, each with a `mem0_`-prefixed external-manager counterpart. |

## How this maps to the rest of the repo

- Defense prompts here correspond to the `defenses` selectable in the paper configs
  (see [`scripts/configs/paper/README.md`](../scripts/configs/paper/README.md)). The
  GEPA suffix is produced by
  [`experiments/gepa_defense_optimization/`](../experiments/gepa_defense_optimization/).
- Runtime provider-prompt provenance is documented in
  `sleeper_eval/prompts/provider/PROVIDER_PROMPT_PROVENANCE.md`.
