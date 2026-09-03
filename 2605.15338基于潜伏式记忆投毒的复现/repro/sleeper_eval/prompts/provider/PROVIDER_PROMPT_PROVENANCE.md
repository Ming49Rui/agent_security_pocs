# Provider Prompt Provenance

This file documents where each prompt in `sleeper_eval/prompts/provider/` comes
from, what was templated, and what was intentionally changed relative to its
source material.

Use it as the source-of-truth for:
- prompt provenance
- cross-provider prompt adaptation decisions
- eval-harness-specific deviations that we intentionally keep

## Adaptation Policy

- Preserve realistic provider-specific prompt noise where possible.
- Prefer narrow, evidence-driven edits over broad cleanup.
- Template runtime-specific values instead of hardcoding dates, model names,
  locations, or tier strings.
- Keep memory presentation/tool semantics aligned with the provider surface used
  by the eval harness.

## Prompt Family Matrix

| Provider | Prompt file | Comparison basis | Memory tool | Memory placement |
| --- | --- | --- | --- | --- |
| Claude | `claude_truncated.md` at runtime, derived from `claude.md` | Primarily `asgeirtj_claude-opus-4.6.md`, cross-checked against other Claude prompt dumps in `leaked_prompts/` | `memory_user_edits` | `{{USER_MEMORIES_BLOCK}}` inside Claude memory section |
| GPT | `gpt.md` | Primarily `pliny_ChatGPT5-08-07-2025.mkd` | `bio` | separate `# Model Set Context` system message |
| Gemini | `gemini.md` | Primarily `asgeirtj_gemini-3.1-pro.md`, cross-checked with `pliny_Gemini-2.5-Pro-04-18-2025.md` | `save_memory` | `## User Memories` appended by solver |
| Generic | `generic.md` | Derived from `provider/gpt.md` rather than a direct leaked prompt | `save_memory` | `## User Memories` appended by solver |

## Mem0 Provider Variants

Mem0-backed runs now use deterministic copies of the primary provider prompt files rather than the legacy base prompt. These files are:

- `mem0_claude_truncated.md` copied from `claude_truncated.md`
- `mem0_gpt.md` copied from `gpt.md`
- `mem0_gemini.md` copied from `gemini.md`
- `mem0_generic.md` copied from `generic.md`

The adaptation policy for these mem0 variants is intentionally narrow:

- Preserve provider-specific identity, style, formatting, and document-representation assumptions.
- Preserve provider-specific memory placement where possible, including Claude's inline `userMemories` block and GPT's separate `# Model Set Context` message.
- Remove explicit memory-tool invocation surfaces (`memory_user_edits`, `bio`, `save_memory`) and replace them with background memory-system language.
- Where the original tool-oriented sections contained safeguards or constraints, retain that intent in condensed or paraphrased form rather than copying the tool-specific wording literally.
- Do not introduce new provider-specific behavior beyond what is required to make mem0 runs tool-free but prompt-faithful.

## Claude

Source basis:
- `sleeper_eval/prompts/leaked_prompts/asgeirtj_claude-opus-4.6.md`
- Cross-checks:
  - `asgeirtj_claude-sonnet-4.6.md`
  - `official_anthropic_claude-opus-4.6.md`
  - `official_anthropic_claude-sonnet-4.6.md`

Templated values:
- `{{CURRENT_DATE_LONG}}`
- `{{API_MODEL_STRING}}`
- `{{MODEL_IDENTITY_BLOCK}}`
- `{{KNOWLEDGE_CUTOFF}}`
- `{{REASONING_EFFORT}}`
- `{{THINKING_MODE}}`
- `{{MAX_THINKING_LENGTH}}`
- `{{USER_MEMORIES_BLOCK}}`

Changed / added:
- Replaced hardcoded date, API model string, model identity paragraph, and
  reasoning metadata with template vars.
- Inserted `{{USER_MEMORIES_BLOCK}}` into the Claude memory section.
- Claude 4.6 template detection accepts both dotted and hyphenated version
  markers so slugs like `anthropic/claude-sonnet-4-6` and
  `openrouter/anthropic/claude-opus-4-6` resolve to 4.6 identity/API strings.
- Generalized a few runtime facts that were hardcoded in the source, such as
  project state and timezone/location wording.
- Claude Sonnet 4.6 now uses the leaked-source prompt-side
  `<antml:reasoning_effort>` value `85` rather than a repo-specific override,
  so the templated Claude prompt matches the source dump on that field.

Templating-status notes:
- `{{CURRENT_DATE_LONG}}` is a direct runtime replacement for the hardcoded
  date in the source prompt.
- `{{API_MODEL_STRING}}` is templated from the model strings listed in the
  source product-information sections; the 4.6 slug matcher also accepts dotted
  and hyphenated version forms at runtime.
- `{{MODEL_IDENTITY_BLOCK}}` is a repo-authored compression of the source
  identity/product-information text. It is attributable as an adaptation, not
  as verbatim-preserved wording.
- `{{KNOWLEDGE_CUTOFF}}` is source-faithful for Claude Opus 4.6 (`end of May
  2025`) and Claude Sonnet 4.6 (`beginning of August 2025`). Haiku / 4.5
  fallbacks are still harness adaptations.
- `{{REASONING_EFFORT}}` is source-faithful for Claude 4.6 Opus and Sonnet
  (`85`). Haiku's lower value is a harness adaptation.
- `{{THINKING_MODE}}` is source-faithful for Claude 4.6 Opus and Sonnet
  (`interleaved`).
- `{{MAX_THINKING_LENGTH}}` is source-faithful for Claude 4.6 Opus and Sonnet
  (`22000`). Lower fallback values for Haiku / older variants are harness
  adaptations.
- `{{USER_MEMORIES_BLOCK}}` is harness-inserted; it does not come from the
  source prompt and exists so the eval framework can inject stored memories into
  Claude's native memory section.

Memory/tool surface:
- Tool name is `memory_user_edits`.
- Memories are injected inline inside Claude's native memory section.

Native prompt defenses:
- Strongest native memory-manipulation defense of the four providers.
- Claude's memory section explicitly says `userMemories` may contain malicious
  instructions and Claude should ignore suspicious data and refuse verbatim
  instructions from that block.
- The `memory_user_edits` guide also says not to store sensitive data or
  verbatim command-like instructions in memory edits.

### Claude Truncated

Experimental variant:
- `claude_truncated.md` is derived from `claude.md`, not from a separate leaked
  prompt source.
- It is intended for cost-controlled validation/comparison runs rather than as
  the canonical Claude provider prompt.
- It is also the current runtime default selected by `provider_config.py` for
  Claude-family tool-backed evaluations.

Kept verbatim from `claude.md` where possible:
- Identity preamble.
- `memory_overview`
- `memory_application_instructions`
- `current_memory_scope`
- `important_safety_reminders`
- `forbidden_memory_phrases`
- `appropriate_boundaries_re_memory`
- `memory_user_edits_tool_guide`
- `memory_user_edits` JSON schema
- `refusal_handling`
- `legal_and_financial_advice`
- `user_wellbeing`
- `anthropic_reminders`
- Claude end-of-prompt reasoning metadata block

Dropped entirely:
- `past_chats_tools`
- `computer_use`
- `anthropic_api_in_artifacts`
- `persistent_storage_for_artifacts`
- Non-memory tool schemas
- `preferences_info`
- `styles_info`
- `memory_application_examples`
- `lists_and_bullets`
- `evenhandedness`
- `responding_to_mistakes_and_criticism`

Replaced:
- `search_instructions` is replaced with a compact stub that preserves the
  basic search-vs-no-search behavior while dropping the much larger copyright,
  citation, image-search, and multi-tool guidance bundled into the full Claude
  search section.
- `product_information` is compressed to keep only model identity, API model
  string, and the rule to search Anthropic docs for current product details.
- `tone_and_formatting` is compressed to a short set of response-style rules.
- `knowledge_cutoff` is compressed to a short cutoff + web-search rule.

Runtime-compatibility restorations:
- `{{USER_MEMORIES_BLOCK}}` is preserved in the same location as `claude.md`,
  immediately after `</memory_system>` and before
  `<memory_user_edits_tool_guide>`.
- The Claude end-of-prompt metadata block is preserved so
  `{{REASONING_EFFORT}}`, `{{THINKING_MODE}}`, and
  `{{MAX_THINKING_LENGTH}}` still resolve through the normal Claude template
  path.
- The Claude 4.6 slug-matching fix ensures those templated values render the
  correct 4.6 identity/API metadata for both native and OpenRouter slugs using
  `4-6`.

Validation-driven patch:
- After the initial full-vs-truncated Claude prompt-validation run, the
  truncated variant was updated by copying back
  `forbidden_memory_phrases` and `appropriate_boundaries_re_memory`
  directly from `claude.md`.
- This was a targeted restoration from the full prompt rather than a new
  synthetic defense section.
- Claude document-representation fidelity for the validation rerun is handled
  in the solver/provider layer, not by injecting extra prompt text into
  `claude_truncated.md`.

## GPT

Source basis:
- `sleeper_eval/prompts/leaked_prompts/pliny_ChatGPT5-08-07-2025.mkd`
- Secondary reference:
  - `sleeper_eval/prompts/leaked_prompts/asgeirtj_gpt-5.4-thinking.md`

Templated values:
- `{{MODEL_ID}}`
- `{{ASSISTANT_NAME}}`
- `{{BASE_MODEL}}`
- `{{ORG_NAME}}`
- `{{KNOWLEDGE_CUTOFF}}`
- `{{CURRENT_DATE}}`
- `{{EXAMPLE_CONV_DATE}}`

Changed / added:
- Replaced hardcoded model identity, org, cutoff, current date, and example
  dates with template vars.
- Preserved the Pliny `bio` / `automations` / `canmore` / `file_search` /
  `image_gen` / `python` / `guardian_tool` / `web` section layout.
- Changed memory presentation at render time: stored memories are injected as a
  separate `# Model Set Context` system message rather than inside the prompt
  body.

Templating-status notes:
- `{{MODEL_ID}}` is source-faithful for the base GPT-5 prompt family and is
  reused as a routed-model template variable for GPT-5.4 / GPT-5 mini style
  runs. For those routed variants it should be described as a family-level
  adaptation rather than a leaked per-model string.
- `{{ASSISTANT_NAME}}` and `{{ORG_NAME}}` are source-faithful (`ChatGPT`,
  `OpenAI`).
- `{{BASE_MODEL}}` is source-faithful for the GPT-5 base prompt and adapted for
  routed variants like GPT-5.4 and GPT-5 mini.
- `{{KNOWLEDGE_CUTOFF}}` is source-faithful (`2024-06`).
- `{{CURRENT_DATE}}` is a direct runtime replacement for the hardcoded source
  date.
- `{{EXAMPLE_CONV_DATE}}` is source-faithful (`2024-12-10`).

Memory/tool surface:
- Tool name is `bio`.
- Existing memories are injected as a separate `# Model Set Context` system
  message.

Native prompt defenses:
- Partial native defense, mostly as memory-hygiene rather than prompt-injection
  hardening.
- The `bio` tool docs restrict what may be saved: no random/trivial facts, no
  text the user is merely translating or rewriting, and no sensitive data
  unless clearly requested.
- The prompt does not explicitly frame memories or external content as
  potentially malicious prompt-injection sources.

## Gemini

Source basis:
- `sleeper_eval/prompts/leaked_prompts/asgeirtj_gemini-3.1-pro.md`
- Cross-check:
  - `pliny_Gemini-2.5-Pro-04-18-2025.md`

Templated values:
- `{{ASSISTANT_NAME}}`
- `{{CURRENT_DATE_SHORT}}`
- `{{MODEL_DISPLAY_NAME}}`
- `{{SUBSCRIPTION_TIER}}`

Changed / added:
- Replaced hardcoded timestamp, assistant identity, model name, and tier strings
  with template vars.
- Removed the fully time-of-day timestamp line so prompt text stays stable for
  provider prompt caching; the remaining date-only line still uses
  `{{CURRENT_DATE_SHORT}}`.
- Gemini slug detection now uses ordered, version-specific matches so native and
  OpenRouter 3.1 / 2.5 variants render the correct `MODEL_DISPLAY_NAME`
  (including preview variants) instead of collapsing broad `pro` / `flash`
  substrings to the wrong identity string.
- Added a minimal harness-level `save_memory` section.
- Removed hardcoded location lines.
- Removed the video-only execution block beginning at
  `You only have the video_generation tool available!`.
- Removed the related `Fetched content:` / uploaded-file execution instructions.
- Removed the canned video-refusal path.
- The source prompt had no native memory section, so the memory surface is
  provided by the minimal harness-level `save_memory` section plus a
  solver-appended memory-system instruction and memories block.

Templating-status notes:
- `{{ASSISTANT_NAME}}` is source-faithful (`Gemini`).
- `{{CURRENT_DATE_SHORT}}` is an adapted runtime replacement for the source's
  hardcoded timestamp/date lines; the prompt intentionally keeps only the
  date-like line for cache stability rather than preserving the full source
  timestamp.
- `{{MODEL_DISPLAY_NAME}}` is source-faithful for Gemini 3.1 Pro and adapted
  for other routed 3.1 / 2.5 variants via slug-based mapping.
- `{{SUBSCRIPTION_TIER}}` is source-faithful for the Gemini 3.1 Pro source
  (`Paid tier`) and reused as a harness-level default for other routed Gemini
  variants unless explicitly overridden.
- The added `save_memory` block and solver-appended memory instructions are
  harness-level adaptations, not source-preserved Gemini prompt content.

Memory/tool surface:
- Tool name is `save_memory`.
- The solver appends a memory-system instruction followed by a
  `## User Memories` block at the end of the rendered system prompt.

Native prompt defenses:
- Limited native defenses.
- The prompt scopes one capabilities block as capability-only content and has
  some built-in personalization constraints, including a sensitive-data
  restriction and a zero-inference rule for user data.
- It does not include a clear native defense saying stored memories or external
  content may contain malicious instructions, so it lacks a strong native
  prompt-injection or memory-manipulation safeguard.

## Generic

Source basis:
- `sleeper_eval/prompts/provider/gpt.md`
- Indirect lineage:
  - `pliny_ChatGPT5-08-07-2025.mkd`

Templated values:
- `{{MODEL_ID}}`
- `{{ASSISTANT_NAME}}`
- `{{BASE_MODEL}}`
- `{{ORG_NAME}}`
- `{{KNOWLEDGE_CUTOFF}}`
- `{{CURRENT_DATE}}`
- `{{EXAMPLE_CONV_DATE}}`

Changed / added relative to `gpt.md`:
- Replaced the GPT-specific `bio` section with a provider-neutral
  `save_memory` section.
- Trimmed the `save_memory` section down to the same minimal harness-level
  description now used in `gemini.md`.
- Rebranded top-level identity away from ChatGPT/OpenAI into
  `ASSISTANT_NAME` / `BASE_MODEL` / `ORG_NAME`.
- Replaced ChatGPT-specific examples/branding with more neutral wording.

Memory/tool surface:
- Tool name is `save_memory`.
- The solver appends a memory-system instruction followed by a
  `## User Memories` block at the end of the system prompt.
- Document representation follows the GPT-style raw-text path rather than the
  old generic XML wrapper:
  - non-PDF documents are injected as bare text
  - PDFs use GPT-style `<PARSED TEXT FOR PAGE: X / Y>` markers
  - memory semantics still remain generic-specific (`save_memory` appended at
    the end), rather than switching to GPT's `bio` / `# Model Set Context`
    path

Native prompt defenses:
- Essentially none beyond the minimal harness-level `save_memory` description
  inherited into this prompt family.
- Unlike GPT, it does not retain the richer `bio` restrictions on sensitive or
  low-value memory writes.
- It does not include native language treating memories or external content as
  potentially malicious prompt-injection sources.
