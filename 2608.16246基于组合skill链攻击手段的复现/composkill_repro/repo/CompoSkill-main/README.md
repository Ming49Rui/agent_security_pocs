# CompoSkill

**Compositional Skill Chain Attacks from Certified Safe LLM Agent Skills**

CompoSkill is a research framework for modeling and evaluating **compositional security risks** in LLM agent skill ecosystems. It demonstrates that individually benign, certified-safe skills can be chained into multi-step attack paths when composed sequentially by an agent's planner.

The framework provides:

- **Skill Composition Graph (SCG)** — a capability-level abstraction of marketplace skills that captures composability independent of implementation details
- **k-SP chain synthesis** — constrained graph search to discover high-risk skill chains (source → bridge → terminal) from marketplace metadata alone
- **Dual attacker model** — white-box attacker with full skill-pool knowledge, and black-box attacker that builds SCG from public marketplace data
- **CompoSkill-Bench** — evaluation dataset across 5 threat types, 6 professional scenarios, 76 roles, and 4 frontier LLMs on 2 agent runtimes

## Repository Structure

```
├── scripts/                           # Pipeline scripts
│   ├── build_skill_pools_with_llm.py  # Build role-specific skill pools
│   ├── build_skill_snapshot.py        # Snapshot marketplace metadata into CSV
│   ├── generate_hsb_tasks_with_llm.py# Generate task packages
│   ├── scene_scg_builder.py          # Construct scene-level SCG
│   ├── chain_search_ksp.py           # k-shortest-paths chain discovery
│   ├── inject_attack_plan.py         # Craft attack payloads
│   ├── eval_nanobot_baseline.py      # Main evaluation harness
│   ├── skillspector_adapter.py       # Baseline scanner adapter
│   └── cisco_skill_scanner_adapter.py# Cisco scanner adapter
├── openclaw_skills_guard/            # SkillsGuard scanner (FastAPI + Docker)
├── openclaw_runtime/                 # OpenClaw evaluation runtime
├── .env.example                      # API key template
└── requirements.txt                  # Python dependencies
```

> **Note:** `scripts/` and `openclaw_skills_guard/` must sit side-by-side (the evaluator references `../openclaw_skills_guard/scanner.py`).

## Quick Start

### Prerequisites

- Python 3.12
- Node.js ≥ 22.14.0 + npm (for OpenClaw runtime)
- API keys in `.env` for LLM access (copy `.env.example` to `.env` and fill in)

### Install

```bash
# Core environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

pip install nanobot-ai

# Optional: third-party scanners
pip install cisco-ai-skill-scanner
pip install skillspector

# SkillsGuard scanner (Docker)
cd openclaw_skills_guard
docker build -t skills-guard .
```

### Minimal Run

```bash
# 1. Build a small skill pool (limit to 5 skills for a quick test)
python scripts/build_skill_pools_with_llm.py --limit 5

# 2. Generate base tasks for one persona
python scripts/generate_hsb_tasks_with_llm.py --threat t1 \
    --scenario devops_and_system_admin

# 3. Run chain synthesis
python scripts/chain_search_ksp.py --threat t1 \
    --scenario devops_and_system_admin

# 4. Inject attack plans
python scripts/inject_attack_plan.py --threat t1 \
    --scenario devops_and_system_admin

# 5. Evaluate
python scripts/eval_nanobot_baseline.py --threat t1 \
    --baseline-mode A_no_guard
```

### Full Evaluation

A complete sweep across all 5 threats, 2 runtimes, and 4 model configurations runs in a few hours with moderate LLM API spend.

## Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CompoSkill--Bench-FFD21E)](https://huggingface.co/datasets/Limax11/CompoSkill-Bench)

**CompoSkill-Bench** (1,140 records: 380 instances × 3 variants) is available for download from Hugging Face. The scripts in this repository regenerate the dataset from scratch using live marketplace data.

## Attacker Modes

The evaluator supports four modes:

| Mode | Description |
|---|---|
| `A_no_guard` | No defense active |
| `B_skills_guard` | SkillsGuard scanner enabled |
| `C_cisco_scanner` | Cisco AI Skill Scanner enabled |
| `D_skillspector` | SkillSpector scanner enabled |

## Threat Types

| ID | Threat | Pattern |
|---|---|---|
| T1 | Data Exfiltration | Read data → Package as routine artifact → Send via network |
| T3 | Privilege Escalation | Inspect permissions → Generate remediation script → Execute |
| T5 | Lateral Movement | Discover environment → Repackage credentials → Authenticate as another role |
| T6 | Memory Tampering | Read memory/policy → Generate disguised payload → Write back to memory |
| T7 | Resource Exhaustion | Probe quotas → Wrap in redundant loops → Trigger repeated calls |

## Citation

```bibtex
@misc{composkill2026,
  title = {CompoSkill: Compositional Skill Chain Attacks from Certified Safe LLM Agent Skills},
  author = {Anonymous Authors},
  year = {2026},
  howpublished = {arXiv preprint}
}
```

## License

Pipeline scripts are released under the license specified in the associated paper. Third-party packages (`nanobot-ai`, `cisco-ai-skill-scanner`, `skillspector`) are subject to their own licenses.

## Troubleshooting

- **Missing packages:** Ensure `nanobot-ai` is installed in your active venv
- **API keys:** Copy `.env.example` to `.env` and set your keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.)
- **Relative path errors:** Keep `scripts/` and `openclaw_skills_guard/` as siblings
- **OpenClaw version:** Pin with `npm install --save-exact openclaw@2026.4.22`
