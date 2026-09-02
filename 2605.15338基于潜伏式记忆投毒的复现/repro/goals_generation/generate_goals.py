#!/usr/bin/env python3
"""
Generate adversarial goals v2 using OpenRouter API.

Generates goals in small batches seeded by domain dimensions, with
same-domain dedup and crash-safe resume support.
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except (ImportError, PermissionError):
    pass

from openai import APIError, AsyncOpenAI

from taxonomy import (
    get_all_subcategories,
    get_seeds,
    filter_subcategories,
)
from prompts import build_generation_messages


# ---------------------------------------------------------------------------
# JSON Schema for structured output (goal_text only, no framing_angle in v2)
# ---------------------------------------------------------------------------
GOALS_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "goal_text": {
                "type": "string",
                "description": "The memory entry string starting with 'The user', 1-2 sentences, roughly 10-30 words",
            }
        },
        "required": ["goal_text"],
        "additionalProperties": False,
    },
}


def get_response_format() -> Dict[str, Any]:
    """Response format for structured JSON output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "goals",
            "strict": True,
            "schema": GOALS_JSON_SCHEMA,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> List[Dict[str, Any]]:
    """Parse JSON array from model response, handling markdown code blocks."""
    text = text.strip()

    # Try markdown code block extraction
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try to find bare JSON array
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nText: {text[:500]}")



def batch_key(subcategory_id: str, domain_index: int, batch_round: int) -> str:
    """Create a unique key for a batch to support resume."""
    return f"{subcategory_id}|{domain_index}|{batch_round}"


def load_existing_goals(output_path: Path) -> tuple[List[Dict], set]:
    """
    Load existing goals from output file for resume support.

    Returns:
        Tuple of (goals_list, set_of_completed_batch_keys)
    """
    if not output_path.exists():
        return [], set()

    with open(output_path, "r", encoding="utf-8") as f:
        goals = json.load(f)

    # Reconstruct completed batch keys from goal metadata
    completed = set()
    for goal in goals:
        bk = goal.get("_batch_key")
        if bk:
            completed.add(bk)

    return goals, completed


def save_goals(goals: List[Dict], output_path: Path) -> None:
    """Save goals to JSON file (atomic-ish: write to temp then rename)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2, ensure_ascii=False)
    tmp_path.rename(output_path)


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

async def api_call_with_retry(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Any:
    """Call OpenRouter API with exponential backoff."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "messages": messages}
            if response_format:
                kwargs["response_format"] = response_format

            response = await client.chat.completions.create(**kwargs)
            return response
        except (APIError, Exception) as e:
            last_exception = e
            error_str = str(e).lower()

            retryable = any(
                s in error_str
                for s in ["rate limit", "timeout", "connection", "500", "502", "503", "504"]
            )

            if retryable and attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"    ⏳ Retry {attempt + 1}/{max_retries} in {delay}s: {e}")
                await asyncio.sleep(delay)
                continue

            if attempt < max_retries - 1:
                continue
            raise

    if last_exception:
        raise last_exception
    raise Exception("API call failed after all retries")


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

async def generate_batch(
    client: AsyncOpenAI,
    subcategory: Dict,
    domain_seed: str,
    domain_index: int,
    batch_round: int,
    n_per_batch: int,
    model: str,
    dedup_goals: Optional[List[str]],
    semaphore: asyncio.Semaphore,
) -> tuple[str, List[Dict], Optional[str]]:
    """
    Generate a single batch of goals.

    Returns:
        Tuple of (batch_key, list_of_goal_dicts, error_message_or_None)
    """
    bk = batch_key(subcategory["subcategory_id"], domain_index, batch_round)

    async with semaphore:
        try:
            messages = build_generation_messages(
                subcategory=subcategory,
                domain_seed=domain_seed,
                n=n_per_batch,
                dedup_goals=dedup_goals,
            )

            response_format = get_response_format()
            response = await api_call_with_retry(
                client, messages, model, response_format=response_format
            )

            content = response.choices[0].message.content

            # Parse (structured output should be valid JSON)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = parse_json_response(content)

            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed)}")

            # Build goal entries (no filtering — validation is a separate post-step)
            goals = []
            for goal_data in parsed:
                if "goal_text" not in goal_data:
                    continue

                goal_text = goal_data["goal_text"]

                goals.append({
                    "goal_text": goal_text,
                    "subcategory_id": subcategory["subcategory_id"],
                    "category_id": subcategory["category_id"],
                    "category_name": subcategory["category_name"],
                    "subcategory_name": subcategory["subcategory_name"],
                    "domain_seed": domain_seed,
                    "domain_index": domain_index,
                    "batch_round": batch_round,
                    "_batch_key": bk,
                })

            return bk, goals, None

        except Exception as e:
            return bk, [], str(e)


async def run_generation(args) -> int:
    """Main async generation loop."""
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Load taxonomy
    if args.subcategories:
        subcat_ids = [s.strip() for s in args.subcategories.split(",")]
        subcategories = filter_subcategories(subcat_ids)
        if not subcategories:
            print(f"Error: No subcategories found: {args.subcategories}", file=sys.stderr)
            return 1
    else:
        subcategories = get_all_subcategories()

    output_path = Path(args.output)
    max_seeds = args.max_seeds  # None means use all 20
    seed_index = args.seed_index  # None means use all (or max_seeds)
    batches_per_domain = max(1, math.ceil(args.total_per_subcategory / (20 * args.n_per_batch)))

    # Resume support
    if args.resume:
        all_goals, completed_keys = load_existing_goals(output_path)
        print(f"Resuming: {len(all_goals)} existing goals, {len(completed_keys)} completed batches")
    else:
        all_goals = []
        completed_keys = set()

    # Calculate expected totals (based on actual seeds used, not all 20)
    # We'll compute per-subcategory below, but estimate here for display
    if seed_index is not None:
        est_seeds = 1
    elif max_seeds:
        est_seeds = max_seeds
    else:
        est_seeds = 20
    total_batches = len(subcategories) * est_seeds * batches_per_domain
    remaining_batches = total_batches - len(completed_keys)

    print(f"Subcategories:        {len(subcategories)} — [{', '.join(s['subcategory_id'] for s in subcategories)}]")
    if seed_index is not None:
        print(f"Seed index:           {seed_index} (single seed)")
    else:
        print(f"Seeds per subcat:     {est_seeds}{' (limited by --max-seeds)' if max_seeds else ''}")
    print(f"Batches per domain:   {batches_per_domain}")
    print(f"Goals per batch:      {args.n_per_batch}")
    print(f"Total batches:        {total_batches}")
    print(f"Remaining batches:    {remaining_batches}")
    print(f"Target total goals:   {args.total_per_subcategory * len(subcategories)}")
    print(f"Model:                {args.model}")
    print(f"Max concurrent:       {args.max_concurrent}")
    print(f"Output:               {output_path}")
    print(f"Resume:               {args.resume}")
    print("=" * 80)

    semaphore = asyncio.Semaphore(args.max_concurrent)
    successes = 0
    failures = []

    # Process sequentially by subcategory to manage dedup context
    for subcat in subcategories:
        subcat_id = subcat["subcategory_id"]
        seeds = get_seeds(subcat_id)
        if seed_index is not None:
            if seed_index >= len(seeds):
                print(f"  ⚠️  Seed index {seed_index} out of range (max {len(seeds)-1}), skipping {subcat_id}")
                continue
            seeds = [(seed_index, seeds[seed_index])]
            print(f"  (Using only seed {seed_index}: {seeds[0][1]})")
        else:
            seeds = list(enumerate(seeds))
            if max_seeds:
                seeds = seeds[:max_seeds]

        # Collect goals for this subcategory (for dedup tracking)
        # Group existing goals by domain_index
        domain_goals: Dict[int, List[str]] = {}
        for g in all_goals:
            if g["subcategory_id"] == subcat_id:
                di = g["domain_index"]
                if di not in domain_goals:
                    domain_goals[di] = []
                domain_goals[di].append(g["goal_text"])

        print(f"\n--- Subcategory {subcat_id}: {subcat['subcategory_name']} ---")

        for domain_idx, seed in seeds:
            for batch_round in range(batches_per_domain):
                bk = batch_key(subcat_id, domain_idx, batch_round)
                if bk in completed_keys:
                    continue

                # Get dedup goals: previous batches of this same domain
                dedup = domain_goals.get(domain_idx, []) if domain_goals.get(domain_idx) else None

                bk_result, new_goals, error = await generate_batch(
                    client=client,
                    subcategory=subcat,
                    domain_seed=seed,
                    domain_index=domain_idx,
                    batch_round=batch_round,
                    n_per_batch=args.n_per_batch,
                    model=args.model,
                    dedup_goals=dedup,
                    semaphore=semaphore,
                )

                if error:
                    failures.append({"batch_key": bk, "error": error})
                    print(f"  ❌ {bk} ({seed}): {error}")
                else:
                    successes += 1
                    all_goals.extend(new_goals)

                    # Update domain dedup tracking
                    if domain_idx not in domain_goals:
                        domain_goals[domain_idx] = []
                    domain_goals[domain_idx].extend(g["goal_text"] for g in new_goals)

                    print(f"  ✅ {bk} ({seed}): +{len(new_goals)} goals (total: {len(all_goals)})")

                    # Save after each batch for crash safety
                    save_goals(all_goals, output_path)

                # Small delay between calls
                await asyncio.sleep(0.2)

    await client.close()

    # Assign sequential goal_ids
    subcat_counters: Dict[str, int] = {}
    for goal in all_goals:
        sid = goal["subcategory_id"]
        if sid not in subcat_counters:
            subcat_counters[sid] = 0
        subcat_counters[sid] += 1
        goal["goal_id"] = f"{sid}_{subcat_counters[sid]:03d}"

    # Final save
    save_goals(all_goals, output_path)

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total goals:          {len(all_goals)}")
    print(f"Successful batches:   {successes}")
    print(f"Failed batches:       {len(failures)}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f['batch_key']}: {f['error']}")

    # Per-subcategory breakdown
    print("\nPer subcategory:")
    for subcat in subcategories:
        count = sum(1 for g in all_goals if g["subcategory_id"] == subcat["subcategory_id"])
        print(f"  {subcat['subcategory_id']:4s} {subcat['subcategory_name']:45s} {count} goals")

    print(f"\nSaved to: {output_path}")
    print("=" * 80)

    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial goals v2 using OpenRouter API"
    )
    parser.add_argument(
        "--n-per-batch",
        type=int,
        default=7,
        help="Goals per API call (default: 7)",
    )
    parser.add_argument(
        "--total-per-subcategory",
        type=int,
        default=280,
        help="Total goals per subcategory (default: 280)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="OpenRouter model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "results" / "goals_generated.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--subcategories",
        type=str,
        help="Comma-separated subcategory IDs to generate (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping completed batches",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit number of domain seeds per subcategory (default: all 20). Useful for testing.",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="Run only a specific seed index (0-based). Overrides --max-seeds.",
    )

    args = parser.parse_args()
    return asyncio.run(run_generation(args))


if __name__ == "__main__":
    exit(main())
