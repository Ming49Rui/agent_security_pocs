#!/usr/bin/env python3
"""
Validate generated goals v2 for quality and correctness.

Checks (from spec):
  1. Every goal starts with "The user"
  2. No meta-language (pattern-based: "inject", "adversary", "trick", "attack" not in medical context, "manipulate" not about prices/markets)
  3. No instruction language ("should" without belief verb, "must", "make the user", "force the user")
  4. Near-duplicate detection within subcategory (Jaccard similarity)
  5. Distribution across seed dimensions (no single domain dominating)
  6. Tag distribution balance (post-tagging)
  7. Category 1 → adversary_type=commercial, Category 2 → adversary_type=political
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Individual goal validation
# ---------------------------------------------------------------------------

# Meta-language patterns that indicate the goal is about the attack itself, not adversarial content
META_PATTERNS = [
    r"\binject\b",  # "inject" as a standalone word
    r"\badversary\b",  # "adversary" as a standalone word
    r"\btrick\b",  # "trick" as a standalone word
    # "attack" only if not part of medical terms
    r"\battack\b(?!\s+(risk|rate|symptoms|prevention|treatment|victim|victims))",
    # "manipulate" only if clearly about manipulating the user/system directly
    # Allow: "manipulates prices", "manipulates public opinion", "manipulates data" (adversarial content)
    # Flag: "manipulate the user", "manipulate the system" (meta-language)
    r"\bmanipulate\b\s+(the\s+)?(user|system|model|llm|ai)",
]

# Instruction keywords - but only flag if not preceded by belief verbs
BELIEF_VERBS = ["believes", "thinks", "supports", "opposes", "prefers", "trusts", "considers", "wants"]
INSTRUCTION_KEYWORDS = ["must", "make the user", "force the user"]


def validate_goal(goal: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Basic validation of a single goal entry."""
    issues = []

    # Required fields
    for field in ["goal_id", "goal_text", "subcategory_id"]:
        if field not in goal:
            issues.append(f"Missing field: {field}")

    goal_text = goal.get("goal_text", "")
    if not goal_text:
        return False, issues

    # Check 1: starts with "The user"
    if not goal_text.strip().startswith("The user"):
        issues.append("Does not start with 'The user'")

    # Check 2: meta-language (using pattern matching, not simple substring)
    goal_lower = goal_text.lower()
    found_meta = []
    for pattern in META_PATTERNS:
        if re.search(pattern, goal_lower):
            # Extract the matched word for reporting
            match = re.search(pattern, goal_lower)
            if match:
                found_meta.append(match.group(0))
    if found_meta:
        issues.append(f"Meta-language: {', '.join(found_meta)}")

    # Check 3: instruction language
    # Check for "should" only if not preceded by a belief verb
    has_belief_verb = any(verb in goal_lower for verb in BELIEF_VERBS)
    if "should" in goal_lower and not has_belief_verb:
        issues.append("Instruction language: should (without belief verb)")
    
    # Check "must" - allow medical/necessity/legal contexts
    # These are legitimate: "must avoid gluten", "must take medication", "must comply with"
    # Only flag if it's clearly an instruction to the user
    if "must" in goal_lower:
        # Allow if it's about requirements/necessities (avoid, take, use, have, get, undergo, follow, comply, meet, maintain)
        # Allow if it's about conditions (has X and must Y)
        if not (re.search(r"\bmust\s+(avoid|take|use|have|get|undergo|follow|comply|meet|maintain)", goal_lower) or
                re.search(r"\band\s+must\s+", goal_lower)):
            issues.append("Instruction language: must")
    
    # Check other instruction keywords
    found_instr = [kw for kw in INSTRUCTION_KEYWORDS if kw in goal_lower]
    if found_instr:
        issues.append(f"Instruction language: {', '.join(found_instr)}")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Similarity checks
# ---------------------------------------------------------------------------

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts (word-level)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def check_duplicates_within_subcategory(
    goals: List[Dict], threshold: float = 0.5
) -> List[Dict]:
    """Check for near-duplicates within the same subcategory."""
    warnings = []

    by_subcat: Dict[str, List[Dict]] = {}
    for g in goals:
        sid = g.get("subcategory_id", "?")
        by_subcat.setdefault(sid, []).append(g)

    for sid, subcat_goals in by_subcat.items():
        for i, g1 in enumerate(subcat_goals):
            for g2 in subcat_goals[i + 1 :]:
                sim = jaccard_similarity(
                    g1.get("goal_text", ""), g2.get("goal_text", "")
                )
                if sim >= threshold:
                    warnings.append({
                        "type": "duplicate_within_subcategory",
                        "subcategory_id": sid,
                        "goal1_id": g1.get("goal_id"),
                        "goal2_id": g2.get("goal_id"),
                        "similarity": round(sim, 3),
                        "goal1_text": g1.get("goal_text", "")[:80],
                        "goal2_text": g2.get("goal_text", "")[:80],
                    })

    return warnings


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------

def check_seed_distribution(goals: List[Dict]) -> List[Dict]:
    """
    Check that goals are reasonably distributed across domain seeds.
    Flag if any single domain has > 2x the expected count.
    """
    warnings = []

    by_subcat: Dict[str, Counter] = {}
    for g in goals:
        sid = g.get("subcategory_id", "?")
        domain = g.get("domain_seed", "unknown")
        by_subcat.setdefault(sid, Counter())[domain] += 1

    for sid, domain_counts in by_subcat.items():
        if not domain_counts:
            continue
        total = sum(domain_counts.values())
        expected_per_domain = total / len(domain_counts)

        for domain, count in domain_counts.items():
            if count > expected_per_domain * 2.5:
                warnings.append({
                    "type": "uneven_seed_distribution",
                    "subcategory_id": sid,
                    "domain_seed": domain,
                    "count": count,
                    "expected": round(expected_per_domain, 1),
                })

    return warnings


def check_tag_distribution(goals: List[Dict]) -> List[Dict]:
    """
    Check tag distribution balance (post-tagging).
    Flag if any tag value > 50% within a subcategory.
    """
    warnings = []
    tag_keys = ["goal_type", "adversary_type", "severity"]

    by_subcat: Dict[str, List[Dict]] = {}
    for g in goals:
        sid = g.get("subcategory_id", "?")
        by_subcat.setdefault(sid, []).append(g)

    for sid, subcat_goals in by_subcat.items():
        tagged = [g for g in subcat_goals if "tags" in g]
        if len(tagged) < 10:
            continue

        for tag_key in tag_keys:
            counts = Counter(g["tags"].get(tag_key, "untagged") for g in tagged)
            total = sum(counts.values())
            for val, count in counts.items():
                pct = count / total
                if pct > 0.5:
                    warnings.append({
                        "type": "tag_imbalance",
                        "subcategory_id": sid,
                        "tag": tag_key,
                        "value": val,
                        "percentage": round(pct * 100, 1),
                        "count": count,
                        "total": total,
                    })

    return warnings


def check_adversary_type_consistency(goals: List[Dict]) -> List[Dict]:
    """Check Cat 1 goals → commercial, Cat 2 → political."""
    warnings = []

    for g in goals:
        tags = g.get("tags")
        if not tags:
            continue

        sid = g.get("subcategory_id", "")
        adv_type = tags.get("adversary_type")

        if sid.startswith("1") and adv_type != "commercial":
            warnings.append({
                "type": "adversary_type_mismatch",
                "goal_id": g.get("goal_id"),
                "subcategory_id": sid,
                "expected": "commercial",
                "got": adv_type,
            })
        elif sid.startswith("2") and adv_type != "political":
            warnings.append({
                "type": "adversary_type_mismatch",
                "goal_id": g.get("goal_id"),
                "subcategory_id": sid,
                "expected": "political",
                "got": adv_type,
            })

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate generated goals v2 for quality and correctness"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file with goals",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Jaccard similarity threshold for flagging duplicates (default: 0.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        goals = json.load(f)

    if not isinstance(goals, list):
        print("Error: Input must be a JSON array", file=sys.stderr)
        return 1

    # Per-goal validation
    invalid_goals = []
    total_issues = 0
    for g in goals:
        is_valid, issues = validate_goal(g)
        if not is_valid:
            invalid_goals.append({
                "goal_id": g.get("goal_id", "?"),
                "issues": issues,
                "goal_text": g.get("goal_text", "")[:80],
            })
            total_issues += len(issues)

    # Duplicate check
    dup_warnings = check_duplicates_within_subcategory(goals, args.similarity_threshold)

    # Distribution checks
    seed_warnings = check_seed_distribution(goals)
    tag_warnings = check_tag_distribution(goals)
    adv_warnings = check_adversary_type_consistency(goals)

    all_warnings = dup_warnings + seed_warnings + tag_warnings + adv_warnings

    # Count stats
    has_tags = sum(1 for g in goals if "tags" in g)
    by_subcat = Counter(g.get("subcategory_id", "?") for g in goals)

    output = {
        "total_goals": len(goals),
        "valid_goals": len(goals) - len(invalid_goals),
        "invalid_goals": len(invalid_goals),
        "total_issues": total_issues,
        "goals_with_tags": has_tags,
        "goals_without_tags": len(goals) - has_tags,
        "by_subcategory": dict(by_subcat.most_common()),
        "warnings": {
            "duplicates": len(dup_warnings),
            "seed_distribution": len(seed_warnings),
            "tag_balance": len(tag_warnings),
            "adversary_type": len(adv_warnings),
            "total": len(all_warnings),
        },
        "invalid_goal_details": invalid_goals[:20],
        "warning_details": all_warnings[:30],
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print("=" * 80)
        print("GOALS V2 — VALIDATION RESULTS")
        print("=" * 80)
        print(f"Total goals:          {len(goals)}")
        print(f"Valid:                {output['valid_goals']}")
        print(f"Invalid:              {len(invalid_goals)}")
        print(f"With tags:            {has_tags}")
        print(f"Without tags:         {len(goals) - has_tags}")

        print(f"\nPer subcategory:")
        for sid, count in sorted(by_subcat.items()):
            print(f"  {sid:4s}  {count} goals")

        print(f"\nWarnings:")
        print(f"  Near-duplicates:    {len(dup_warnings)}")
        print(f"  Seed distribution:  {len(seed_warnings)}")
        print(f"  Tag imbalance:      {len(tag_warnings)}")
        print(f"  Adversary type:     {len(adv_warnings)}")

        if invalid_goals:
            print(f"\nInvalid goals (showing first 20):")
            for ig in invalid_goals[:20]:
                print(f"  {ig['goal_id']}: {ig['issues']}")
                print(f"    \"{ig['goal_text']}...\"")

        if dup_warnings:
            print(f"\nNear-duplicate pairs (showing first 10):")
            for w in dup_warnings[:10]:
                print(f"  {w['goal1_id']} <-> {w['goal2_id']} (sim={w['similarity']:.2f})")
                print(f"    \"{w['goal1_text']}...\"")
                print(f"    \"{w['goal2_text']}...\"")

        if seed_warnings:
            print(f"\nSeed distribution warnings:")
            for w in seed_warnings:
                print(f"  {w['subcategory_id']} / {w['domain_seed']}: {w['count']} goals (expected ~{w['expected']})")

        if tag_warnings:
            print(f"\nTag imbalance warnings:")
            for w in tag_warnings:
                print(f"  {w['subcategory_id']} {w['tag']}={w['value']}: {w['percentage']}% ({w['count']}/{w['total']})")

        if adv_warnings:
            print(f"\nAdversary type mismatches (showing first 10):")
            for w in adv_warnings[:10]:
                print(f"  {w['goal_id']}: expected {w['expected']}, got {w['got']}")

        print("=" * 80)

        # Exit code
        if invalid_goals or all_warnings:
            print(f"\n⚠️  {len(invalid_goals)} invalid goals, {len(all_warnings)} warnings")
        else:
            print(f"\n✅ All {len(goals)} goals passed validation")

    return 0 if not invalid_goals and not all_warnings else 1


if __name__ == "__main__":
    exit(main())
