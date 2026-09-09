"""Feature extraction for the 19-feature trajectory signature (paper Sec 2.3).

Tools are reduced to short tokens:
    list    -> memory.list_all_facts
    recall  -> memory.recall_fact
    save    -> memory.save_fact
    draft   -> email.draft_email
    send    -> email.send_email
    rag     -> rag.retrieve

All features are computed on the TRIGGER session only, from the ordered tool-call
sequence (already sorted by execution timestamp by the caller).
"""

from typing import List, Sequence

from collections import Counter

TOKENS = ("list", "recall", "save", "draft", "send", "rag")

FEATURE_NAMES = [
    # Count features (7)
    "recall_count", "list_count", "save_count", "draft_count",
    "send_count", "rag_count", "seq_len",
    # Structural features (4)
    "send_without_recall", "recall_before_send", "recall_to_send_ratio", "max_recall_chain",
    # Bigram transitions (5)
    "list_then_recall", "recall_then_recall", "recall_then_draft",
    "list_then_draft", "draft_then_send",
    # Entry point (3)
    "first_tool_is_list", "first_tool_is_recall", "first_tool_is_draft",
]

COUNT_FEATURES = FEATURE_NAMES[:6]           # 6 per-tool counts (seq_len handled separately)
FREQUENCY_GROUP = FEATURE_NAMES[:7]          # Count group used in ablation (7 features)
MECHANISTIC_GROUP = ["recall_before_send", "send_without_recall"]
RATIO_GROUP = ["recall_to_send_ratio", "max_recall_chain"]
BIGRAM_GROUP = ["list_then_recall", "recall_then_recall", "recall_then_draft",
                "list_then_draft", "draft_then_send"]
FIRST_TOOL_GROUP = ["first_tool_is_list", "first_tool_is_recall", "first_tool_is_draft"]
RECALL_RELATED_GROUP = [  # all 9 recall-related features (paper Sec 3.6)
    "recall_before_send", "send_without_recall", "recall_count",
    "recall_to_send_ratio", "max_recall_chain",
    "recall_then_recall", "recall_then_draft", "list_then_recall", "first_tool_is_recall",
]

# Prefix-only variant: full 19 minus the 6 features that need the send phase or
# the completed session (paper Sec 3.9).
PREFIX_13_FEATURES = [
    "recall_count", "list_count", "save_count", "draft_count", "rag_count",
    "max_recall_chain",
    "list_then_recall", "recall_then_recall", "recall_then_draft", "list_then_draft",
    "first_tool_is_list", "first_tool_is_recall", "first_tool_is_draft",
]

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def max_recall_chain(seq: Sequence[str]) -> int:
    longest = cur = 0
    for t in seq:
        if t == "recall":
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def extract_features(seq: Sequence[str]) -> List[float]:
    """Return the 19 features as a float vector, in FEATURE_NAMES order.

    Timestamp ordering is the caller's responsibility; transition statistics
    must reflect actual execution order (paper Sec 2.3 verification step).
    """
    cnt = Counter(seq)
    n = len(seq)

    recall_cnt = cnt.get("recall", 0)
    send_cnt = cnt.get("send", 0)

    first_recall = seq.index("recall") if "recall" in seq else None
    first_send = seq.index("send") if "send" in seq else None

    send_without_recall = 1.0 if (first_send is not None and first_recall is None) else 0.0
    recall_before_send = 1.0 if (first_recall is not None and first_send is not None
                                 and first_recall < first_send) else 0.0

    bigrams = Counter(zip(seq, seq[1:]))
    bigram = lambda a, b: float(bigrams.get((a, b), 0))

    first = seq[0] if seq else None

    feats = [
        float(recall_cnt),
        float(cnt.get("list", 0)),
        float(cnt.get("save", 0)),
        float(cnt.get("draft", 0)),
        float(send_cnt),
        float(cnt.get("rag", 0)),
        float(n),
        send_without_recall,
        recall_before_send,
        float(recall_cnt / max(send_cnt, 1)),
        float(max_recall_chain(seq)),
        bigram("list", "recall"),
        bigram("recall", "recall"),
        bigram("recall", "draft"),
        bigram("list", "draft"),
        bigram("draft", "send"),
        1.0 if first == "list" else 0.0,
        1.0 if first == "recall" else 0.0,
        1.0 if first == "draft" else 0.0,
    ]
    assert len(feats) == 19, len(feats)
    return feats