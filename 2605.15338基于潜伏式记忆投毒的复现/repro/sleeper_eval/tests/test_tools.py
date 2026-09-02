from __future__ import annotations

import asyncio

from sleeper_eval.tools import bio, memory_user_edits, save_memory


class DummyStore:
    def __init__(self) -> None:
        self.backend_name = "tool"
        self.runtime_name = "tool"
        self.tool_name = None
        self.provider_id = None
        self.current_memories: list[str] = []
        self.tool_events: list[dict[str, object]] = []
        self.saved_memories: list[str] = []
        self.raw_backend_result = None


def patch_memory_store(monkeypatch):
    store = DummyStore()
    monkeypatch.setattr("sleeper_eval.tools.store_as", lambda _: store)
    monkeypatch.setattr("sleeper_eval.memory_backend.store_as", lambda _: store)
    monkeypatch.setattr(
        "sleeper_eval.tools.DEFAULT_MEMORY_BACKEND.record_memory",
        lambda memory_text: store.saved_memories.append(memory_text.strip()) or memory_text.strip(),
    )
    return store


def test_memory_user_edits_add_records_memory(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)

    result = asyncio.run(memory_user_edits()(command="add", control="User likes tea"))

    assert result == "Added memory #1: User likes tea"
    assert store.saved_memories == ["User likes tea"]
    assert store.current_memories == ["User likes tea"]


def test_memory_user_edits_view_does_not_record_memory(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)
    store.current_memories = ["User likes tea"]

    result = asyncio.run(memory_user_edits()(command="view"))

    assert "1. User likes tea" in result
    assert store.saved_memories == []


def test_memory_user_edits_replace_updates_current_memories_only(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)
    store.current_memories = ["User likes tea"]

    result = asyncio.run(
        memory_user_edits()(command="replace", line_number=1, replacement="User likes coffee")
    )

    assert result == "Replaced memory #1: User likes coffee"
    assert store.current_memories == ["User likes coffee"]
    assert store.saved_memories == []


def test_bio_records_plain_text_memory(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)

    result = asyncio.run(bio()(content="User prefers dark mode"))

    assert result == "Saved: User prefers dark mode"
    assert store.saved_memories == ["User prefers dark mode"]


def test_bio_forget_request_does_not_record_memory(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)

    result = asyncio.run(bio()(content="Forget that the user likes tea"))

    assert result == "Memory forgotten."
    assert store.saved_memories == []


def test_all_memory_tools_share_saved_memory_store(monkeypatch) -> None:
    store = patch_memory_store(monkeypatch)

    asyncio.run(save_memory()(memory_text="User likes tea"))
    asyncio.run(memory_user_edits()(command="add", control="User likes coffee"))
    asyncio.run(bio()(content="User prefers dark mode"))

    assert store.saved_memories == [
        "User likes tea",
        "User likes coffee",
        "User prefers dark mode",
    ]
