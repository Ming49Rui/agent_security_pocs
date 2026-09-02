"""Custom tools for sleeper memory evaluation."""

from __future__ import annotations

from inspect_ai.util import store_as

from inspect_ai.tool import tool

from sleeper_eval.memory_backend import DEFAULT_MEMORY_BACKEND, SavedMemoryStore


def _store_for_tool(tool_name: str) -> SavedMemoryStore:
    store = store_as(SavedMemoryStore)
    store.tool_name = tool_name
    return store


def _record_memory(memory_text: str, *, tool_name: str) -> str:
    store = _store_for_tool(tool_name)
    cleaned = DEFAULT_MEMORY_BACKEND.record_memory(memory_text)
    store.current_memories.append(cleaned)
    store.tool_events.append(
        {"tool_name": tool_name, "action": "add", "content": cleaned}
    )
    return cleaned


def _format_numbered_memories(memories: list[str]) -> str:
    if not memories:
        return "Viewed memory edits:\nNo memory edits found."
    lines = [f"{index}. {memory}" for index, memory in enumerate(memories, start=1)]
    return "Viewed memory edits:\n" + "\n".join(lines)


@tool
def save_memory():
    """Tool for saving a new user memory."""

    async def execute(memory_text: str) -> str:
        """Save one short memory about the user.

        Args:
            memory_text: Concise fact or preference to remember about the user.

        Returns:
            Confirmation that the memory was saved.
        """

        cleaned = _record_memory(memory_text, tool_name="save_memory")
        return f"Saved memory: {cleaned}"

    return execute


@tool
def memory_user_edits():
    """Claude-style structured memory editing tool."""

    async def execute(
        command: str,
        control: str | None = None,
        line_number: int | None = None,
        replacement: str | None = None,
    ) -> str:
        """Manage remembered user facts with structured edit commands.

        Args:
            command: One of view, add, remove, or replace.
            control: New memory text to add when command is add.
            line_number: One-indexed memory line to remove or replace.
            replacement: Replacement memory text when command is replace.

        Returns:
            A confirmation string describing the applied memory edit.
        """
        store = _store_for_tool("memory_user_edits")

        if command == "view":
            store.tool_events.append({"tool_name": "memory_user_edits", "action": "view"})
            return _format_numbered_memories(store.current_memories)

        if command == "add":
            if not control or not control.strip():
                return "Invalid command or missing parameters."
            cleaned = _record_memory(control, tool_name="memory_user_edits")
            return f"Added memory #{len(store.current_memories)}: {cleaned}"

        if command == "remove":
            if line_number is None or not 1 <= line_number <= len(store.current_memories):
                return "Invalid command or missing parameters."
            removed = store.current_memories.pop(line_number - 1)
            store.tool_events.append(
                {
                    "tool_name": "memory_user_edits",
                    "action": "remove",
                    "line_number": line_number,
                    "content": removed,
                }
            )
            return f"Removed memory #{line_number}: {removed}"

        if command == "replace":
            if (
                line_number is None
                or not replacement
                or not replacement.strip()
                or not 1 <= line_number <= len(store.current_memories)
            ):
                return "Invalid command or missing parameters."
            cleaned = replacement.strip()
            store.current_memories[line_number - 1] = cleaned
            store.tool_events.append(
                {
                    "tool_name": "memory_user_edits",
                    "action": "replace",
                    "line_number": line_number,
                    "content": cleaned,
                }
            )
            return f"Replaced memory #{line_number}: {cleaned}"

        return "Invalid command or missing parameters."

    return execute


@tool
def bio():
    """GPT-style plain-text memory tool."""

    async def execute(content: str) -> str:
        """Persist one plain-text memory instruction.

        Args:
            content: A plain-text memory statement beginning with User or Forget.

        Returns:
            A confirmation that the memory was saved or a forget request was processed.
        """
        store = _store_for_tool("bio")
        cleaned_content = content.strip()
        if cleaned_content.lower().startswith("forget"):
            store.tool_events.append(
                {"tool_name": "bio", "action": "forget", "content": cleaned_content}
            )
            return "Memory forgotten."

        cleaned = _record_memory(cleaned_content, tool_name="bio")
        return f"Saved: {cleaned}"

    return execute
