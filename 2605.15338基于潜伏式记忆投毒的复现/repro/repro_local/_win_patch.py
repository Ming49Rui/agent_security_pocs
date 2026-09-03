"""针对 inspect-ai 在 fsspec 日志路径上的 Windows 兼容补丁。

在 Windows 上，inspect-ai 会把日志文件路径规范成 fsspec 的规范形式
'file:/C:/Users/...'（单斜杠）。inspect 自带的 local_path() 只处理
'file://'（双斜杠），于是原始 'file:/...' 字符串会原样进入 atomic_write()，
其中 Path('file:/...').parent 在 Windows 上会塌缩成 'file:'，随后
os.makedirs() 报 WinError 123。

本补丁在 eval.py 绑定 local_path 的命名空间里扩展它，使其同样处理单斜杠
形式，让后续所有日志写入/读取都拿到普通 Windows 路径。仅在本地启动器中
生效，不改动上游代码。
"""

from __future__ import annotations

import inspect_ai.log._recorders.eval as _ev


def _patch() -> None:
    if getattr(_ev.local_path, "_zcode_win_patched", False):
        return
    _orig = _ev.local_path

    def _patched(filename: str) -> str:
        lower = filename.lower()
        if lower.startswith("file:/") and not lower.startswith("file://"):
            rest = filename[len("file:/"):]
            if rest.startswith("/"):
                rest = rest[1:]
            if rest:
                return rest
        return _orig(filename)

    _patched._zcode_win_patched = True  # type: ignore[attr-defined]
    _ev.local_path = _patched


_patch()
