"""Minimal stdlib-only OpenAI-compatible client.

Implements just the surface run_unified.py uses from the `openai` package:
    OpenAI(base_url=..., api_key=...).chat.completions.create(...)
The response is wrapped in attribute-accessible objects so
`r.choices[0].message.tool_calls[0].function.name` etc. work unchanged.
No third-party dependencies (pip is unavailable in this environment).
"""

import json
import urllib.request

DEFAULT_TIMEOUT = 180


def _wrap(x):
    if isinstance(x, dict):
        return _AttrDict(x)
    if isinstance(x, list):
        return [_wrap(i) for i in x]
    return x


class _AttrDict:
    """Attr-style access over a dict; missing keys raise AttributeError."""

    __slots__ = ("_d",)

    def __init__(self, d):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name):
        d = object.__getattribute__(self, "_d")
        if name in d:
            return _wrap(d[name])
        raise AttributeError(name)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_d"))


class _Completions:
    def __init__(self, base_url, api_key, timeout=DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def create(self, model=None, messages=None, tools=None, temperature=None, **kwargs):
        body = {"model": model, "messages": messages or [], "temperature": temperature}
        if tools:
            body["tools"] = tools
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _wrap(data)


class OpenAI:
    """OpenAI-compatible API client (base_url + api_key)."""

    def __init__(self, base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT, **kwargs):
        self.base_url = base_url or os_environ("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os_environ("OPENAI_API_KEY", "")
        self.chat = _Namespace(completions=_Completions(self.base_url, self.api_key, timeout))


class AzureOpenAI(OpenAI):
    """Azure-flavoured alias; only used if the azure runner is enabled."""

    def __init__(self, azure_endpoint=None, api_key=None, api_version=None, timeout=DEFAULT_TIMEOUT, **kwargs):
        base = azure_endpoint.rstrip("/") if azure_endpoint else self._default()
        super().__init__(base_url=base, api_key=api_key, timeout=timeout)

    def _default(self):
        raise ValueError("AzureOpenAI requires azure_endpoint")


class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def os_environ(name, default=""):
    import os
    return os.environ.get(name, default)