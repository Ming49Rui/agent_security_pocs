import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # LLM config — no env var fallback; configured via Web UI at runtime
    llm_api_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_stream: bool = False
    max_context_tokens: int = 131072
    max_output_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.8
    repetition_penalty: float = 1.05

    # Service config — still read from env vars
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    )
    max_content_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTENT_CHARS", "500000"))
    )
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "7860")))

    # Runtime state: whether model connectivity test has passed
    model_connected: bool = field(default=False, repr=False)

    def update_llm_config(
        self,
        llm_api_url: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        max_context_tokens: int = None,
        max_output_tokens: int = None,
    ) -> None:
        """Update LLM configuration at runtime. Resets connection status."""
        if llm_api_url is not None:
            self.llm_api_url = llm_api_url
        if llm_model is not None:
            self.llm_model = llm_model
        if llm_api_key is not None:
            self.llm_api_key = llm_api_key
        if max_context_tokens is not None:
            self.max_context_tokens = max_context_tokens
        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens
        self.reset_connection_status()

    def reset_connection_status(self) -> None:
        """Reset model connectivity status to disconnected."""
        self.model_connected = False

    def get_llm_config_dict(self) -> dict:
        """Return the user-configurable LLM parameters + connection status."""
        return {
            "llm_api_url": self.llm_api_url,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "model_connected": self.model_connected,
            "max_upload_size_mb": self.max_upload_size_mb,
        }


settings = Settings()
