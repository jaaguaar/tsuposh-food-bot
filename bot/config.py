from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bot_token: str
    log_level: str = "INFO"

    # Azure OpenAI / Azure AI Foundry (OpenAI SDK, v1 endpoint)
    azure_openai_base_url: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None

    # Backward-compatible aliases (old config names)
    azure_ai_inference_endpoint: str | None = None
    azure_ai_inference_key: str | None = None
    azure_ai_model_deployment: str | None = None
    azure_ai_api_version: str = "2024-05-01-preview"

    use_ai_manager: bool = Field(default=True)

    # Optional model defaults (used by wrapper if caller doesn't override)
    azure_openai_temperature: float = 0.15
    azure_openai_max_tokens: int = 450

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_ai_base_url(self) -> str | None:
        # Prefer new Azure OpenAI v1 endpoint; fallback to legacy inference endpoint if provided.
        return self.azure_openai_base_url or self.azure_ai_inference_endpoint

    @property
    def resolved_ai_api_key(self) -> str | None:
        return self.azure_openai_api_key or self.azure_ai_inference_key

    @property
    def resolved_ai_deployment(self) -> str | None:
        return self.azure_openai_deployment or self.azure_ai_model_deployment


settings = Settings()
