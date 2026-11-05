"""Application settings using Pydantic BaseSettings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from edcraft_backend.config.environments import Environment


class AppSettings(BaseSettings):
    """Application metadata. Env prefix: APP_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="APP_", case_sensitive=False, extra="ignore"
    )

    env: Environment = Field(
        default=Environment.DEVELOPMENT, description="Application environment"
    )
    name: str = Field(default="EdCraft Backend API", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")


class DatabaseSettings(BaseSettings):
    """Database configuration. Env prefix: DATABASE_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="DATABASE_", case_sensitive=False, extra="ignore"
    )

    url: PostgresDsn | None = Field(
        default=None, description="PostgreSQL database URL (required via environment)"
    )
    echo: bool | None = Field(default=None, description="Echo SQL queries to logs")

    @field_validator("url", mode="after")
    @classmethod
    def validate_url(cls, v: PostgresDsn | None) -> PostgresDsn:
        """Ensure database URL is provided via environment variables."""
        if v is None:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set it in your .env.{APP_ENV} file or as a system environment variable."
            )
        return v

    @field_validator("echo", mode="before")
    @classmethod
    def set_echo(cls, v: Any) -> bool:
        """Auto-enable database echo in development if not explicitly set."""
        if v is not None:
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes", "on")
            return bool(v)
        # load_env_files() runs before Settings(), so APP_ENV is already set
        return os.getenv("APP_ENV", "development") == "development"


class CorsSettings(BaseSettings):
    """CORS configuration. Env prefix: CORS_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="CORS_", case_sensitive=False, extra="ignore"
    )

    origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins",
    )
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )


class ServerSettings(BaseSettings):
    """Server configuration. Env prefix: SERVER_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="SERVER_", case_sensitive=False, extra="ignore"
    )

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")


class JwtSettings(BaseSettings):
    """JWT configuration. Env prefix: JWT_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="JWT_", case_sensitive=False, extra="ignore"
    )

    secret: str = Field(default="change-me", description="Generate with: openssl rand -hex 32")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=14)
    issuer: str = Field(default="edcraft")
    audience: str = Field(default="edcraft")
    kid: str = Field(default="1")


class OAuthGoogleSettings(BaseSettings):
    """Google OAuth configuration. Env prefix: OAUTH_GOOGLE_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="OAUTH_GOOGLE_", case_sensitive=False, extra="ignore"
    )

    client_id: str | None = Field(default=None)
    client_secret: str | None = Field(default=None)
    redirect_uri: str = Field(
        default="http://localhost:8000/auth/oauth/google/callback"
    )


class OAuthGithubSettings(BaseSettings):
    """GitHub OAuth configuration. Env prefix: OAUTH_GITHUB_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="OAUTH_GITHUB_", case_sensitive=False, extra="ignore"
    )

    client_id: str | None = Field(default=None)
    client_secret: str | None = Field(default=None)
    redirect_uri: str = Field(
        default="http://localhost:8000/auth/oauth/github/callback"
    )


class NomadSettings(BaseSettings):
    """Nomad job queue configuration."""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="NOMAD_", case_sensitive=False, extra="ignore"
    )

    host: str = Field(default="host.docker.internal", description="Nomad agent host")
    port: int = Field(default=4646, description="Nomad HTTP API port")
    token: str | None = Field(default=None, description="Nomad ACL token")
    namespace: str | None = Field(default=None, description="Nomad namespace for job isolation")
    datacenters: list[str] = Field(
        default_factory=lambda: ["dc1"],
        description="Datacenters where jobs can be scheduled",
    )
    cpu_mhz: int = Field(default=500, description="CPU resource limit per worker task (MHz)")
    memory_mb: int = Field(default=512, description="Memory resource limit per worker task (MB)")
    callback_base_url: str = Field(
        default="http://host.docker.internal:8000",
        description=(
            "Base URL the worker uses to POST results back "
            "(must be reachable from inside the worker container)"
        ),
    )
    container_image: str = Field(
        default="ghcr.io/edcraft-org/edcraft-engine:latest",
        description="Engine Docker image for worker jobs",
    )
    registry_username: str | None = Field(
        default=None, description="Registry username for pulling private container images"
    )
    registry_password: str | None = Field(
        default=None, description="Registry password / token for pulling private container images"
    )
    container_workdir: str = "/local"


class EmailSettings(BaseSettings):
    """Email configuration. Env prefix: EMAIL_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="EMAIL_", case_sensitive=False, extra="ignore"
    )

    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_user: str | None = Field(default=None, description="SMTP username")
    smtp_password: str | None = Field(default=None, description="SMTP password")
    from_email: str = Field(
        default="noreply@edcraft.com", description="From email address"
    )
    from_name: str = Field(default="EdCraft", description="From name")
    enabled: bool = Field(
        default=True, description="Enable email sending (set to False for dev mode)"
    )
    verification_token_expire_hours: int = Field(
        default=24, description="Email verification token expiration in hours"
    )


class AuthoringProfileSettings(BaseModel):
    """One server-configured authoring model profile (never credentials)."""

    model_config = {"extra": "forbid", "frozen": True}

    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=120)


class AuthoringSettings(BaseSettings):
    """File-backed AI authoring configuration. Env prefix: AUTHORING_"""

    model_config = SettingsConfigDict(
        env_file=None, env_prefix="AUTHORING_", case_sensitive=False, extra="ignore"
    )

    storage_dir: Path = Field(
        default=Path(".data/authoring"),
        description=(
            "Durable directory for owner-scoped authoring job records. "
            "Docker deployments set an absolute path on a mounted volume."
        ),
    )
    max_active_jobs: int = Field(default=4, ge=1, le=32)
    max_active_jobs_per_user: int = Field(default=2, ge=1, le=32)
    max_inflight_jobs_per_user: int = Field(default=6, ge=1, le=100)
    max_queue: int = Field(default=40, ge=0, le=100)
    max_quantity: int = Field(default=50, ge=1, le=50)
    shutdown_timeout_seconds: float = Field(default=30.0, gt=0)
    executor: Literal["inline", "nomad"] = Field(
        default="inline",
        description="Execution mode for AI authoring: in-process or Nomad batch jobs.",
    )
    worker_token_ttl_seconds: int = Field(
        default=7200,
        gt=0,
        description="Lifetime of a Nomad worker callback token.",
    )
    job_timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="Maximum time to wait for a Nomad authoring worker.",
    )
    nomad_container_image: str | None = Field(
        default=None,
        description="Worker image; falls back to NOMAD_CONTAINER_IMAGE.",
    )
    nomad_callback_base_url: str | None = Field(
        default=None,
        description=(
            "Backend base URL reachable from workers; "
            "falls back to NOMAD_CALLBACK_BASE_URL."
        ),
    )
    nomad_force_pull: bool = Field(
        default=False,
        description="Force Nomad to pull the worker image before starting.",
    )
    model_profiles: list[AuthoringProfileSettings] = Field(default_factory=list)
    encryption_key: SecretStr | None = Field(
        default=None,
        description=(
            "Secret used to encrypt user provider keys at rest. When omitted, "
            "the installation JWT secret is used."
        ),
    )

    @model_validator(mode="after")
    def require_unique_profiles(self) -> "AuthoringSettings":
        ids = [profile.id for profile in self.model_profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("authoring model profile ids must be unique")
        return self


class Settings(BaseSettings):
    """
    Application settings with environment-based configuration.

    Configuration is loaded in the following order (later overrides earlier):
    1. .env.{APP_ENV} (environment-specific)
    2. .env.local (local overrides)
    3. System environment variables (highest priority)
    """

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    jwt: JwtSettings = Field(default_factory=JwtSettings)
    oauth_google: OAuthGoogleSettings = Field(default_factory=OAuthGoogleSettings)
    oauth_github: OAuthGithubSettings = Field(default_factory=OAuthGithubSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    nomad: NomadSettings = Field(default_factory=NomadSettings)
    authoring: AuthoringSettings = Field(default_factory=AuthoringSettings)

    # Standalone settings
    log_level: str = Field(default="info", description="Logging level")
    password_min_length: int = Field(default=12)
    frontend_url: str = Field(
        default="http://localhost:5173", description="Frontend URL for OAuth redirects"
    )
    session_secret: str = Field(
        default="change-me-session", description="Secret key for signing session cookies"
    )

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app.env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app.env == Environment.DEVELOPMENT

    @property
    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.app.env == Environment.TEST


def get_project_root() -> Path:
    """
    Find project root by looking for pyproject.toml.

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def load_env_files(app_env: str | None = None) -> None:
    """
    Load environment files in the correct order.

    Load order (later overrides earlier):
    1. .env.{APP_ENV} (environment-specific configuration)
    2. .env.local (local overrides)
    3. System environment variables (highest priority)

    Args:
        app_env: Application environment. If None, reads from APP_ENV env var.
    """
    from dotenv import load_dotenv

    project_root = get_project_root()
    env = app_env or os.getenv("APP_ENV", "development")

    env_file = project_root / f".env.{env}"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    local_env = project_root / ".env.local"
    if local_env.exists():
        load_dotenv(local_env, override=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to ensure we only load settings once.
    Use this function throughout the application to access settings.

    Returns:
        Settings instance
    """
    load_env_files()
    return Settings()


settings = get_settings()
