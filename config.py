"""
Central config, loaded once from environment variables.
Keeping every setting in one place means main.py, auth.py and risk_engine.py
never hardcode a secret or a magic number — they all import `settings`.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./simguard.db")

    # In real deployment this MUST come from an environment variable / secrets
    # manager, never a hardcoded default. The fallback below only exists so
    # the app can run locally without extra setup.
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-in-prod")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    OTP_EXPIRE_MINUTES: int = 5  # how long a post-login OTP token stays valid

    # Risk thresholds — the boundaries that turn a numeric score into a decision.
    RISK_ALLOW_BELOW: int = 35
    RISK_BLOCK_AT_OR_ABOVE: int = 70

    # Carrier SIM-swap API (see vonage_client.py). Empty = not connected; the
    # client then raises NotImplementedError instead of calling out.
    VONAGE_APPLICATION_ID: str = os.getenv("VONAGE_APPLICATION_ID", "")
    VONAGE_PRIVATE_KEY_PATH: str = os.getenv("VONAGE_PRIVATE_KEY_PATH", "")
    VONAGE_API_BASE_URL: str = os.getenv("VONAGE_API_BASE_URL", "")

    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_MINUTES: int = 15


settings = Settings()
