"""
Carrier SIM-swap check client (Vonage-style).

*** UNVERIFIED, BEST-EFFORT IMPLEMENTATION — ARCHITECTURE, NOT A VERIFIED
*** INTEGRATION. The request/response shape below (endpoint path, JSON fields,
*** JWT claims) is what a carrier SIM-swap API is *expected* to look like. It
*** has NOT been tested against the real Vonage API, because no credentials
*** were available. Check it against Vonage's current docs before relying on it.

Why this exists:
  The risk engine currently treats `user.sim_swapped_at` as the SIM-swap
  signal, and the admin "simulate swap" button sets it by hand. In production
  that timestamp would come from the carrier. This module is the seam where
  that real lookup would plug in.

Status: standalone. It is deliberately NOT wired into risk_engine.py or
main.py yet.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import httpx
from jose import jwt

from config import settings

REQUEST_TIMEOUT_SECONDS = 10


def _missing_settings() -> list:
    required = {
        "VONAGE_APPLICATION_ID": settings.VONAGE_APPLICATION_ID,
        "VONAGE_PRIVATE_KEY_PATH": settings.VONAGE_PRIVATE_KEY_PATH,
        "VONAGE_API_BASE_URL": settings.VONAGE_API_BASE_URL,
    }
    return [name for name, value in required.items() if not value]


def _build_jwt() -> str:
    # Vonage-style application auth: a short-lived RS256 JWT carrying the
    # application id, signed with the application's private key.
    with open(settings.VONAGE_PRIVATE_KEY_PATH) as f:
        private_key = f.read()
    now = datetime.now(timezone.utc)
    claims = {
        "application_id": settings.VONAGE_APPLICATION_ID,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # The rest of the app uses naive UTC (datetime.utcnow), so match that.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def check_sim_swap(phone_number: str) -> dict:
    """
    Ask the carrier when this number's SIM was last swapped.

    Returns {"swapped": bool, "swapped_at": datetime | None}, where
    `swapped_at` is naive UTC (matching User.sim_swapped_at).

    Raises NotImplementedError when the carrier settings are not configured.
    """
    missing = _missing_settings()
    if missing:
        raise NotImplementedError(
            "check_sim_swap is architecture-ready but not connected to a real "
            "carrier: no Vonage credentials are configured "
            f"(missing: {', '.join(missing)}). Set them in the environment to "
            "enable a live lookup."
        )

    # Expected shape (UNVERIFIED): POST <base>/sim-swap/retrieve-date with
    # {"phoneNumber": "..."} -> {"latestSimChange": "<ISO-8601>" | null}.
    response = httpx.post(
        f"{settings.VONAGE_API_BASE_URL.rstrip('/')}/sim-swap/retrieve-date",
        json={"phoneNumber": phone_number},
        headers={"Authorization": f"Bearer {_build_jwt()}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    swapped_at = _parse_timestamp(response.json().get("latestSimChange"))
    return {"swapped": swapped_at is not None, "swapped_at": swapped_at}
