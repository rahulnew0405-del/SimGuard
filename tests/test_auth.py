"""
Unit tests for auth.py: password hashing, JWTs, and account lockout.
Run with: pytest tests/test_auth.py -v
"""
import base64
import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jose import jwt

from auth import (
    hash_password, verify_password, create_access_token, decode_token,
    is_account_locked, register_failed_attempt, reset_failed_attempts,
)
from config import settings
from models import User


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ---------------------------------------------------------------- hashing

# Purpose: the stored value must never be the plaintext, and it must be a real
# bcrypt hash ($2b$...), i.e. a slow salted hash rather than something reversible.
def test_hash_is_not_plaintext_and_is_bcrypt():
    h = hash_password("correct-horse")
    assert h != "correct-horse"
    assert h.startswith("$2")


# Purpose: the happy path — right password verifies, wrong password does not.
def test_verify_password_correct_and_incorrect():
    h = hash_password("correct-horse")
    assert verify_password("correct-horse", h) is True
    assert verify_password("wrong-horse", h) is False


# Purpose: bcrypt salts per hash, so the same password hashed twice must give
# different strings (defeats rainbow tables) while both still verify.
def test_same_password_hashes_differently_but_both_verify():
    h1, h2 = hash_password("same-password"), hash_password("same-password")
    assert h1 != h2
    assert verify_password("same-password", h1)
    assert verify_password("same-password", h2)


# ------------------------------------------------------------------- JWT

# Purpose: a freshly created token round-trips — decoding returns our claims
# plus exp/iat, and exp is in the future.
def test_create_and_decode_token_roundtrip():
    token = create_access_token({"sub": "42"})
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "42"
    assert "exp" in payload and "iat" in payload
    assert payload["exp"] > datetime.utcnow().timestamp()


# Purpose: the core security property — if a client edits the payload (here,
# changing sub to another user's id) but keeps the old signature, the
# HMAC no longer matches and decode_token must return None.
def test_tampered_payload_is_rejected():
    token = create_access_token({"sub": "1"})
    header, payload, signature = token.split(".")
    claims = json.loads(_unb64(payload))
    claims["sub"] = "999"
    forged_payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
    forged = f"{header}.{forged_payload}.{signature}"

    assert decode_token(forged) is None


# Purpose: a token signed with a different secret (an attacker guessing keys)
# must not be accepted, and garbage input must fail closed rather than raise.
def test_wrong_secret_and_garbage_tokens_are_rejected():
    forged = jwt.encode(
        {"sub": "1", "exp": datetime.utcnow() + timedelta(minutes=5)},
        "not-the-real-secret", algorithm=settings.JWT_ALGORITHM,
    )
    assert decode_token(forged) is None
    assert decode_token("not.a.jwt") is None
    assert decode_token("") is None


# Purpose: expired tokens are refused. (create_access_token treats 0 as "use
# the default", so we pass a negative lifetime to get a token already expired.)
def test_expired_token_is_rejected():
    token = create_access_token({"sub": "1"}, expires_minutes=-5)
    assert decode_token(token) is None


# --------------------------------------------------------------- lockout

def _user(**kw):
    return User(email="a@example.com", phone="9999999999", password_hash="x", **kw)


# Purpose: the exact boundary — 4 failures must NOT lock the account, the 5th
# must (MAX_FAILED_ATTEMPTS == 5), and the lock must last ~LOCKOUT_MINUTES.
def test_account_locks_on_fifth_failed_attempt():
    assert settings.MAX_FAILED_ATTEMPTS == 5
    user, db = _user(failed_attempts=0), MagicMock()

    for _ in range(4):
        register_failed_attempt(user, db)
    assert user.failed_attempts == 4
    assert not is_account_locked(user)

    register_failed_attempt(user, db)
    assert user.failed_attempts == 5
    assert is_account_locked(user)
    remaining = user.locked_until - datetime.utcnow()
    assert timedelta(minutes=settings.LOCKOUT_MINUTES - 1) < remaining <= timedelta(
        minutes=settings.LOCKOUT_MINUTES
    )
    assert db.commit.call_count == 5  # every failure is persisted


# Purpose: a lock in the past is expired, so the user can try again; a user
# who was never locked is not reported as locked.
def test_expired_lock_is_not_locked():
    assert not is_account_locked(_user())
    assert not is_account_locked(
        _user(locked_until=datetime.utcnow() - timedelta(seconds=1))
    )


# Purpose: a successful login clears the counter and any lock, so failures
# don't accumulate across separate sessions.
def test_reset_failed_attempts_clears_counter_and_lock():
    user = _user(
        failed_attempts=5, locked_until=datetime.utcnow() + timedelta(minutes=10)
    )
    db = MagicMock()
    reset_failed_attempts(user, db)
    assert user.failed_attempts == 0
    assert user.locked_until is None
    assert not is_account_locked(user)
    db.commit.assert_called_once()
