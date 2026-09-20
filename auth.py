"""
Password hashing + JWT issuing/verification + account lockout.

Why bcrypt, if asked:
  bcrypt is a "slow" hash function by design (it has a configurable work
  factor / cost). A fast hash like plain SHA-256 lets an attacker who steals
  your password-hash table try billions of guesses per second on a GPU.
  bcrypt also auto-generates a random salt per password, so two users with
  the same password get completely different hashes — this defeats rainbow
  table attacks.

Why JWT here:
  Once risk-assessed and OTP-verified, the server issues a signed token
  instead of keeping server-side session state. The signature (HMAC-SHA256
  with JWT_SECRET) means the token can't be forged or edited by the client —
  any tampering invalidates the signature and decode_token() returns None.
"""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None


# ---- Account lockout: a rule-based guard independent of the risk engine ----
# Five wrong passwords locks the account for 15 minutes, regardless of risk
# score. This stops brute-force password guessing even if every other signal
# (device, IP, SIM state) looks completely normal.

def is_account_locked(user) -> bool:
    return bool(user.locked_until and datetime.utcnow() < user.locked_until)


def register_failed_attempt(user, db):
    user.failed_attempts = (user.failed_attempts or 0) + 1
    if user.failed_attempts >= settings.MAX_FAILED_ATTEMPTS:
        user.locked_until = datetime.utcnow() + timedelta(minutes=settings.LOCKOUT_MINUTES)
    db.commit()


def reset_failed_attempts(user, db):
    user.failed_attempts = 0
    user.locked_until = None
    db.commit()
