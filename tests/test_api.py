"""
API-level tests for main.py using FastAPI's TestClient and an in-memory SQLite
DB (same idea as test_risk.py). Run with: pytest tests/test_api.py -v

Setup notes:
  - DATABASE_URL is pointed at :memory: BEFORE importing main, so importing it
    never creates/touches the real simguard.db.
  - main.py mounts StaticFiles("pages") relative to the CWD, so we import it
    from the project root.
  - get_db is overridden per test with a fresh in-memory DB (StaticPool keeps
    one shared connection, otherwise each connection would get its own empty DB).
  - The slowapi rate limiter is disabled so tests don't trip "5/minute".
"""
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

_cwd = os.getcwd()
os.chdir(ROOT)
try:
    import main
finally:
    os.chdir(_cwd)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth import create_access_token
from models import Base, User, Transaction, FraudAlert, LoginAttempt, PendingOtp

DEVICE = "device-1"
CREDS = {"email": "alice@example.com", "phone": "9876543210", "password": "s3cretpass"}


@pytest.fixture
def env():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[main.get_db] = override_get_db
    main.limiter.enabled = False
    # Starlette 0.36.3's TestClient hard-codes scope["client"] = None, and
    # main.py reads request.client.host, so fill it in at the ASGI layer.
    async def app_with_client(scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope.get("client") is None:
            scope["client"] = ("testclient", 50000)
        await main.app(scope, receive, send)

    yield TestClient(app_with_client), Session
    main.app.dependency_overrides.clear()
    main.limiter.enabled = True


def _register(client, **overrides):
    return client.post("/api/register", json={**CREDS, **overrides})


def _login(client, password=CREDS["password"], email=CREDS["email"]):
    return client.post(
        "/api/login", json={"email": email, "password": password, "device_id": DEVICE}
    )


# --------------------------------------------------------------- register

# Purpose: registering a valid user succeeds and the password is stored as a
# bcrypt hash, never as plaintext.
def test_register_success_stores_hashed_password(env):
    client, Session = env
    r = _register(client)
    assert r.status_code == 200
    assert r.json()["message"] == "Registered"

    user = Session().query(User).filter_by(email=CREDS["email"]).one()
    assert user.password_hash != CREDS["password"]
    assert user.password_hash.startswith("$2")


# Purpose: the same email can't register twice (400), and only one row exists.
def test_register_duplicate_email_rejected(env):
    client, Session = env
    assert _register(client).status_code == 200
    r = _register(client, phone="1112223334")
    assert r.status_code == 400
    assert Session().query(User).count() == 1


# Purpose: Pydantic input validation — a too-short password (min 8) and a
# malformed email are rejected with 422 before touching the DB.
def test_register_validation_errors(env):
    client, Session = env
    assert _register(client, password="short").status_code == 422
    assert _register(client, email="not-an-email").status_code == 422
    assert Session().query(User).count() == 0


# ------------------------------------------------------------------ login

# Purpose: correct credentials pass the password check and reach the risk
# engine; a fresh user on a new device is not blocked, so we get the
# "OTP required" response with a risk assessment.
def test_login_success(env):
    client, _ = env
    _register(client)
    r = _login(client)
    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "Risk-assessed, OTP required"
    assert body["risk"]["action"] in ("ALLOW", "CHALLENGE")


# Purpose: wrong password and unknown email both return the same generic 401
# (no user enumeration), and a wrong password bumps failed_attempts.
def test_login_failure_wrong_password_and_unknown_email(env):
    client, Session = env
    _register(client)

    r = _login(client, password="wrong-password")
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid credentials"
    assert Session().query(User).one().failed_attempts == 1

    r = _login(client, email="nobody@example.com")
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid credentials"


# Purpose: end-to-end lockout — after 5 wrong passwords the 6th attempt gets
# 423 even with the CORRECT password, proving the lock is enforced in the route.
def test_login_locks_after_five_failures(env):
    client, _ = env
    _register(client)
    for _ in range(5):
        assert _login(client, password="wrong-password").status_code == 401
    assert _login(client).status_code == 423


# Purpose: a recent SIM swap makes login itself return 403 "Login blocked"
# (the risk engine's hard override), even with correct credentials.
def test_login_blocked_when_sim_recently_swapped(env):
    client, Session = env
    uid = _register(client).json()["user_id"]
    assert client.post(f"/api/admin/simulate-swap?user_id={uid}").status_code == 200

    r = _login(client)
    assert r.status_code == 403
    assert "SIM_SWAP_CRITICAL" in r.json()["detail"]["risk"]["flags"]


# --------------------------------------------------------------- transfer

def _auth_header(user_id):
    return {"Authorization": f"Bearer {create_access_token({'sub': str(user_id)})}"}


# Purpose: /api/transfer requires a valid JWT — no header, or a garbage token,
# gets 401 and never reaches the risk engine.
def test_transfer_requires_valid_token(env):
    client, _ = env
    body = {"amount": 100, "device_id": DEVICE}
    assert client.post("/api/transfer", json=body).status_code == 401
    r = client.post("/api/transfer", json=body, headers={"Authorization": "Bearer junk"})
    assert r.status_code == 401


# Purpose: the risk-blocking path. A user whose SIM was swapped moments ago
# attempts a transfer: expect 403, a "blocked" Transaction row, a FraudAlert
# row, and — most importantly — the balance is NOT debited.
def test_transfer_blocked_by_risk_engine(env):
    client, Session = env
    uid = _register(client).json()["user_id"]

    db = Session()
    user = db.query(User).filter_by(id=uid).one()
    user.sim_swapped_at = datetime.utcnow()
    starting_balance = user.balance
    db.commit()
    db.close()

    r = client.post(
        "/api/transfer", json={"amount": 5000, "device_id": "attacker-device"},
        headers=_auth_header(uid),
    )
    assert r.status_code == 403
    detail = r.json()["detail"]
    assert detail["message"] == "Transfer blocked"
    assert detail["risk"]["action"] == "BLOCK"
    assert "SIM_SWAP_CRITICAL" in detail["risk"]["flags"]

    db = Session()
    assert db.query(User).filter_by(id=uid).one().balance == starting_balance
    txn = db.query(Transaction).one()
    assert txn.status == "blocked" and txn.amount == 5000
    alert = db.query(FraudAlert).one()
    assert "SIM_SWAP_CRITICAL" in alert.reason


# ------------------------------------------------------------------- user

# Purpose: GET /api/user/{id} returns email and balance for the token's own
# id, and nothing else — in particular no password_hash.
def test_get_own_user_returns_email_and_balance_only(env):
    client, _ = env
    uid = _register(client).json()["user_id"]
    r = client.get(f"/api/user/{uid}", headers=_auth_header(uid))
    assert r.status_code == 200
    assert r.json() == {"email": CREDS["email"], "balance": 50000.0}


# Purpose: a valid token for user A must not read user B's data — requesting
# another id gets 403 (and an unauthenticated request gets 401).
def test_get_other_users_data_is_forbidden(env):
    client, _ = env
    uid_a = _register(client).json()["user_id"]
    uid_b = _register(
        client, email="bob@example.com", phone="1112223334"
    ).json()["user_id"]

    r = client.get(f"/api/user/{uid_b}", headers=_auth_header(uid_a))
    assert r.status_code == 403
    assert "email" not in r.json() and "balance" not in r.json()
    assert client.get(f"/api/user/{uid_a}").status_code == 401


# Purpose: GET /api/user/{id}/logins returns only the token owner's attempts —
# with two users who each logged in, user A sees exactly their own row and
# none of B's fields leak (no user_id in the payload at all).
def test_get_own_logins_returns_only_own_attempts(env):
    client, _ = env
    uid_a = _register(client).json()["user_id"]
    _register(client, email="bob@example.com", phone="1112223334")
    assert _login(client).status_code == 200
    assert _login(client, email="bob@example.com").status_code == 200
    assert _login(client, email="bob@example.com").status_code == 200

    r = client.get(f"/api/user/{uid_a}/logins", headers=_auth_header(uid_a))
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert set(rows[0]) == {"risk_score", "level", "action", "flags", "created_at"}


# Purpose: a valid token for user A must not read user B's login history —
# requesting another id gets 403 with no rows (and no token gets 401).
def test_get_other_users_logins_is_forbidden(env):
    client, _ = env
    uid_a = _register(client).json()["user_id"]
    uid_b = _register(
        client, email="bob@example.com", phone="1112223334"
    ).json()["user_id"]
    assert _login(client, email="bob@example.com").status_code == 200

    r = client.get(f"/api/user/{uid_b}/logins", headers=_auth_header(uid_a))
    assert r.status_code == 403
    assert isinstance(r.json(), dict) and "detail" in r.json()
    assert client.get(f"/api/user/{uid_a}/logins").status_code == 401


# ------------------------------------------------------------------- admin

# Purpose: GET /api/admin/users lists every user with exactly the safe fields,
# exposes sim_swapped as a bool (not the raw timestamp), and never leaks
# password_hash.
def test_admin_users_lists_safe_fields_only(env):
    client, Session = env
    uid_a = _register(client).json()["user_id"]
    _register(client, email="bob@example.com", phone="1112223334")
    client.post(f"/api/admin/simulate-swap?user_id={uid_a}")

    r = client.get("/api/admin/users")
    assert r.status_code == 200
    rows = r.json()
    assert [u["email"] for u in rows] == [CREDS["email"], "bob@example.com"]
    for u in rows:
        assert set(u) == {"id", "email", "phone", "balance", "created_at", "sim_swapped"}
        assert isinstance(u["sim_swapped"], bool)
    assert [u["sim_swapped"] for u in rows] == [True, False]
    assert "password_hash" not in r.text and "$2" not in r.text


# Purpose: GET /api/admin/stats on an empty DB returns zeros, and always
# includes all three action keys so the UI never has to guard for missing ones.
def test_admin_stats_empty(env):
    client, _ = env
    r = client.get("/api/admin/stats")
    assert r.status_code == 200
    assert r.json() == {
        "logins_last_24h": {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0},
        "total_users": 0,
        "fraud_alerts_total": 0,
    }


# Purpose: stats counts login attempts by action for the last 24h only (a
# 25-hour-old attempt is excluded), plus total users and total fraud alerts.
def test_admin_stats_counts_recent_attempts_users_and_alerts(env):
    client, Session = env
    uid = _register(client).json()["user_id"]
    _register(client, email="bob@example.com", phone="1112223334")

    now = datetime.utcnow()
    db = Session()
    for action, age_hours in [
        ("ALLOW", 1), ("ALLOW", 2), ("CHALLENGE", 3), ("BLOCK", 4),
        ("BLOCK", 25),  # outside the 24h window
    ]:
        db.add(LoginAttempt(
            user_id=uid, action=action, created_at=now - timedelta(hours=age_hours)
        ))
    db.add(FraudAlert(user_id=uid, reason="SIM_SWAP_CRITICAL", risk_score=80))
    db.add(FraudAlert(user_id=uid, reason="UNKNOWN_DEVICE", risk_score=75))
    db.commit()
    db.close()

    assert client.get("/api/admin/stats").json() == {
        "logins_last_24h": {"ALLOW": 2, "CHALLENGE": 1, "BLOCK": 1},
        "total_users": 2,
        "fraud_alerts_total": 2,
    }


# -------------------------------------------------------------------- OTP

def _login_for_otp_token(client):
    r = _login(client)
    assert r.status_code == 200
    return r.json()["otp_token"]


def _verify(client, token, otp="123456"):
    return client.post(f"/api/verify-otp?token={token}&otp={otp}")


# Purpose: login (ALLOW/CHALLENGE) returns an opaque otp_token and does NOT
# expose user_id; presenting that token with a 6-digit OTP succeeds and yields
# a working JWT for the right user, plus the user_id the dashboard needs.
def test_verify_otp_with_valid_token_succeeds(env):
    client, _ = env
    uid = _register(client).json()["user_id"]
    body = _login(client).json()
    assert "user_id" not in body
    assert len(body["otp_token"]) >= 32

    r = _verify(client, body["otp_token"])
    assert r.status_code == 200
    assert r.json()["token_type"] == "bearer"
    assert r.json()["user_id"] == uid
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get(f"/api/user/{uid}", headers=headers).status_code == 200


# Purpose: tokens are single-use — the first verify succeeds, and reusing the
# same token afterwards fails with 401.
def test_verify_otp_token_is_single_use(env):
    client, _ = env
    _register(client)
    token = _login_for_otp_token(client)
    assert _verify(client, token).status_code == 200
    assert _verify(client, token).status_code == 401


# Purpose: an unknown token is rejected (401) — this is the bypass fix: you
# can't get a JWT without having passed login first.
def test_verify_otp_unknown_token_rejected(env):
    client, _ = env
    _register(client)
    assert _verify(client, "not-a-real-token").status_code == 401


# Purpose: an expired token is rejected (401) even with a well-formed OTP,
# and the stale record is cleaned up. Expiry is simulated by backdating
# expires_at in the DB.
def test_verify_otp_expired_token_rejected(env):
    client, Session = env
    _register(client)
    token = _login_for_otp_token(client)

    db = Session()
    db.query(PendingOtp).filter_by(token=token).one().expires_at = (
        datetime.utcnow() - timedelta(seconds=1)
    )
    db.commit()
    db.close()

    assert _verify(client, token).status_code == 401
    assert Session().query(PendingOtp).count() == 0


# Purpose: /api/verify-otp is rate limited to 10/minute per client. The env
# fixture disables the limiter for every other test, so this one turns it back
# on (with clean counters): requests 1-10 are handled (401 for a bogus token),
# and the 11th is rejected with 429 before reaching the route logic.
def test_verify_otp_is_rate_limited(env):
    client, _ = env
    main.limiter.enabled = True
    main.limiter.reset()
    codes = [_verify(client, "bogus-token").status_code for _ in range(11)]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429


# Purpose: the old pattern — verify-otp with a bare user_id — no longer works.
# The route now requires `token`, so the request is rejected (422) and no JWT
# is issued for any user_id.
def test_verify_otp_old_user_id_pattern_no_longer_works(env):
    client, _ = env
    uid = _register(client).json()["user_id"]
    r = client.post(f"/api/verify-otp?user_id={uid}&otp=123456")
    assert r.status_code == 422
    assert "access_token" not in r.text


# Purpose: a malformed OTP is a 400 that does NOT burn the token, so the user
# can retry the OTP; the token is consumed only on success.
def test_verify_otp_bad_format_keeps_token_usable(env):
    client, _ = env
    _register(client)
    token = _login_for_otp_token(client)
    assert _verify(client, token, otp="abc").status_code == 400
    assert _verify(client, token, otp="12345").status_code == 400
    assert _verify(client, token).status_code == 200


# Purpose: a BLOCKED login (recent SIM swap) never issues an OTP token, so a
# blocked attacker can't reach the OTP step at all.
def test_blocked_login_creates_no_otp_token(env):
    client, Session = env
    uid = _register(client).json()["user_id"]
    client.post(f"/api/admin/simulate-swap?user_id={uid}")
    r = _login(client)
    assert r.status_code == 403
    assert "otp_token" not in r.text
    assert Session().query(PendingOtp).count() == 0


# Purpose: input validation on the transfer body — non-positive amounts are
# rejected with 422 (Field(gt=0)) and leave no transaction behind.
def test_transfer_rejects_non_positive_amount(env):
    client, Session = env
    uid = _register(client).json()["user_id"]
    r = client.post(
        "/api/transfer", json={"amount": 0, "device_id": DEVICE},
        headers=_auth_header(uid),
    )
    assert r.status_code == 422
    assert Session().query(Transaction).count() == 0
