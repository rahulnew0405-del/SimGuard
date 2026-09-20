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
from datetime import datetime

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
from models import Base, User, Transaction, FraudAlert

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
