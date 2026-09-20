"""
Tests for the risk engine's rule logic. Run with: pytest tests/ -v
"""
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models import Base, User
from risk_engine import risk_engine


@pytest.fixture
def db():
    # Fresh in-memory DB per test; StaticPool keeps one shared connection,
    # otherwise each connection would get its own empty in-memory DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def make_user(db, sim_swapped_at=None, account_age_days=365,
              email="test@example.com", phone="9999999999"):
    user = User(
        email=email, phone=phone,
        password_hash="x", failed_attempts=0,
        sim_swapped_at=sim_swapped_at,
        created_at=datetime.utcnow() - timedelta(days=account_age_days),
    )
    db.add(user)
    db.commit()
    return user


def test_normal_login_is_low_risk(db):
    user = make_user(db)
    result = risk_engine.assess(user, device_id="known-device", ip="10.0.0.1", amount=500, db=db)
    # First time seeing this device_id -> counted as unknown, so expect at most MEDIUM
    assert result["action"] in ("ALLOW", "CHALLENGE")


def test_recent_sim_swap_blocks(db):
    user = make_user(db, sim_swapped_at=datetime.utcnow() - timedelta(minutes=30))
    result = risk_engine.assess(user, device_id="new-device", ip="203.0.113.5", amount=5000, db=db)
    assert result["action"] == "BLOCK"
    assert "SIM_SWAP_CRITICAL" in result["flags"]


def test_old_harmless_sim_swap_does_not_auto_block(db):
    user = make_user(db, sim_swapped_at=datetime.utcnow() - timedelta(days=200))
    result = risk_engine.assess(user, device_id="d1", ip="10.0.0.1", amount=100, db=db)
    assert result["action"] != "BLOCK"


def test_high_value_transaction_raises_score(db):
    user_low = make_user(db)
    low_result = risk_engine.assess(user_low, device_id="d2", ip="10.0.0.1", amount=100, db=db)

    user_high = make_user(db, email="high@example.com", phone="8888888888")
    high_result = risk_engine.assess(user_high, device_id="d3", ip="10.0.0.1", amount=150000, db=db)

    assert high_result["score"] >= low_result["score"]
