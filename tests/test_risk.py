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
from models import Base, User, LoginAttempt
from risk_engine import risk_engine
from geo_sim import haversine_km, lookup_city


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


# ---- Geo-velocity (IMPOSSIBLE_TRAVEL). Locations are SIMULATED via geo_sim.py:
# 203.0.113.x = Mumbai, 198.51.100.x = Delhi (~1,150 km apart), 192.0.2.x = London.

MUMBAI_IP, DELHI_IP = "203.0.113.10", "198.51.100.7"


def add_prior_login(db, user, ip, seconds_ago, success=True):
    db.add(LoginAttempt(
        user_id=user.id, device_id="d0", ip_address=ip, success=success,
        created_at=datetime.utcnow() - timedelta(seconds=seconds_ago),
    ))
    db.commit()


# Purpose: sanity-check the Haversine helper and the simulated lookup — the
# Mumbai-Delhi great-circle distance is roughly 1,150 km, and IPs outside the
# simulated table (private, loopback, empty) resolve to no location.
def test_haversine_and_simulated_lookup():
    _, lat1, lon1 = lookup_city(MUMBAI_IP)
    _, lat2, lon2 = lookup_city(DELHI_IP)
    assert 1100 < haversine_km(lat1, lon1, lat2, lon2) < 1200
    assert haversine_km(lat1, lon1, lat1, lon1) == 0
    assert lookup_city("10.0.0.1") is None
    assert lookup_city("testclient") is None
    assert lookup_city(None) is None


# Purpose: normal case — a second login from the same (simulated) city moments
# later is not suspicious, so no IMPOSSIBLE_TRAVEL flag.
def test_same_location_no_impossible_travel(db):
    user = make_user(db)
    add_prior_login(db, user, MUMBAI_IP, seconds_ago=30)
    result = risk_engine.assess(user, device_id="d1", ip="203.0.113.99", amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" not in result["flags"]


# Purpose: Mumbai then Delhi 30 seconds later implies thousands of km/h, which
# is physically impossible, so IMPOSSIBLE_TRAVEL must fire and add its 40 rule
# points (the final score is higher than for the same login with no prior one).
def test_impossible_travel_flags(db):
    user = make_user(db)
    baseline = risk_engine.assess(user, device_id="d1", ip=DELHI_IP, amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" not in baseline["flags"]  # no prior login yet

    add_prior_login(db, user, MUMBAI_IP, seconds_ago=30)
    result = risk_engine.assess(user, device_id="d1", ip=DELHI_IP, amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" in result["flags"]
    assert result["score"] > baseline["score"]


# Purpose: same Mumbai -> Delhi jump but 6 hours apart is ~190 km/h, well within
# what a flight or train allows, so no flag.
def test_plausible_travel_no_flag(db):
    user = make_user(db)
    add_prior_login(db, user, MUMBAI_IP, seconds_ago=6 * 3600)
    result = risk_engine.assess(user, device_id="d1", ip=DELHI_IP, amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" not in result["flags"]


# Purpose: no signal without knowledge — an unknown/private IP on either side,
# or no prior login at all, must never produce the flag.
def test_unknown_locations_never_flag(db):
    user = make_user(db)
    add_prior_login(db, user, "10.0.0.1", seconds_ago=5)  # private prior IP
    r1 = risk_engine.assess(user, device_id="d1", ip=DELHI_IP, amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" not in r1["flags"]

    other = make_user(db, email="other@example.com", phone="7777777777")
    add_prior_login(db, other, MUMBAI_IP, seconds_ago=5)
    r2 = risk_engine.assess(other, device_id="d1", ip="192.168.1.5", amount=100, db=db)  # private current IP
    assert "IMPOSSIBLE_TRAVEL" not in r2["flags"]


# Purpose: only ACCEPTED logins count as "last location". A blocked attempt
# from another city (e.g. an attacker) must not make the real user's next
# login look like impossible travel.
def test_blocked_prior_attempt_is_ignored(db):
    user = make_user(db)
    add_prior_login(db, user, MUMBAI_IP, seconds_ago=30, success=False)
    result = risk_engine.assess(user, device_id="d1", ip=DELHI_IP, amount=100, db=db)
    assert "IMPOSSIBLE_TRAVEL" not in result["flags"]
