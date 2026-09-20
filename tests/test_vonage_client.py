"""
Tests for vonage_client.py. The client is an unverified, best-effort sketch
(no real credentials), so these tests check OUR behaviour — the unconfigured
guard and the request we build — not the real Vonage API.
Run with: pytest tests/test_vonage_client.py -v
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

import vonage_client
from config import settings


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


# Purpose: with no carrier credentials configured, check_sim_swap must refuse
# to run and raise NotImplementedError explaining this is architecture-ready
# but not connected to a real carrier, naming what's missing. It must not
# attempt any network call.
def test_unconfigured_raises_not_implemented(monkeypatch):
    for name in ("VONAGE_APPLICATION_ID", "VONAGE_PRIVATE_KEY_PATH", "VONAGE_API_BASE_URL"):
        monkeypatch.setattr(settings, name, "")

    def no_network(*a, **kw):
        raise AssertionError("must not call the network when unconfigured")

    monkeypatch.setattr(vonage_client.httpx, "post", no_network)

    with pytest.raises(NotImplementedError) as exc:
        vonage_client.check_sim_swap("9876543210")
    msg = str(exc.value)
    assert "not connected to a real carrier" in msg
    assert "VONAGE_APPLICATION_ID" in msg


# Purpose: with (fake) credentials configured, the client builds the request
# we expect — POST to <base>/sim-swap/retrieve-date with the phone number in
# the body and a Bearer RS256 JWT carrying the application id — and maps the
# response to swapped/swapped_at (naive UTC). Swap date present -> swapped.
def test_configured_builds_request_and_parses_swap(monkeypatch, tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file = tmp_path / "private.key"
    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    monkeypatch.setattr(settings, "VONAGE_APPLICATION_ID", "app-123")
    monkeypatch.setattr(settings, "VONAGE_PRIVATE_KEY_PATH", str(key_file))
    monkeypatch.setattr(settings, "VONAGE_API_BASE_URL", "https://carrier.example/")

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return FakeResponse({"latestSimChange": "2026-09-20T10:30:00Z"})

    monkeypatch.setattr(vonage_client.httpx, "post", fake_post)

    result = vonage_client.check_sim_swap("9876543210")

    assert captured["url"] == "https://carrier.example/sim-swap/retrieve-date"
    assert captured["json"] == {"phoneNumber": "9876543210"}
    assert captured["timeout"] == vonage_client.REQUEST_TIMEOUT_SECONDS
    scheme, token = captured["headers"]["Authorization"].split(" ", 1)
    assert scheme == "Bearer"
    claims = jwt.decode(token, public_pem, algorithms=["RS256"])
    assert claims["application_id"] == "app-123"

    assert result == {"swapped": True, "swapped_at": datetime(2026, 9, 20, 10, 30)}

    # No swap on record -> not swapped.
    monkeypatch.setattr(
        vonage_client.httpx, "post",
        lambda *a, **kw: FakeResponse({"latestSimChange": None}),
    )
    assert vonage_client.check_sim_swap("9876543210") == {"swapped": False, "swapped_at": None}
