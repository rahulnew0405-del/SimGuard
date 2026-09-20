"""
SQLAlchemy models — the actual database schema.

Design notes (say these in the interview if asked "is your schema normalized?"):
- User holds only user-owned facts (email, password hash, phone, sim-swap state).
- KnownDevice, LoginAttempt, Transaction, FraudAlert are separate tables linked
  by user_id (a foreign key) rather than columns bolted onto User — that's 3NF:
  each table describes one entity, and repeating data (many devices per user,
  many logins per user) lives in its own table instead of duplicating rows.
- Indexing: user_id is indexed on every child table because every query here
  is "give me this user's devices / logins / transactions" — the most
  frequent access pattern decides what gets an index.
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    balance = Column(Float, default=50000.0)

    # SIM-swap state: when the SIM was last (simulated) swapped. NULL = never.
    sim_swapped_at = Column(DateTime, nullable=True)

    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    devices = relationship("KnownDevice", back_populates="user")
    logins = relationship("LoginAttempt", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")


class KnownDevice(Base):
    __tablename__ = "known_devices"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    device_id = Column(String, nullable=False)  # a browser fingerprint / random client id
    first_seen = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="devices")


class LoginAttempt(Base):
    """Every login attempt, successful or not — this is the audit trail."""
    __tablename__ = "login_attempts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    device_id = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    risk_score = Column(Integer, nullable=True)
    risk_level = Column(String, nullable=True)   # LOW / MEDIUM / HIGH
    action = Column(String, nullable=True)        # ALLOW / CHALLENGE / BLOCK
    flags = Column(Text, nullable=True)            # comma-separated reason codes
    success = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logins")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=True)
    status = Column(String, default="completed")  # completed / blocked
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="transactions")


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    reason = Column(String, nullable=False)
    risk_score = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
