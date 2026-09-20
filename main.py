"""
FastAPI app — every HTTP route lives here. Kept deliberately flat (one file)
since the project is small; a bigger app would split this into routers, but
you should be able to say *why* it's flat here rather than pretend it needs
to be.
"""
import random
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from models import Base, User, LoginAttempt, Transaction, FraudAlert, PendingOtp
from auth import (
    hash_password, verify_password, create_access_token, decode_token,
    is_account_locked, register_failed_attempt, reset_failed_attempts,
)
from risk_engine import risk_engine

# ---- DB setup ----
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- Rate limiting: caps requests per IP on sensitive endpoints ----
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SimGuard — Risk-Based Authentication")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="pages"), name="static")


# ---- Request/response schemas (Pydantic does input validation here) ----
class RegisterIn(BaseModel):
    email: EmailStr
    phone: str = Field(min_length=10, max_length=15)
    password: str = Field(min_length=8)


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    device_id: str


class TransferIn(BaseModel):
    amount: float = Field(gt=0)
    device_id: str


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    payload = decode_token(auth.split(" ", 1)[1])
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    user = db.query(User).filter_by(id=int(payload["sub"])).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user


# ---- Routes ----

@app.get("/")
def home():
    return FileResponse("pages/login.html")


@app.post("/api/register")
@limiter.limit("5/minute")
def register(request: Request, body: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter_by(email=body.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        email=body.email, phone=body.phone,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    return {"message": "Registered", "user_id": user.id}


@app.post("/api/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=body.email).first()
    if not user:
        raise HTTPException(401, "Invalid credentials")

    if is_account_locked(user):
        raise HTTPException(423, "Account locked. Try again later.")

    if not verify_password(body.password, user.password_hash):
        register_failed_attempt(user, db)
        raise HTTPException(401, "Invalid credentials")

    reset_failed_attempts(user, db)

    # Run the risk assessment on THIS login attempt.
    result = risk_engine.assess(
        user, device_id=body.device_id,
        ip=request.client.host, amount=0, db=db,
    )

    attempt = LoginAttempt(
        user_id=user.id, device_id=body.device_id, ip_address=request.client.host,
        risk_score=result["score"], risk_level=result["level"], action=result["action"],
        flags=",".join(result["flags"]), success=(result["action"] != "BLOCK"),
    )
    db.add(attempt)

    if result["action"] == "BLOCK":
        db.add(FraudAlert(
            user_id=user.id, reason=",".join(result["flags"]),
            risk_score=result["score"],
        ))
        db.commit()
        raise HTTPException(403, {"message": "Login blocked", "risk": result})

    # Only reached for ALLOW / CHALLENGE. The opaque token is the only way to
    # get to /api/verify-otp; the client never gets to name a user_id there.
    otp_token = secrets.token_urlsafe(32)
    db.add(PendingOtp(
        user_id=user.id, token=otp_token,
        expires_at=datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES),
    ))
    db.commit()
    return {"message": "Risk-assessed, OTP required", "risk": result, "otp_token": otp_token}


@app.post("/api/verify-otp")
@limiter.limit("10/minute")
def verify_otp(request: Request, token: str, otp: str, db: Session = Depends(get_db)):
    # Demo OTP: any 6-digit code is accepted (a real system would text one
    # via an SMS gateway — this app's whole point is that SMS OTP alone is
    # exactly what a SIM-swap attacker can intercept, which is *why* the
    # risk engine exists as a layer in front of it).
    pending = db.query(PendingOtp).filter_by(token=token).first()
    if not pending:
        raise HTTPException(401, "Invalid or expired OTP token")
    if datetime.utcnow() >= pending.expires_at:
        db.delete(pending)
        db.commit()
        raise HTTPException(401, "Invalid or expired OTP token")

    # A malformed OTP doesn't burn the token; only success or expiry does.
    if len(otp) != 6 or not otp.isdigit():
        raise HTTPException(400, "Invalid OTP format")

    user = db.query(User).filter_by(id=pending.user_id).first()
    if not user:
        raise HTTPException(401, "Invalid or expired OTP token")

    db.delete(pending)  # single use
    db.commit()
    access_token = create_access_token({"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.id}


@app.post("/api/transfer")
@limiter.limit("10/minute")
def transfer(
    request: Request, body: TransferIn,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    result = risk_engine.assess(
        user, device_id=body.device_id, ip=request.client.host,
        amount=body.amount, db=db,
    )
    status = "blocked" if result["action"] == "BLOCK" else "completed"
    txn = Transaction(user_id=user.id, amount=body.amount, risk_score=result["score"], status=status)
    db.add(txn)

    if status == "blocked":
        db.add(FraudAlert(user_id=user.id, reason=",".join(result["flags"]), risk_score=result["score"]))
        db.commit()
        raise HTTPException(403, {"message": "Transfer blocked", "risk": result})

    user.balance -= body.amount
    db.commit()
    return {"message": "Transfer completed", "risk": result, "new_balance": user.balance}


@app.get("/api/user/{user_id}")
def get_user(user_id: int, user: User = Depends(get_current_user)):
    # A token only grants access to its own account.
    if user.id != user_id:
        raise HTTPException(403, "Forbidden")
    return {"email": user.email, "balance": user.balance}


@app.get("/api/user/{user_id}/logins")
def get_user_logins(
    user_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    if user.id != user_id:
        raise HTTPException(403, "Forbidden")
    rows = (
        db.query(LoginAttempt).filter_by(user_id=user_id)
        .order_by(LoginAttempt.created_at.desc()).limit(20).all()
    )
    return [
        {
            "risk_score": r.risk_score, "level": r.risk_level, "action": r.action,
            "flags": r.flags, "created_at": r.created_at.isoformat(),
        } for r in rows
    ]


# ---- Admin / demo endpoints ----

@app.post("/api/admin/simulate-swap")
def simulate_swap(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.sim_swapped_at = datetime.utcnow()
    db.commit()
    return {"message": f"SIM swap simulated for user {user_id}"}


@app.post("/api/admin/reset-swap")
def reset_swap(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.sim_swapped_at = None
    db.commit()
    return {"message": f"Reset for user {user_id}"}


@app.get("/api/admin/fraud-logs")
def fraud_logs(db: Session = Depends(get_db)):
    rows = db.query(LoginAttempt).order_by(LoginAttempt.created_at.desc()).limit(50).all()
    return [
        {
            "user_id": r.user_id, "risk_score": r.risk_score, "level": r.risk_level,
            "action": r.action, "flags": r.flags, "created_at": r.created_at.isoformat(),
        } for r in rows
    ]


@app.get("/api/admin/users")
def admin_users(db: Session = Depends(get_db)):
    rows = db.query(User).order_by(User.id).all()
    return [
        {
            "id": u.id, "email": u.email, "phone": u.phone, "balance": u.balance,
            "created_at": u.created_at.isoformat(),
            "sim_swapped": u.sim_swapped_at is not None,
        } for u in rows
    ]


@app.get("/api/admin/stats")
def admin_stats(db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(hours=24)
    counts = dict(
        db.query(LoginAttempt.action, func.count(LoginAttempt.id))
        .filter(LoginAttempt.created_at >= since)
        .group_by(LoginAttempt.action).all()
    )
    return {
        "logins_last_24h": {a: counts.get(a, 0) for a in ("ALLOW", "CHALLENGE", "BLOCK")},
        "total_users": db.query(User).count(),
        "fraud_alerts_total": db.query(FraudAlert).count(),
    }


@app.get("/api/admin/model-meta")
def model_meta():
    return risk_engine.meta


@app.get("/api/health")
def health():
    return {"status": "ok"}
