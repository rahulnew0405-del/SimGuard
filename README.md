# SimGuard

A risk-based authentication demo aimed at SIM-swap fraud. SMS one-time codes
are exactly what a SIM-swap attacker intercepts, so SimGuard puts a risk
engine *in front of* the OTP step: every login and transfer gets a 0-100 risk
score, and high-risk attempts are blocked outright.

Everything below describes what the code in this repo does today.

## What it does

A FastAPI backend with a small static HTML frontend (`pages/`). Users can
register, log in, verify an OTP, view their balance and login history, and make
transfers. An admin page can simulate a SIM swap for a user so you can watch
the risk engine react.

### Login and transfer flow

1. **Register** (`POST /api/register`) - stores the user with a bcrypt password hash.
2. **Login** (`POST /api/login`)
   - Unknown email or wrong password: `401`. Five wrong passwords lock the
     account for 15 minutes (`423`), regardless of risk score.
   - Correct password: the risk engine scores the attempt (SIM-swap recency,
     device, IP, failed attempts, plus an ML score).
   - **BLOCK** (score >= 70): `403`, a `FraudAlert` is recorded.
   - **ALLOW** (score < 35) or **CHALLENGE** (35-69): `200`, "OTP required",
     with the risk result and a single-use, opaque `otp_token` (valid for 5
     minutes). The response does not include the user's id.
3. **Verify OTP** (`POST /api/verify-otp`, query: `token`, `otp`) - looks up
   the `otp_token` from step 2; a missing or expired token is rejected with
   `401`. On success the token is deleted (single use) and a JWT
   (60-minute expiry) is issued along with the `user_id`. The OTP value itself
   is a demo: any 6-digit code is accepted. A malformed OTP returns `400` and
   leaves the token usable.
4. **Transfer** (`POST /api/transfer`, JWT required) - the engine scores the
   transfer including the amount. **BLOCK** returns `403` and records a
   `blocked` transaction plus a `FraudAlert`. Otherwise the transfer completes
   and the balance is debited.

Note: the server currently treats **ALLOW and CHALLENGE identically** for both
login and transfer. The action is returned and logged, but CHALLENGE does not
add an extra verification step. Only BLOCK changes behavior.

## Tech stack

From `requirements.txt` (versions exactly as listed there):

| Area | Packages |
|---|---|
| Web | fastapi==0.110.0, uvicorn==0.27.1, python-multipart==0.0.9 |
| Database | sqlalchemy>=2.0.36 (SQLite by default) |
| Auth | python-jose[cryptography]==3.3.0, passlib[bcrypt]==1.7.4, bcrypt<4.1 |
| Validation | email-validator==2.3.0 (Pydantic `EmailStr`) |
| Rate limiting | slowapi==0.1.9 |
| ML | scikit-learn>=1.5.0, numpy>=1.26.0, pandas>=2.0.0, joblib==1.3.2 |
| Config | python-dotenv==1.0.1 |
| Testing / HTTP | pytest==8.1.1, httpx==0.27.0 |

The frontend is plain HTML/CSS/JS with no framework.

## Risk engine: rules + ML blend (60/40)

Defined in `risk_engine.py`. The final score blends hand-written rules with a
model:

```
final_score = min(0.60 * rule_score + 0.40 * (ml_probability * 100), 100)
```

- **Rules are transparent and instant to reason about.** "SIM swapped in the
  last hour" should always be dangerous, and you don't want a learned weight to
  talk that down.
- **The model captures interactions** that are hard to hand-write as rules
  (for example, old account + known device + odd hour is fine, but new account
  + unknown device + odd hour is not).
- **A hard rule can't be silently overridden by the model.** Blending at
  60/40 keeps the rules dominant, and a SIM swap under 2 hours old forces the
  score to at least the block threshold (70) regardless of the blend.

Rule points: SIM swap < 2h (+55), < 48h (+35), older (+5); unknown device
(+15); amount > 50,000 (+10) or > 100,000 (+20); external IP (+8); 3 or more
failed attempts (+10). Thresholds come from `config.py`: ALLOW below 35, BLOCK
at 70 or above, CHALLENGE in between.

The model is a `RandomForestClassifier` over 8 features (`hours_since_sim_swap`,
`is_known_device`, `transaction_amount`, `is_external_ip`, `login_hour`,
`failed_attempts_1h`, `account_age_days`, `txns_last_24h`).

## Model training and the data-leakage fix

The model is trained on **synthetic data** (`ml/train_model.py`).

According to that script's docstring, the original reference version of this
idea generated every fraud row with `hours_since_sim_swap > 0` and every
legitimate row with `-1`. That encodes the label directly in one input
feature: the model doesn't learn a pattern, it learns "if this column is not
-1, output fraud". It scored AUC = 1.0, which is a symptom of data leakage,
not a good model.

This version fixes it two ways:

1. **A minority of legitimate users also have a past, harmless SIM swap**
   (12% of all users have some swap history, and a quarter of those swaps are
   old), so the feature is a strong signal, not a perfect one.
2. **The label is sampled from a probability**, a logistic function of the
   weighted features plus randomness, rather than assigned deterministically.
   Some noise is irreducible, as in a real fraud dataset.

Actual numbers, read from `ml/model_meta.json` (10,000 synthetic samples):

| Metric | Value |
|---|---|
| 5-fold CV ROC-AUC (train split) | 0.7466 (+/- 0.0151) |
| Test ROC-AUC | 0.7805 |
| Precision | 0.4406 |
| Recall | 0.5625 |
| F1 | 0.4941 |

These numbers are modest, and that is expected. Because both the data and the
labels are generated from a hand-written formula, they measure how well the
forest recovers that formula, **not** how it would perform against real
fraud. `/api/admin/model-meta` serves this file.

## Running it

```powershell
# from the project root (the app reads pages/ and creates simguard.db relative to it)
python -m venv venv                 # first time only
venv\Scripts\activate               # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# only if ml/fraud_model.pkl doesn't exist (it's committed, so usually skip)
python ml/train_model.py

uvicorn main:app --port 8001
```

Then open http://localhost:8001. **Use port 8001 rather than the default
8000**, since 8000 is occupied on some machines.

The SQLite database (`simguard.db`) is created automatically on first start.
Configuration comes from environment variables or a `.env` file
(`DATABASE_URL`, `JWT_SECRET`; see `config.py`).

## API endpoints

Read from `main.py`:

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/` | none | serves `pages/login.html` |
| GET | `/static/*` | none | serves `pages/` (`login.html`, `dashboard.html`, `admin.html`) |
| POST | `/api/register` | none | body: `email`, `phone`, `password`; rate limit 5/min |
| POST | `/api/login` | none | body: `email`, `password`, `device_id`; rate limit 10/min; returns `otp_token` |
| POST | `/api/verify-otp` | `otp_token` | query: `token`, `otp` (demo: any 6 digits); returns a JWT and `user_id` |
| POST | `/api/transfer` | JWT | body: `amount`, `device_id`; rate limit 10/min |
| GET | `/api/user/{user_id}` | JWT, own id only | returns `email`, `balance`; `403` for another user's id |
| GET | `/api/user/{user_id}/logins` | JWT, own id only | that user's 20 most recent login attempts |
| POST | `/api/admin/simulate-swap` | none | query: `user_id`; marks the SIM as just swapped |
| POST | `/api/admin/reset-swap` | none | query: `user_id` |
| GET | `/api/admin/fraud-logs` | none | 50 most recent login attempts, all users |
| GET | `/api/admin/users` | none | all users (no password hash) |
| GET | `/api/admin/stats` | none | login counts by action (24h), total users, total fraud alerts |
| GET | `/api/admin/model-meta` | none | contents of `ml/model_meta.json` |
| GET | `/api/health` | none | `{"status": "ok"}` |

## Tests

```powershell
pytest tests/ -v
```

There are currently **40 tests** (they all pass at the time of writing):
password hashing, JWT tampering, account lockout, the register / login /
OTP / transfer / user / admin API routes, the risk rules, and the carrier
client's guard clause. API tests use an in-memory SQLite database per test.

## Known limitations

- **All `/api/admin/*` routes are unauthenticated by design** (demo scope).
  They expose every user's email, phone and balance and can trigger a SIM
  swap. Don't deploy this as-is.
- **The OTP value is a demo, but the OTP step is tied to a real login.**
  `/api/verify-otp` requires the single-use `otp_token` that only a successful
  password + risk check (ALLOW or CHALLENGE, never BLOCK) can obtain, so a JWT
  can't be minted for an arbitrary `user_id`. Once you hold a valid token,
  though, *any* 6-digit code passes, because no SMS is sent or checked. Tokens
  that are never used are only removed when someone presents them after they
  expire, so unused rows accumulate in the `pending_otps` table.
- **CHALLENGE doesn't challenge.** ALLOW and CHALLENGE are handled the same by
  the server (see the flow above).
- **`JWT_SECRET` has a hardcoded development default** in `config.py`. Set it
  through the environment for anything beyond local use.
- **`vonage_client.py` is architecture only.** It sketches what a carrier
  SIM-swap lookup client would look like, but it is an unverified best-effort
  implementation of the expected request/response shape, has not been tested
  against the real Vonage API (no credentials available), and is **not wired
  into `risk_engine.py`** or `main.py`. Today the SIM-swap signal is
  `User.sim_swapped_at`, set by the admin simulate-swap endpoint. Unconfigured,
  `check_sim_swap()` raises `NotImplementedError`.
- **The ML model is trained on synthetic data**, so its metrics say little
  about real-world fraud detection (see above).
