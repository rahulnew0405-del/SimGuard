"""
The risk engine: turns raw signals about a login into a 0-100 score and a
decision (ALLOW / CHALLENGE / BLOCK).

Why blend rules AND a model instead of just one?
  - Rules are transparent and instant to reason about ("SIM swapped in the
    last hour" should always be treated as dangerous, full stop — you don't
    want a model's learned weights to ever talk that down to a low score).
  - The ML model captures interactions between signals that are hard to
    hand-write as rules (e.g. "old account + known device + odd hour" is
    fine, but "new account + unknown device + odd hour" is not — the model
    learns that combination from data instead of someone enumerating every
    case).
  - Blending them (60% rules, 40% ML here) means a hard rule can't be
    silently overridden by the model, but the model still adds nuance on
    top of the rules for cases that aren't clear-cut.
"""
import json
import os
from datetime import datetime, timedelta

import joblib
import pandas as pd

from config import settings

ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
MODEL_PATH = os.path.join(ML_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
META_PATH = os.path.join(ML_DIR, "model_meta.json")

FEATURES = [
    "hours_since_sim_swap", "is_known_device", "transaction_amount",
    "is_external_ip", "login_hour", "failed_attempts_1h",
    "account_age_days", "txns_last_24h",
]


class RiskEngine:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)

    def assess(self, user, device_id: str, ip: str, amount: float, db) -> dict:
        flags = []
        rule_score = 0

        # 1. SIM swap recency — the core signal for this whole project.
        hours_since_swap = -1.0
        if user.sim_swapped_at:
            hours_since_swap = (datetime.utcnow() - user.sim_swapped_at).total_seconds() / 3600
            if hours_since_swap < 2:
                rule_score += 55
                flags.append("SIM_SWAP_CRITICAL")
            elif hours_since_swap < 48:
                rule_score += 35
                flags.append("SIM_SWAP_RECENT")
            else:
                rule_score += 5  # old swap, mostly harmless but still logged
                flags.append("SIM_SWAP_HISTORICAL")

        # 2. Device recognition
        known_device = self._is_known_device(user.id, device_id, db)
        if not known_device:
            rule_score += 15
            flags.append("UNKNOWN_DEVICE")

        # 3. Transaction amount
        if amount > 100000:
            rule_score += 20
            flags.append("VERY_HIGH_VALUE_TXN")
        elif amount > 50000:
            rule_score += 10
            flags.append("HIGH_VALUE_TXN")

        # 4. External IP
        external_ip = self._is_external_ip(ip)
        if external_ip:
            rule_score += 8
            flags.append("EXTERNAL_IP")

        # 5. Recent failed attempts
        failed = user.failed_attempts or 0
        if failed >= 3:
            rule_score += 10
            flags.append(f"FAILED_ATTEMPTS_{failed}")

        # 6. ML score
        ml_prob = self._ml_score(
            hours_since_sim_swap=hours_since_swap,
            is_known_device=int(known_device),
            transaction_amount=amount,
            is_external_ip=int(external_ip),
            login_hour=datetime.utcnow().hour,
            failed_attempts_1h=failed,
            account_age_days=self._account_age(user),
            txns_last_24h=self._txns_last_24h(user.id, db),
        )
        ml_score = ml_prob * 100

        final_score = int(min(0.60 * rule_score + 0.40 * ml_score, 100))

        # Hard override: a SIM swap in the last 2 hours always blocks,
        # regardless of what the blended score says.
        if "SIM_SWAP_CRITICAL" in flags:
            final_score = max(final_score, settings.RISK_BLOCK_AT_OR_ABOVE)

        if final_score >= settings.RISK_BLOCK_AT_OR_ABOVE:
            level, action = "HIGH", "BLOCK"
        elif final_score >= settings.RISK_ALLOW_BELOW:
            level, action = "MEDIUM", "CHALLENGE"
        else:
            level, action = "LOW", "ALLOW"

        return {
            "score": final_score, "level": level, "action": action,
            "flags": flags, "ml_prob": round(ml_prob, 4),
        }

    def _ml_score(self, **kwargs) -> float:
        X = pd.DataFrame([[kwargs[f] for f in FEATURES]], columns=FEATURES)
        X_s = self.scaler.transform(X)
        return float(self.model.predict_proba(X_s)[0][1])

    def _is_known_device(self, uid: int, device_id: str, db) -> bool:
        from models import KnownDevice
        row = db.query(KnownDevice).filter_by(user_id=uid, device_id=device_id).first()
        if not row:
            db.add(KnownDevice(user_id=uid, device_id=device_id))
            db.commit()
            return False
        return True

    def _is_external_ip(self, ip: str) -> bool:
        local = ("127.0.0.1", "localhost", "::1", "testclient")
        return ip not in local and not ip.startswith(("10.", "192.168.", "172."))

    def _account_age(self, user) -> int:
        return max((datetime.utcnow() - user.created_at).days, 1)

    def _txns_last_24h(self, uid: int, db) -> int:
        from models import Transaction
        since = datetime.utcnow() - timedelta(hours=24)
        return db.query(Transaction).filter(
            Transaction.user_id == uid, Transaction.created_at >= since
        ).count()


risk_engine = RiskEngine()
