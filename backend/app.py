"""
SIH26165 - Safety Report Analyzer Backend
Flask REST API
"""

import sys
import os
import sqlite3
import json
import pickle
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Make the model/ directory importable
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
sys.path.insert(0, os.path.abspath(MODEL_DIR))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from life_saving_rules import detect_life_saving_rule, get_rule_scores
from hazard_extractor import extract_hazard_barrier_activity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "safety_reports_expanded_3018.csv"
)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports.db")

# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------
# If the highest class probability is below this, the prediction is flagged
# as low confidence.  0.50 (50%) is a conservative choice: the model must
# be at least as confident as a coin-flip for its top class.

CONFIDENCE_THRESHOLD = 50.0

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

print("[SIH26165] Loading dataset...")
data = pd.read_csv(DATASET_PATH)
X_all = data["report_text"].astype(str)
y_all = data["sif_potential"].astype(str)
CLASS_LABELS = ["HIGH", "MEDIUM", "LOW"]

# ---------------------------------------------------------------------------
# Train / test split  (80 / 20, stratified, fixed seed)
# ---------------------------------------------------------------------------

print("[SIH26165] Splitting dataset 80/20 (stratified, random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all,
)
print(f"[SIH26165] Train: {len(X_train)}  |  Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# Build and train the model  (fit on TRAIN only)
# ---------------------------------------------------------------------------

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "safety_model.pkl")

_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
    )),
])

if os.path.exists(MODEL_SAVE_PATH):
    print("[SIH26165] Loading existing model from disk...")
    with open(MODEL_SAVE_PATH, "rb") as f:
        _model = pickle.load(f)
else:
    print("[SIH26165] Training TF-IDF + LogisticRegression model on TRAIN split...")
    _model.fit(X_train, y_train)
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(_model, f)
    print("[SIH26165] Model saved to", MODEL_SAVE_PATH)

# ---------------------------------------------------------------------------
# Held-out evaluation on the TEST split
# ---------------------------------------------------------------------------

print("[SIH26165] Evaluating on held-out test set...")
y_pred = _model.predict(X_test)

_eval_accuracy = float(accuracy_score(y_test, y_pred))
_eval_precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
_eval_recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
_eval_f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

_cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
_per_class_report = classification_report(
    y_test, y_pred, labels=CLASS_LABELS, zero_division=0, output_dict=True
)

high_recall = float(_per_class_report.get("HIGH", {}).get("recall", 0.0))

_EVAL = {
    "accuracy": _eval_accuracy,
    "precision_macro": _eval_precision_macro,
    "recall_macro": _eval_recall_macro,
    "f1_macro": _eval_f1_macro,
    "confusion_matrix": _cm.tolist(),
    "class_labels": CLASS_LABELS,
    "per_class": {
        cls: {
            "precision": float(_per_class_report[cls]["precision"]),
            "recall": float(_per_class_report[cls]["recall"]),
            "f1-score": float(_per_class_report[cls]["f1-score"]),
            "support": int(_per_class_report[cls]["support"]),
        }
        for cls in CLASS_LABELS
    },
    "high_risk_recall": high_recall,
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "dataset_size": int(len(data)),
    "class_distribution": {
        cls: int((y_all == cls).sum()) for cls in CLASS_LABELS
    },
    "random_state": 42,
    "test_size": 0.20,
    "features": "TF-IDF unigrams and bigrams (1,2)-gram",
    "model_type": "TF-IDF + LogisticRegression",
}

print(
    f"[SIH26165] Evaluation — Accuracy: {_EVAL['accuracy']*100:.2f}% | "
    f"F1: {_EVAL['f1_macro']*100:.2f}% | "
    f"HIGH recall: {high_recall*100:.2f}%"
)

# ---------------------------------------------------------------------------
# Feature-weight explainability  (extract top TF-IDF terms per class)
# ---------------------------------------------------------------------------

_tfidf = _model.named_steps["tfidf"]
_classifier = _model.named_steps["classifier"]
_feature_names = np.array(_tfidf.get_feature_names_out())
_coef = _classifier.coef_

_top_features_per_class = {}
for idx, cls in enumerate(CLASS_LABELS):
    top_indices = np.argsort(_coef[idx])[-15:][::-1]
    _top_features_per_class[cls] = _feature_names[top_indices].tolist()


def get_key_indicators(report_text, prediction):
    """Return terms from the report text that have positive model weight for
    the predicted class."""
    report_lower = report_text.lower()
    tokens = set(re.findall(r"[a-z]+", report_lower))

    cls_idx = CLASS_LABELS.index(prediction)

    scored = []
    for token in tokens:
        if len(token) < 3:
            continue
        matches = np.where(_feature_names == token)[0]
        if len(matches) == 0:
            continue
        weight = float(_coef[cls_idx][matches[0]])
        if weight > 0:
            scored.append((token, weight))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [token for token, _ in scored[:5]]


# ---------------------------------------------------------------------------
# Explanation & recommended action
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SAFETY CONSISTENCY GUARD — Critical indicator detection
# ---------------------------------------------------------------------------
# Evidence weights:
#   barrier_failure  = 3  (direct evidence of control failure)
#   critical hazard  = 2  (strong hazard signal)
#   critical rule    = 1  (contextual safety rule)
#
# Override thresholds:
#   score >= 6  → final risk = HIGH
#   score >= 3  → final risk = MEDIUM (minimum for any override)
#   score <  3  → no override, ML prediction stands
#
# The guard is ALWAYS conservative: when in doubt, it escalates.

CRITICAL_HAZARDS = frozenset({
    "Toxic Atmosphere",
    "Fire / Explosion",
    "Unexpected Equipment Startup",
    "Electrical Shock",
    "Suspended Load",
    "Fall from Height",
    "Caught Between / Crushing",
    "Underground Utility Damage",
    "Vehicle Collision",
})

CRITICAL_BARRIERS = frozenset({
    "Gas Testing Not Performed",
    "Energy Isolation Not Verified",
    "Fall Protection Not Used",
    "Load Control / Exclusion Zone Failure",
    "Hot Work Controls Missing",
    "Excavation Authorization / Utility Check Missing",
    "Vehicle Spotter Missing",
    "Electrical Isolation Not Performed",
    "Standby Person Missing",
})

CRITICAL_RULES = frozenset({
    "Confined Space",
    "Energy Isolation",
    "Line of Fire",
    "Hot Work",
    "Working at Height",
    "Ground Disturbance",
    "Lifting Operations",
    "Driving and Vehicle Safety",
    "Electrical Safety",
})

WEIGHT_BARRIER = 3.0
WEIGHT_HAZARD = 2.0
WEIGHT_RULE = 1.0
OVERRIDE_THRESHOLD = 3.0
HIGH_ESCALATION_SCORE = 6.0

# Phrases that explicitly indicate controls/verification were in place.
# These CONTRADICT a barrier-failure or uncontrolled critical scenario
# and suppress a false escalation when no barrier failure is detected.
POSITIVE_CONTROL_PHRASES = [
    "proper ppe", "ppe was used", "ppe was worn", "wearing ppe",
    "wore ppe", "with proper ppe",
    "gas testing was performed", "gas test was performed",
    "gas testing completed", "gas test completed",
    "had adequate gas testing", "verified gas testing",
    "gas testing verified", "gas test verified",
    "had adequate ventilation", "verified ventilation",
    "ventilation was adequate", "was adequate",
    "permit was issued", "permit issued",
    "was isolated", "isolation was verified",
    "lockout applied", "loto applied",
    "fall protection was used", "harness was used", "harness was worn",
    "spotter was present", "using a spotter", "with a spotter",
]

SAFE_CONTEXT_PHRASES = [
    "verified", "verified that", "had adequate", "adequate", "confirmed",
    "calibrated", "was calibrated", "tested before", "tested prior",
    "was tested", "passed", "completed before", "properly used",
    "was present", "was used", "was worn", "wore", "wearing",
]


def _has_controls_evidence(report_text):
    """Check if the report explicitly indicates that safety controls were in use."""
    text = report_text.lower()
    return any(phrase in text for phrase in POSITIVE_CONTROL_PHRASES)


def _is_controlled_context(report_text, barrier_failure):
    """Report explicitly describes compliant, verified conditions AND no
    barrier failure is detected — do not escalate keyword-only matches."""
    if barrier_failure and barrier_failure != "Not detected":
        return False
    text = report_text.lower()
    return any(phrase in text for phrase in SAFE_CONTEXT_PHRASES)


def _compute_safety_guard(ml_prediction, rule, hazard, barrier_failure,
                          controlled_context):
    """Compute the safety consistency guard result.

    Two distinct concepts:
      - triggered   (risk_override):        the guard CHANGED the ML risk.
      - safety_review_required:             human safety review is required
        because the report contains critical safety evidence — even when
        the ML risk already matches the severity (e.g. ML=HIGH + barrier
        failure detected).

    Returns:
        dict with keys: triggered, evidence, evidence_score, final_risk,
        override_reason, review_reason, safety_review_required
    """
    evidence = []
    score = 0.0

    if hazard and hazard in CRITICAL_HAZARDS:
        score += WEIGHT_HAZARD
        evidence.append(f"{hazard} (hazard)")

    if rule and rule != "No clear Life-Saving Rule detected" and rule in CRITICAL_RULES:
        score += WEIGHT_RULE
        evidence.append(f"{rule} (life-saving rule)")

    if barrier_failure and barrier_failure != "Not detected" and barrier_failure in CRITICAL_BARRIERS:
        score += WEIGHT_BARRIER
        evidence.append(f"{barrier_failure} (barrier failure)")

    evidence_summary = ", ".join(evidence) if evidence else "None"

    # A report that explicitly describes compliant, verified conditions
    # (verified testing, PPE in use, permits issued, etc.) and shows NO
    # barrier failure provides positive evidence of control.  Keyword-only
    # critical matches (hazard/rule) do not warrant escalation or review.
    if controlled_context:
        return {
            "triggered": False,
            "evidence": evidence,
            "evidence_score": score,
            "final_risk": ml_prediction,
            "override_reason": None,
            "review_reason": None,
            "safety_review_required": False,
        }

    # ML already predicts HIGH.  The risk does not need to be overridden,
    # but critical safety evidence still requires human review.
    if ml_prediction == "HIGH":
        if evidence:
            review_reason = (
                f"ML prediction is already {ml_prediction} and critical "
                f"safety indicators were detected ({evidence_summary}). "
                "Human safety review is required to confirm the severity "
                "and verify corrective actions."
            )
        else:
            review_reason = None
        return {
            "triggered": False,
            "evidence": evidence,
            "evidence_score": score,
            "final_risk": "HIGH",
            "override_reason": None,
            "review_reason": review_reason,
            "safety_review_required": bool(evidence),
        }

    # ML is MEDIUM/LOW with insufficient evidence — no override, no review.
    if score < OVERRIDE_THRESHOLD:
        return {
            "triggered": False,
            "evidence": evidence,
            "evidence_score": score,
            "final_risk": ml_prediction,
            "override_reason": None,
            "review_reason": None,
            "safety_review_required": False,
        }

    # ML is MEDIUM/LOW with sufficient critical evidence — escalate.
    if score >= HIGH_ESCALATION_SCORE:
        final_risk = "HIGH"
        reason = (
            f"Multiple critical safety indicators detected "
            f"(evidence score: {score:.0f}): "
            + evidence_summary + ". "
            "The ML classifier predicted "
            f"{ml_prediction} risk, but the severity of detected indicators "
            "warrants escalation to HIGH risk."
        )
    else:
        final_risk = "MEDIUM"
        reason = (
            f"Critical safety indicators detected "
            f"(evidence score: {score:.0f}): "
            + evidence_summary + ". "
            "The ML classifier predicted "
            f"{ml_prediction} risk, but these indicators require safety review. "
            "Minimum risk set to MEDIUM pending investigation."
        )

    return {
        "triggered": True,
        "evidence": evidence,
        "evidence_score": score,
        "final_risk": final_risk,
        "override_reason": reason,
        "review_reason": reason,
        "safety_review_required": True,
    }


# ---------------------------------------------------------------------------
# Explanation (evidence-based — never claims controls are present without proof)
# ---------------------------------------------------------------------------

def build_explanation(ml_prediction, confidence, final_risk, override,
                      override_reason, review_reason, life_saving_rule,
                      hazard, activity, barrier_failure,
                      controls_evidence):
    """Build a dynamic, evidence-based explanation."""
    parts = []

    # --- Detection summary ---
    detection_parts = []
    if hazard and hazard != "Not detected":
        detection_parts.append(hazard)
    if life_saving_rule and life_saving_rule != "No clear Life-Saving Rule detected":
        detection_parts.append(life_saving_rule + " scenario")
    if barrier_failure and barrier_failure != "Not detected":
        detection_parts.append(barrier_failure)

    # --- Override path: ML prediction ≠ safety guard result ---
    if override:
        if detection_parts:
            parts.append(
                "SafetyAI identified critical indicators: "
                + "; ".join(detection_parts) + "."
            )
        parts.append(override_reason)
    else:
        # --- Normal (non-override) path ---
        prob_str = f"{confidence:.1f}%"
        parts.append(
            f"SafetyAI classified this report as {final_risk} risk "
            f"with {prob_str} confidence."
        )
        if detection_parts:
            parts.append(
                "Detected indicators: " + "; ".join(detection_parts) + "."
            )
        else:
            parts.append(
                "No specific hazard, barrier failure, or life-saving rule "
                "was detected in the report text."
            )
        # Review-required path even without override (e.g. ML HIGH already
        # matching the severity, but critical safety evidence present):
        if review_reason:
            parts.append(review_reason)

    # --- Controls evidence (only when explicitly found in report) ---
    if controls_evidence:
        parts.append(
            "The report indicates that safety controls were in use."
        )
    elif final_risk == "LOW" and not override:
        parts.append(
            "No critical safety indicators were detected and no explicit "
            "controls evidence was found in the report."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Recommended action (hazard/rule-specific, never claims controls in place)
# ---------------------------------------------------------------------------

_HAZARD_SPECIFIC_ADDITIONS = {
    "Toxic Atmosphere": (
        " Verify atmospheric testing (oxygen, flammable gas, toxic gas levels) "
        "before any entry or hot work."
    ),
    "Fire / Explosion": (
        " Remove ignition sources, maintain a fire watch, and verify "
        "gas-free conditions."
    ),
    "Unexpected Equipment Startup": (
        " Apply Lockout/Tagout (LOTO) to all energy sources and verify "
        "zero energy state."
    ),
    "Electrical Shock": (
        " De-energize the system, apply LOTO, and verify zero voltage "
        "before approaching."
    ),
    "Suspended Load": (
        " Clear personnel from beneath the load and establish an exclusion "
        "zone."
    ),
    "Fall from Height": (
        " Verify fall protection (harness, lanyard, anchor) is in place "
        "before any work at height."
    ),
    "Caught Between / Crushing": (
        " Establish machine guarding and ensure clear pathways free of "
        "pinch points."
    ),
    "Underground Utility Damage": (
        " Perform underground utility scanning and obtain a ground "
        "disturbance permit."
    ),
    "Vehicle Collision": (
        " Assign a trained spotter and establish pedestrian exclusion zones."
    ),
}

_RECOMMENDED_ACTIONS = {
    "Confined Space": (
        "Immediately stop entry. Ensure an entry permit is issued, "
        "atmospheric gas testing (oxygen, LEL, toxics) is completed, "
        "a standby person is stationed, and rescue equipment is available."
    ),
    "Energy Isolation": (
        "Apply full Lockout/Tagout (LOTO) procedure. Verify zero energy "
        "state before any maintenance work continues."
    ),
    "Line of Fire": (
        "Establish an exclusion zone. No personnel should be within the "
        "swing radius or drop zone. Halt lifting/movement until the area "
        "is cleared."
    ),
    "Hot Work": (
        "Suspend hot work immediately. Perform gas testing, obtain a valid "
        "hot work permit, assign a fire watch, and remove/protect "
        "flammable materials."
    ),
    "Working at Height": (
        "Stop work at height. Ensure all workers use approved fall "
        "protection (harness, lanyard). Inspect anchor points before "
        "resuming."
    ),
    "Ground Disturbance": (
        "Stop excavation. Obtain a ground disturbance permit, scan for "
        "underground utilities, and verify clearances before resuming work."
    ),
    "Lifting Operations": (
        "Halt all lifting. Inspect rigging, verify load weight, establish "
        "exclusion zones, and confirm a qualified rigger is supervising."
    ),
    "Driving and Vehicle Safety": (
        "Stop vehicle movement. Assign a trained spotter, establish "
        "pedestrian exclusion zones, and verify driver competency."
    ),
    "Electrical Safety": (
        "De-energize immediately. Apply LOTO, test for zero voltage, and "
        "ensure only qualified electricians work on energized systems."
    ),
}


def get_recommended_action(life_saving_rule, final_risk, ml_prediction,
                           hazard, barrier_failure, override):
    """Return recommended action based on detected rule, final risk, and
    whether a safety guard override was triggered."""
    parts = []

    if override:
        parts.append(
            "SAFETY REVIEW REQUIRED — Escalate to your safety supervisor "
            "before continuing this activity."
        )

    # Rule-specific primary action
    if life_saving_rule and life_saving_rule != "No clear Life-Saving Rule detected":
        action = _RECOMMENDED_ACTIONS.get(life_saving_rule)
        if action:
            parts.append(action)
    elif final_risk == "HIGH":
        parts.append(
            "STOP WORK IMMEDIATELY. Report to your supervisor and conduct "
            "a Job Hazard Analysis before resuming any activity."
        )
    elif final_risk == "MEDIUM":
        parts.append(
            "Review and strengthen existing controls before proceeding. "
            "Brief your team on the identified risk."
        )
    else:
        parts.append(
            "Continue standard observation and report any changes in "
            "conditions."
        )

    # Hazard-specific supplemental action
    if hazard and hazard != "Not detected" and hazard in _HAZARD_SPECIFIC_ADDITIONS:
        parts.append(_HAZARD_SPECIFIC_ADDITIONS[hazard])

    # Barrier-specific note (only when barrier failure detected and not
    # already covered by the rule-based action above)
    if (barrier_failure and barrier_failure != "Not detected"
            and barrier_failure in CRITICAL_BARRIERS):
        parts.append(
            f"Address the detected barrier failure: {barrier_failure}."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_report(report_text):
    report_text = str(report_text).strip()

    if not report_text:
        return {"error": "Empty report text provided."}

    # --- ML prediction ---
    ml_prediction = _model.predict([report_text])[0]
    probabilities = _model.predict_proba([report_text])[0]
    class_names = list(_model.classes_)
    confidence = round(float(max(probabilities)) * 100, 2)
    prob_dict = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(class_names, probabilities)
    }

    # --- Rule-based detections ---
    rule, rule_score = detect_life_saving_rule(report_text)
    rule_scores = get_rule_scores(report_text)
    details = extract_hazard_barrier_activity(report_text)

    # --- Feature-weight indicators ---
    key_indicators = get_key_indicators(report_text, ml_prediction)

    # --- Confidence flag ---
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    # --- Controls evidence check ---
    controls_evidence = _has_controls_evidence(report_text)
    controlled_context = _is_controlled_context(report_text, details["barrier_failure"])

    # --- SAFETY GUARD: consistency check between ML and detections ---
    guard = _compute_safety_guard(
        ml_prediction, rule, details["hazard"], details["barrier_failure"],
        controlled_context,
    )
    final_risk = guard["final_risk"]
    override = guard["triggered"]

    # --- Explanation (evidence-based) ---
    explanation = build_explanation(
        ml_prediction, confidence, final_risk, override,
        guard["override_reason"], guard["review_reason"], rule,
        details["hazard"], details["activity"], details["barrier_failure"],
        controls_evidence,
    )
    if low_confidence:
        explanation += (
            " Additionally, the ML model's confidence was below the "
            f"{CONFIDENCE_THRESHOLD}% threshold — human review is advised."
        )

    # --- Recommended action (reflects final risk + overrides) ---
    recommended_action = get_recommended_action(
        rule, final_risk, ml_prediction,
        details["hazard"], details["barrier_failure"], override,
    )

    # --- Assemble result ---
    result = {
        "report_text": report_text,

        # ML raw output
        "ml_prediction": ml_prediction,
        "ml_confidence": confidence,
        "class_probabilities": prob_dict,

        # Final risk assessment (after safety guard)
        "sif_potential": final_risk,
        "risk_level": final_risk,

        # Safety guard fields
        "risk_override": override,
        "override_reason": guard["override_reason"],
        "review_reason": guard["review_reason"],
        "safety_review_required": guard["safety_review_required"],
        "critical_evidence": guard["evidence"],
        "evidence_score": guard["evidence_score"],

        # Rule-based detections
        "life_saving_rule": rule,
        "rule_keyword_matches": rule_score,
        "rule_scores": rule_scores,
        "activity": details["activity"],
        "hazard": details["hazard"],
        "barrier_failure": details["barrier_failure"],
        "hazard_keyword_matches": details["hazard_keyword_matches"],
        "barrier_keyword_matches": details["barrier_keyword_matches"],
        "activity_keyword_matches": details["activity_keyword_matches"],

        # Confidence + explanation
        "confidence": confidence,
        "low_confidence_flag": low_confidence,
        "controls_evidence": controls_evidence,
        "key_indicators": key_indicators,
        "explanation": explanation,
        "recommended_action": recommended_action,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
    }

    return result


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                report_text        TEXT NOT NULL,
                sif_potential      TEXT,
                confidence         REAL,
                life_saving_rule   TEXT,
                activity           TEXT,
                hazard             TEXT,
                barrier_failure    TEXT,
                explanation        TEXT,
                recommended_action TEXT,
                full_result        TEXT,
                analyzed_at        TEXT
            )
        """)
        conn.commit()


init_db()


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
)

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "SIH26165 Safety Report Analyzer",
        "version": "2.0.0",
        "model": _EVAL["model_type"],
        "dataset_size": _EVAL["dataset_size"],
        "accuracy": round(_EVAL["accuracy"] * 100, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": _EVAL["model_type"],
        "accuracy_percent": round(_EVAL["accuracy"] * 100, 2),
        "precision_macro": round(_EVAL["precision_macro"] * 100, 2),
        "recall_macro": round(_EVAL["recall_macro"] * 100, 2),
        "f1_macro": round(_EVAL["f1_macro"] * 100, 2),
        "high_risk_recall": round(_EVAL["high_risk_recall"] * 100, 2),
        "dataset_size": _EVAL["dataset_size"],
        "training_set_size": _EVAL["training_samples"],
        "test_set_size": _EVAL["test_samples"],
        "class_distribution": _EVAL["class_distribution"],
        "class_labels": _EVAL["class_labels"],
        "per_class": _EVAL["per_class"],
        "confusion_matrix": _EVAL["confusion_matrix"],
        "features": _EVAL["features"],
        "random_state": _EVAL["random_state"],
        "test_size": _EVAL["test_size"],
        "life_saving_rules": 9,
        "hazard_categories": 9,
        "barrier_categories": 9,
        "activity_categories": 8,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(silent=True)
    if not body or "report_text" not in body:
        return jsonify({"error": "Missing 'report_text' in request body."}), 400

    result = analyze_report(body["report_text"])

    if "error" in result:
        return jsonify(result), 400

    with get_db() as conn:
        conn.execute(
            "INSERT INTO reports "
            "(report_text, sif_potential, confidence, life_saving_rule, "
            "activity, hazard, barrier_failure, explanation, "
            "recommended_action, full_result, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                result["report_text"],
                result["sif_potential"],
                result["confidence"],
                result["life_saving_rule"],
                result["activity"],
                result["hazard"],
                result["barrier_failure"],
                result["explanation"],
                result["recommended_action"],
                json.dumps(result),
                result["analyzed_at"],
            )
        )
        conn.commit()

    return jsonify(result)


@app.route("/api/reports", methods=["POST"])
def save_report():
    body = request.get_json(silent=True)
    if not body or "report_text" not in body:
        return jsonify({"error": "Missing 'report_text'."}), 400

    result = analyze_report(body["report_text"])
    if "error" in result:
        return jsonify(result), 400

    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO reports "
            "(report_text, sif_potential, confidence, life_saving_rule, "
            "activity, hazard, barrier_failure, explanation, "
            "recommended_action, full_result, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                result["report_text"],
                result["sif_potential"],
                result["confidence"],
                result["life_saving_rule"],
                result["activity"],
                result["hazard"],
                result["barrier_failure"],
                result["explanation"],
                result["recommended_action"],
                json.dumps(result),
                result["analyzed_at"],
            )
        )
        conn.commit()
        result["id"] = cur.lastrowid

    return jsonify(result), 201


@app.route("/api/reports", methods=["GET"])
def get_reports():
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, report_text, sif_potential, confidence, "
            "life_saving_rule, activity, hazard, barrier_failure, "
            "explanation, recommended_action, analyzed_at "
            "FROM reports ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        counts = conn.execute(
            "SELECT sif_potential, COUNT(*) as cnt FROM reports GROUP BY sif_potential"
        ).fetchall()

    count_dict = {r["sif_potential"]: r["cnt"] for r in counts}
    return jsonify({
        "total": total,
        "offset": offset,
        "limit": limit,
        "counts": count_dict,
        "reports": [dict(r) for r in rows],
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        counts = conn.execute(
            "SELECT sif_potential, COUNT(*) as cnt FROM reports GROUP BY sif_potential"
        ).fetchall()
        recent = conn.execute(
            "SELECT id, report_text, sif_potential, confidence, analyzed_at "
            "FROM reports ORDER BY id DESC LIMIT 5"
        ).fetchall()

    count_dict = {r["sif_potential"]: r["cnt"] for r in counts}
    return jsonify({
        "total_analyzed": total,
        "high_risk": count_dict.get("HIGH", 0),
        "medium_risk": count_dict.get("MEDIUM", 0),
        "low_risk": count_dict.get("LOW", 0),
        "recent": [dict(r) for r in recent],
    })


if __name__ == "__main__":
    print("[SIH26165] Starting Flask backend on http://127.0.0.1:8080")
    print(f"[SIH26165] Dataset size: {len(data)} rows")
    print(f"[SIH26165] Model: {_EVAL['model_type']}")
    print(f"[SIH26165] Accuracy: {_EVAL['accuracy']*100:.2f}% | F1: {_EVAL['f1_macro']*100:.2f}%")
    app.run(debug=True, host="127.0.0.1", port=8080, use_reloader=False)
