"""LoanIQ - loan approval prediction powered by a trained DecisionTreeClassifier."""
import logging
import math
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("loaniq")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# Exact feature order the model was trained with.
FEATURE_ORDER = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file",
]

# Categorical codes = alphabetical label encoding (sklearn LabelEncoder / pandas
# category order). Inferred from the split thresholds in decision.pkl; confirm them
# against the notebook you trained with (see README notes in the chat reply).
#   gender:    female=0, male=1
#   education: Associate=0, Bachelor=1, Doctorate=2, High School=3, Master=4
#   home:      MORTGAGE=0, OTHER=1, OWN=2, RENT=3
#   intent:    DEBTCONSOLIDATION=0, EDUCATION=1, HOMEIMPROVEMENT=2, MEDICAL=3, PERSONAL=4, VENTURE=5
#   defaults:  No=0, Yes=1
ICONS = {
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "card": '<rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "gauge": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
}

FIELDS = [
    {"name": "person_age", "label": "Age", "short": "Age", "group": "Applicant", "kind": "int",
     "min": 18, "max": 100, "step": 1, "icon": "user", "placeholder": "e.g. 29",
     "hint": "Applicant's age in years."},
    {"name": "person_gender", "label": "Gender", "short": "Gender", "group": "Applicant", "kind": "select",
     "icon": "users", "options": [("Female", 0), ("Male", 1)], "hint": "As recorded in the training data."},
    {"name": "person_education", "label": "Education", "short": "Education", "group": "Applicant", "kind": "select",
     "icon": "book", "hint": "Highest level completed.",
     "options": [("High school", 3), ("Associate", 0), ("Bachelor", 1), ("Master", 4), ("Doctorate", 2)]},
    {"name": "person_income", "label": "Annual income (USD)", "short": "Income", "group": "Applicant", "kind": "float",
     "min": 1, "max": 10000000, "step": 1, "icon": "dollar", "placeholder": "e.g. 65000",
     "hint": "Yearly income before tax."},
    {"name": "person_emp_exp", "label": "Work experience (years)", "short": "Experience", "group": "Applicant",
     "kind": "int", "min": 0, "max": 80, "step": 1, "icon": "briefcase", "placeholder": "e.g. 4",
     "hint": "Total years of employment."},
    {"name": "person_home_ownership", "label": "Home ownership", "short": "Home", "group": "Applicant", "kind": "select",
     "icon": "home", "hint": "Current housing situation.",
     "options": [("Rent", 3), ("Mortgage", 0), ("Own", 2), ("Other", 1)]},
    {"name": "loan_amnt", "label": "Loan amount (USD)", "short": "Loan amount", "group": "Loan", "kind": "float",
     "min": 1, "max": 10000000, "step": 1, "icon": "card", "placeholder": "e.g. 12000",
     "hint": "Amount requested."},
    {"name": "loan_intent", "label": "Loan purpose", "short": "Purpose", "group": "Loan", "kind": "select",
     "icon": "target", "hint": "What the loan is for.",
     "options": [("Debt consolidation", 0), ("Education", 1), ("Home improvement", 2),
                 ("Medical", 3), ("Personal", 4), ("Venture", 5)]},
    {"name": "loan_int_rate", "label": "Interest rate (%)", "short": "Interest rate", "group": "Loan", "kind": "float",
     "min": 0, "max": 40, "step": 0.01, "icon": "percent", "placeholder": "e.g. 11.5",
     "hint": "Annual rate offered on the loan."},
    {"name": "loan_percent_income", "label": "Loan as share of income", "short": "Loan % income", "group": "Loan",
     "kind": "computed", "icon": "pie", "hint": "Calculated: loan amount ÷ annual income."},
    {"name": "cb_person_cred_hist_length", "label": "Credit history (years)", "short": "Credit history",
     "group": "Credit profile", "kind": "float", "min": 0, "max": 80, "step": 1, "icon": "clock",
     "placeholder": "e.g. 6", "hint": "How long the applicant has had credit."},
    {"name": "credit_score", "label": "Credit score", "short": "Credit score", "group": "Credit profile",
     "kind": "int", "min": 300, "max": 850, "step": 1, "icon": "gauge", "placeholder": "300 to 850",
     "hint": "Credit bureau score."},
    {"name": "previous_loan_defaults_on_file", "label": "Previous loan defaults", "short": "Past defaults",
     "group": "Credit profile", "kind": "select", "icon": "shield", "hint": "Any default on record.",
     "options": [("No", 0), ("Yes", 1)]},
]
assert [f["name"] for f in FIELDS] == FEATURE_ORDER
FIELD_BY_NAME = {f["name"]: f for f in FIELDS}


def field_groups():
    groups = []
    for f in FIELDS:
        if not groups or groups[-1]["title"] != f["group"]:
            groups.append({"title": f["group"], "fields": []})
        groups[-1]["fields"].append({**f, "icon_svg": ICONS[f["icon"]]})
    return groups


def load_model():
    try:
        with open(MODEL_PATH, "rb") as fh:
            m = pickle.load(fh)
        names = list(getattr(m, "feature_names_in_", []))
        if names != FEATURE_ORDER:
            raise ValueError(f"Unexpected feature names: {names}")
        return m
    except Exception:
        log.exception("Could not load %s", MODEL_PATH)
        return None


model = load_model()


def build_insights(m):
    if m is None:
        return None
    t = m.tree_
    root = t.value[0][0]
    importances = sorted(
        ({"name": n, "label": FIELD_BY_NAME[n]["label"].split(" (")[0], "value": round(float(v), 4)}
         for n, v in zip(m.feature_names_in_, m.feature_importances_)),
        key=lambda d: -d["value"],
    )
    # First three levels of the real tree, for the hero illustration.
    nodes, queue = [], [(0, 0, -1)]
    while queue:
        nid, depth, parent = queue.pop(0)
        v = t.value[nid][0]
        internal = t.children_left[nid] != -1 and depth < 3
        nodes.append({
            "id": nid, "parent": parent, "depth": depth, "terminal": not internal,
            "label": FIELD_BY_NAME[m.feature_names_in_[t.feature[nid]]]["short"] if internal else "",
            "approved": bool(v[1] >= v[0]),
        })
        if internal:
            queue.append((int(t.children_left[nid]), depth + 1, nid))
            queue.append((int(t.children_right[nid]), depth + 1, nid))
    return {
        "importances": importances,
        "tree": nodes,
        "facts": {
            "algorithm": "Decision Tree Classifier",
            "criterion": str(m.criterion).title(),
            "depth": int(m.get_depth()),
            "nodes": int(t.node_count),
            "leaves": int(m.get_n_leaves()),
            "features": int(m.n_features_in_),
            "training_records": int(t.n_node_samples[0]),
            "approved_share": round(float(root[1] / root.sum()), 4),
        },
    }


INSIGHTS = build_insights(model)


def parse_form(form):
    """Validate the 12 user-entered fields and derive loan_percent_income."""
    values, errors = {}, {}
    for f in FIELDS:
        name = f["name"]
        if f["kind"] == "computed":
            continue
        raw = (form.get(name) or "").strip()
        if not raw:
            errors[name] = "This field is required."
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[name] = "Enter a number."
            continue
        if not math.isfinite(num):
            errors[name] = "Enter a valid number."
            continue
        if f["kind"] == "select":
            if num not in {c for _, c in f["options"]} or not num.is_integer():
                errors[name] = "Choose one of the listed options."
                continue
            values[name] = int(num)
            continue
        if f["kind"] == "int" and not num.is_integer():
            errors[name] = "Enter a whole number."
            continue
        if num < f["min"] or num > f["max"]:
            errors[name] = f"Enter a value between {f['min']:g} and {f['max']:g}."
            continue
        values[name] = int(num) if f["kind"] == "int" else num

    if not errors:
        age = values["person_age"]
        if values["person_emp_exp"] > age:
            errors["person_emp_exp"] = "Work experience can't be more than the applicant's age."
        if values["cb_person_cred_hist_length"] > age:
            errors["cb_person_cred_hist_length"] = "Credit history can't be longer than the applicant's age."
    if not errors:
        # Training data stores this ratio rounded to two decimals.
        values["loan_percent_income"] = round(values["loan_amnt"] / values["person_income"], 2)
    return values, errors


def run_prediction(values):
    row = {n: float(values[n]) for n in model.feature_names_in_}
    X = pd.DataFrame([row], columns=list(model.feature_names_in_))
    pred = int(model.predict(X)[0])
    path_steps = int(model.decision_path(X).nnz) - 1
    return {
        "approved": pred == 1,
        "prediction": pred,
        "label": "Loan approved" if pred == 1 else "Loan rejected",
        "path_steps": path_steps,
        "loan_percent_income": values["loan_percent_income"],
        # Not reported: the tree is fully grown, so predict_proba is always 0 or 1
        # and would not be a meaningful confidence score.
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        groups=field_groups(),
        insights=INSIGHTS,
        model_ready=model is not None,
        approved_pct=round(INSIGHTS["facts"]["approved_share"] * 100) if INSIGHTS else None,
    )


@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="The prediction model is unavailable right now. Please try again later."), 503
    try:
        values, errors = parse_form(request.form)
        if errors:
            return jsonify(error="Check the highlighted fields and try again.", errors=errors), 422
        return jsonify(result=run_prediction(values))
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="We couldn't process this application. Please check your entries and try again."), 500


@app.get("/health")
def health():
    return jsonify(status="ok" if model is not None else "model_unavailable"), (200 if model is not None else 503)


@app.errorhandler(404)
def not_found(_):
    return jsonify(error="That page doesn't exist."), 404


@app.errorhandler(405)
def bad_method(_):
    return jsonify(error="That action isn't supported."), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="The submitted data is too large."), 413


@app.errorhandler(500)
def server_error(_):
    return jsonify(error="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
