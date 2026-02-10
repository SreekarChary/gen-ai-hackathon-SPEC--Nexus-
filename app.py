"""
ClaimWatch AI — Flask Web Application (Module 4)
Serves the UI for submitting claims and viewing fraud predictions.
"""
import os
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")


# ─── Routes ────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard — shows model performance metrics."""
    # Try to load saved metrics; fall back to placeholder if not trained yet.
    metrics = None
    try:
        import joblib, config
        metrics_path = os.path.join(config.MODELS_DIR, "metrics.pkl")
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
    except Exception:
        pass

    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the claim submission form."""
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, run prediction, store result in session."""
    from src.predict import predict_claim

    # Collect all form fields into a dict
    claim_data = {key: value for key, value in request.form.items()}

    # Run prediction
    result = predict_claim(claim_data)

    # Store in session for the result page
    session["result"] = result
    session["claim_data"] = claim_data
    return redirect(url_for("result"))


@app.route("/result")
def result():
    """Display prediction result with reasoning."""
    result = session.get("result")
    claim_data = session.get("claim_data")
    if not result:
        return redirect(url_for("predict_form"))
    return render_template("result.html", result=result, claim_data=claim_data)


# ─── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(debug=debug, port=port)
