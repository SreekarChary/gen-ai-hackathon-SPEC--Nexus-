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
    """Dashboard — shows model performance metrics for selected category."""
    import joblib, config
    category = session.get("category", config.DEFAULT_CATEGORY)
    
    metrics = None
    try:
        metrics_path = os.path.join(config.get_model_dir(category), "metrics.pkl")
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
    except Exception:
        pass

    return render_template("index.html", metrics=metrics, category=category, categories=config.CATEGORIES)


@app.route("/select_category/<category>")
def select_category(category):
    """Update selected category in session."""
    import config
    if category in config.CATEGORIES:
        session["category"] = category
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET"])
def predict_form():
    """Render the category-specific claim submission form."""
    import config
    category = session.get("category", config.DEFAULT_CATEGORY)
    template = f"predict_{category}.html" if category != "vehicle" else "predict.html"
    return render_template(template, category=category)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, run prediction, store result in session."""
    import config
    from src.predict import predict_claim
    
    category = session.get("category", config.DEFAULT_CATEGORY)
    claim_data = {key: value for key, value in request.form.items()}

    # Run prediction for the specific category
    result = predict_claim(claim_data, category=category)

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
