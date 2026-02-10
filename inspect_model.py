import joblib
import os

model_path = "models/vehicle/best_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if hasattr(model, "feature_names_in_"):
        print(f"Features in: {model.feature_names_in_[:10]}... (Total: {len(model.feature_names_in_)})")
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        print(f"Booster feature names: {booster.feature_names[:10]}... (Total: {len(booster.feature_names)})")
else:
    print("Model not found")
