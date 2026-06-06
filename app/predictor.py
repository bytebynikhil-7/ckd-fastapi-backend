import joblib

models = {}

def load_models():
    global models

    models["random_forest"] = joblib.load("models/RandomForest_model.pkl")
    models["adaboost"] = joblib.load("models/AdaBoost_model.pkl")
    models["gradient_boosting"] = joblib.load("models/GradientBoosting_model.pkl")

def predict(model_name, features):
    model = models[model_name]

    prediction = model.predict([features])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        confidence = round(max(probabilities) * 100, 1)
    else:
        confidence = 90.0

    return prediction, confidence
