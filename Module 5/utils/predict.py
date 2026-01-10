import joblib
from utils.preprocess import preprocess_input
from config import HABITABILITY_THRESHOLD

xgb_model = joblib.load("models/xgboost_reg.pkl")
xgbc_model = joblib.load("models/xgboost_classifier.pkl")


def predict_habitability(input_data):
    X = preprocess_input(input_data)
    

    score = float(xgb_model.predict(X)[0])
    label = int(score >= HABITABILITY_THRESHOLD)

    return {
        "habitability_score": round(score, 4),
        "habitability_class": label
    }
