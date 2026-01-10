from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
import joblib
import os


# Configuration
DB_PATH = "exoplanets.db"
CLEANED_CSV = "data/exoplanet_feature_engineered_dataset.csv"
HABITABILITY_THRESHOLD = 0.7


# Load Models
reg_model = joblib.load("models/xgboost_reg.pkl")
cls_model = joblib.load("models/xgboost_classifier.pkl")
MODEL_FEATURES = joblib.load("models/model_features.pkl")


# Flask App
app = Flask(__name__)
app.config["DEBUG"] = True

# Database Utilities
def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS planets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        planet_name TEXT,
        st_teff REAL,
        st_rad REAL,
        st_mass REAL,
        st_met REAL,
        st_luminosity REAL,
        pl_orbper REAL,
        pl_orbeccen REAL,
        pl_insol REAL,
        source TEXT
    )
    """)

    conn.commit()
    conn.close()

def load_cleaned_dataset():
    if not os.path.exists(CLEANED_CSV):
        return

    df = pd.read_csv(CLEANED_CSV)
    df = df[["pl_name"] + MODEL_FEATURES]
    df["source"] = "dataset"

    conn = get_db()
    df.rename(columns={"pl_name": "planet_name"}, inplace=True)
    df.to_sql("planets", conn, if_exists="append", index=False)
    conn.close()


# Initialize DB
init_db()

# Load dataset only ONCE
if not os.path.exists("db_initialized.flag"):
    load_cleaned_dataset()
    open("db_initialized.flag", "w").close()


# Standard Response
def response(status, message, data=None):
    return jsonify({
        "status": status,
        "message": message,
        "data": data
    })


# Routes

@app.route("/", methods=["GET"])
def home():
    return response(
        "success",
        "Exoplanet Habitability API running",
        {
            "endpoints": ["/add_planet", "/predict", "/rank"]
        }
    )


# POST /add_planet
@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.get_json()

    try:
        row = {
            "planet_name": data.get("planet_name", "Unknown"),
            **{f: data[f] for f in MODEL_FEATURES},
            "source": "user"
        }

        df = pd.DataFrame([row])
        conn = get_db()
        df.to_sql("planets", conn, if_exists="append", index=False)
        conn.close()

        return response("success", "Planet added successfully")

    except Exception as e:
        return response("error", str(e)), 400


# POST /predict
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        input_df = pd.DataFrame([data])
        input_df = input_df[MODEL_FEATURES]

        score = float(reg_model.predict(input_df)[0])
        cls = int(cls_model.predict(input_df)[0])
        confidence = float(cls_model.predict_proba(input_df)[0][1])

        return response(
            "success",
            "Prediction generated successfully",
            {
                "planet_name": data.get("planet_name", "Unknown"),
                "habitability": cls,
                "score": round(score, 4),
                "confidence": round(confidence, 4)
            }
        )

    except Exception as e:
        return response("error", str(e)), 400

# GET /rank
@app.route("/rank", methods=["GET"])
def rank():
    top_n = int(request.args.get("top", 10))

    conn = get_db()
    df = pd.read_sql("SELECT * FROM planets", conn)
    conn.close()

    X = df[MODEL_FEATURES]
    df["habitability_score"] = reg_model.predict(X)

    ranked = (
        df[["planet_name", "habitability_score"]]
        .drop_duplicates()
        .sort_values("habitability_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ranked["rank"] = ranked.index + 1

    return response(
        "success",
        "Ranking generated successfully",
        ranked.to_dict(orient="records")
    )


if __name__ == "__main__":
    app.run()
