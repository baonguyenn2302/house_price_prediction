"""Flask server for the house price prediction web application."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")

app = Flask(__name__)


def optional_float(value, field_name):
    """Convert an optional numeric input to a float or NaN."""
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} phải là một số hợp lệ.") from error


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Dữ liệu đầu vào không hợp lệ.")

        area = optional_float(data.get("Area"), "Diện tích")
        if np.isnan(area) or area <= 0:
            raise ValueError("Diện tích phải lớn hơn 0.")

        input_data = pd.DataFrame([{
            "Area": area,
            "Frontage": optional_float(data.get("Frontage"), "Mặt tiền"),
            "Access Road": optional_float(data.get("Access_Road"), "Đường vào"),
            "House direction": data.get("House_direction") or np.nan,
            "Balcony direction": data.get("Balcony_direction") or np.nan,
            "Floors": optional_float(data.get("Floors"), "Số tầng"),
            "Bedrooms": optional_float(data.get("Bedrooms"), "Phòng ngủ"),
            "Bathrooms": optional_float(data.get("Bathrooms"), "Phòng tắm"),
            "Legal status": data.get("Legal_status") or np.nan,
            "Furniture state": data.get("Furniture_state") or np.nan,
            "Location": data.get("Location") or "Hồ Chí Minh",
        }])
        prediction = max(0, float(model.predict(input_data)[0]))

        return jsonify({
            "prediction": round(prediction, 2),
            "formatted_prediction": f"{prediction:,.2f}",
            "unit": "tỷ VNĐ",
            "model_name": "SVR",
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
