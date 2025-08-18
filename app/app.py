from flask import Flask, request, render_template
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    try:
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        # Retention recommendation
        if prediction == 1:
            recommendation = "Offer Discount or Personalized Retention Plan"
        else:
            recommendation = "Customer Likely to Stay, Continue Monitoring"

        return render_template(
            "index.html",
            prediction_text=f"Churn Prediction: {'Yes' if prediction==1 else 'No'} | Action: {recommendation}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
