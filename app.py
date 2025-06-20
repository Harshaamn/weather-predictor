from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = tf.keras.models.load_model("gru_model.h5")
scaler = joblib.load("scaler.gz")
df = pd.read_csv("weather_results.csv")

features = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M',
            'ALLSKY_SFC_SW_DWN', 'WS2M', 'ALLSKY_SFC_PAR_TOT']
SEQ_LEN = 5

@app.route('/')
def index():
    districts = sorted(df['District'].unique())
    return render_template("index.html", districts=districts)

@app.route('/predict', methods=["POST"])
def predict():
    district = request.form['district']
    year = int(request.form['year'])

    df_district = df[df['District'] == district].sort_values("Year")
    df_district['Year'] = df_district['Year'].astype(int)

    past_data = df_district[df_district['Year'] < year].tail(SEQ_LEN)
    if len(past_data) < SEQ_LEN:
        return jsonify({"error": "❌ Not enough data before that year!"})

    X_input = scaler.transform(past_data[features]).reshape(1, SEQ_LEN, len(features))
    y_pred = model.predict(X_input)
    y_pred = scaler.inverse_transform(y_pred)[0]

    result = {features[i]: round(y_pred[i], 2) for i in range(len(features))}
    return jsonify({
        "district": district,
        "year": year,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
