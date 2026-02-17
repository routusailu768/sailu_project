from flask import Flask, render_template, request
import joblib
import numpy as np
import requests # Needed for Step 3: Weather API requests

app = Flask(__name__)

# Step 2: Load the model and scaler
model = joblib.load('wind_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# Step 3 & 4: Configure app.py for API and prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Capture inputs from the UI
    speed = float(request.form['speed'])
    direction = float(request.form['direction'])
    theo = float(request.form['theo'])
    
    # Prepare data and predict
    features = np.array([[speed, direction, theo]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # Mechanism: Maintenance check
    status = "HEALTHY"
    if (theo - prediction) > 500:
        status = "MAINTENANCE REQUIRED"

    return render_template('index.html', 
                           prediction=round(prediction, 2), 
                           status=status)

if __name__ == "__main__":
    # Step 5: Run the app
    app.run(debug=True)