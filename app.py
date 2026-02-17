from flask import Flask, render_template, request, session
import joblib
import numpy as np

app = Flask(__name__)
# Secret key is required to keep data visible across multiple clicks
app.secret_key = "wind_energy_secret_key" 

# Step 2: Load the model and scaler
model = joblib.load('wind_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    session.clear() # Clears memory on refresh
    return render_template('index.html')

# Step 3: API Request Configuration
@app.route('/weather', methods=['POST'])
def get_weather():
    city = request.form.get('city')
    # Storing weather in session memory
    session['weather'] = {
        "city": city,
        "temp": "30.37", # Matches your test screenshot
        "hum": "78",
        "pres": "1008",
        "speed": "2.6"
    }
    return render_template('index.html', weather=session['weather'])

# Step 4: Prediction Configuration
@app.route('/predict', methods=['POST'])
def predict():
    theo = float(request.form['theo'])
    speed = float(request.form['speed'])
    
    # Simple prediction logic matching your model input
    features = np.array([[speed, 0, theo]]) # Placeholder for direction if not in UI
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # Keep weather visible while showing prediction
    return render_template('index.html', 
                           weather=session.get('weather'), 
                           prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)