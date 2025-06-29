from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("student_model.pkl")  # Load trained model

@app.route('/')
def home():
    return render_template('index.html')  # Load UI

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get form data as JSON
    df = pd.DataFrame([data])  # Convert to DataFrame
    prediction = model.predict(df)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)