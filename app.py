from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    """
    Render the homepage with the form for user input.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and make predictions.
    """
    try:
        # Get data from the form
        age = float(request.form['age'])
        gender = 1 if request.form['gender'] == 'female' else 0
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Create input array for the model
        input_data = np.array([[age, gender, height, weight, duration, heart_rate, body_temp]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=f"Estimated Calories Burned: {prediction:.2f}")
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
