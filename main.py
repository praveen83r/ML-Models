from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Logistic Regression Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting data as a JSON request
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
