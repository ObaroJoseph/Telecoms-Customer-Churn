from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    call_failure= int(request.form['call_failure'])
    complains = int(request.form['complains'])
    subscription_length = int(request.form['subscription_length'])
    charge_amount = int(request.form['charge_amount'])
    seconds_of_use = int(request.form['seconds_of_use'])
    frequency_of_use = int(request.form['frequency_of_use'])
    frequency_of_sms = int(request.form['frequency_of_sms'])
    distinct_called_numbers = int(request.form['distinct_called_numbers'])
    age_group = int(request.form['age_group'])
    tariff_plan = int(request.form['tariff_plan'])
    status = int(request.form['status'])
    customer_value = int(request.form['customer_value'])

    # Create a feature vector from the input values
    features = np.array([call_failure, complains, subscription_length, charge_amount, seconds_of_use,
                         frequency_of_use, frequency_of_sms, distinct_called_numbers, age_group,
                         tariff_plan, status, customer_value]).reshape(1, -1)

    # Make the prediction
    churn_prediction = model.predict(features)

    # Convert the prediction to a human-readable result
    result = 'Churn' if churn_prediction[0] == 1 else 'No Churn'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)



    
