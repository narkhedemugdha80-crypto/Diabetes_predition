import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML & CSS Template combined inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Portal</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 30px;
            font-size: 2rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.9rem;
            color: #475569;
        }
        input, select {
            padding: 12px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #fff;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        .btn-submit {
            grid-column: 1 / -1;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 10px;
        }
        .btn-submit:hover {
            background-color: var(--primary-hover);
        }
        .btn-submit:active {
            transform: scale(0.99);
        }
        .result-card {
            margin-top: 30px;
            padding: 20px;
            background: #e0e7ff;
            border-left: 6px solid var(--primary);
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: bold;
            color: #312e81;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>SVM Model Prediction Portal</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" step="any" name="age" required>
                </div>
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select name="gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="city">City</label>
                    <input type="text" name="city" required>
                </div>
                <div class="form-group">
                    <label for="bmi">BMI</label>
                    <input type="number" step="any" name="bmi" required>
                </div>
                <div class="form-group">
                    <label for="family_history_diabetes">Family History of Diabetes</label>
                    <select name="family_history_diabetes" required>
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="physical_activity_level">Physical Activity Level</label>
                    <select name="physical_activity_level" required>
                        <option value="Low">Low</option>
                        <option value="Moderate">Moderate</option>
                        <option value="High">High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="diet_type">Diet Type</label>
                    <input type="text" name="diet_type" required>
                </div>
                <div class="form-group">
                    <label for="smoking_status">Smoking Status</label>
                    <select name="smoking_status" required>
                        <option value="Non-smoker">Non-smoker</option>
                        <option value="Smoker">Smoker</option>
                        <option value="Former">Former</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="alcohol_consumption">Alcohol Consumption</label>
                    <input type="text" name="alcohol_consumption" required>
                </div>
                <div class="form-group">
                    <label for="hours_sleep_per_night">Hours Sleep Per Night</label>
                    <input type="number" step="any" name="hours_sleep_per_night" required>
                </div>
                <div class="form-group">
                    <label for="stress_level">Stress Level</label>
                    <input type="text" name="stress_level" required>
                </div>
                <div class="form-group">
                    <label for="fasting_blood_sugar">Fasting Blood Sugar</label>
                    <input type="number" step="any" name="fasting_blood_sugar" required>
                </div>
                <div class="form-group">
                    <label for="hba1c_level">HbA1c Level</label>
                    <input type="number" step="any" name="hba1c_level" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_systolic">Blood Pressure Systolic</label>
                    <input type="number" step="any" name="blood_pressure_systolic" required>
                </div>
                <div class="form-group">
                    <label for="blood_pressure_diastolic">Blood Pressure Diastolic</label>
                    <input type="number" step="any" name="blood_pressure_diastolic" required>
                </div>
                <div class="form-group">
                    <label for=" waist_circumference_cm">Waist Circumference (cm)</label>
                    <input type="number" step="any" name="waist_circumference_cm" required>
                </div>
                <div class="form-group">
                    <label for="income_bracket">Income Bracket</label>
                    <input type="text" name="income_bracket" required>
                </div>
            </div>
            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction_text %}
            <div class="result-card">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract numerical features directly from form inputs
            # Note: If your model expects encoded categorical variables, map/encode them appropriately here.
            features = [
                float(request.form['age']),
                # Categorical features parsed as raw strings (Modify if your preprocessing pipeline requires numeric encoding)
                request.form['gender'],
                request.form['city'],
                float(request.form['bmi']),
                request.form['family_history_diabetes'],
                request.form['physical_activity_level'],
                request.form['diet_type'],
                request.form['smoking_status'],
                request.form['alcohol_consumption'],
                float(request.form['hours_sleep_per_night']),
                request.form['stress_level'],
                float(request.form['fasting_blood_sugar']),
                float(request.form['hba1c_level']),
                float(request.form['blood_pressure_systolic']),
                float(request.form['blood_pressure_diastolic']),
                float(request.form['waist_circumference_cm']),
                request.form['income_bracket']
            ]
            
            # Reshape input for model prediction
            # Depending on whether `svm.pkl` is wrapped in a sklearn Pipeline (which handles categorical encoding automatically) 
            # or raw SVC, you can pass features directly:
            prediction = model.predict([features])
            prediction_text = f"Prediction Result: {prediction[0]}"
        except Exception as e:
            prediction_text = f"Error during prediction: {str5(e) if 'str5' else str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.host = '0.0.0.0'
    app.run(debug=True)
