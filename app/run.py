from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None  # Initialize the prediction variable

    if request.method == 'POST':  # Check if the request is POST
        # Extract data from the form and convert it to float
        input_data = [
            float(request.form['InscClaimAmtReimbursed']),
            float(request.form['DeductibleAmtPaid']),
            float(request.form['RenalDiseaseIndicator']),
            float(request.form['ChronicCond_Alzheimer']),
            float(request.form['ChronicCond_Heartfailure']),
            float(request.form['ChronicCond_KidneyDisease']),
            float(request.form['ChronicCond_Cancer']),
            float(request.form['ChronicCond_ObstrPulmonary']),
            float(request.form['ChronicCond_Depression']),
            float(request.form['ChronicCond_Diabetes']),
            float(request.form['ChronicCond_IschemicHeart']),
            float(request.form['ChronicCond_Osteoporasis']),
            float(request.form['ChronicCond_rheumatoidarthritis']),
            float(request.form['ChronicCond_stroke']),
            float(request.form['AdmissionDays']),
            float(request.form['ClaimDays']),
            float(request.form['is_dead']),  # Treating as a float (0 or 1)
            float(request.form['Num_UniqueClmDiagnosis']),
            float(request.form['Num_UniqueProcedureCode']),
            float(request.form['NumPhysicians']),
        ]

        # Convert the input data into a pandas DataFrame for prediction
        X = pd.DataFrame([input_data], columns=[
            'InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
            'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
            'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
            'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'AdmissionDays', 'ClaimDays', 'is_dead',
            'Num_UniqueClmDiagnosis', 'Num_UniqueProcedureCode', 'NumPhysicians'
        ])

        # Make the prediction and get the probability of fraud
        pred_proba = model.predict_proba(X)[0][1]
        pred = pred_proba  # Store the predicted probability of fraud

    return render_template('index.html', pred=pred)  # Return the result to the template

if __name__ == '__main__':
    app.run(debug=True)

