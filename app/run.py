import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    if request.method == 'POST':
        # Collect all input data
        bene_id = request.form['BeneID']  # Collect BeneID
        claim_id = request.form['ClaimID']  # Collect ClaimID
        
        input_data = [
            float(bene_id),  # Include BeneID as a feature
            float(claim_id),  # Include ClaimID as a feature
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
            float(request.form['is_dead']),
            float(request.form['Num_UniqueClmDiagnosis']),
            float(request.form['Num_UniqueProcedureCode']),
            float(request.form['NumPhysicians']),
            float(request.form['NoOfMonths_PartACov']),
            float(request.form['NoOfMonths_PartBCov']),
            float(request.form['IPAnnualReimbursementAmt']),
            float(request.form['IPAnnualDeductibleAmt']),
            float(request.form['OPAnnualReimbursementAmt']),
            float(request.form['OPAnnualDeductibleAmt']),
            float(request.form['Age'])
        ]
        
        # Convert the input data to a DataFrame for prediction
        X = pd.DataFrame([input_data], columns=[
            'BeneID', 'ClaimID', 'InscClaimAmtReimbursed', 'DeductibleAmtPaid', 
            'RenalDiseaseIndicator', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 
            'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 
            'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 
            'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 
            'AdmissionDays', 'ClaimDays', 'is_dead', 'Num_UniqueClmDiagnosis', 
            'Num_UniqueProcedureCode', 'NumPhysicians', 'NoOfMonths_PartACov', 
            'NoOfMonths_PartBCov', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 
            'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age'
        ])

        # Make the prediction
        pred_proba = model.predict_proba(X)[0][1]  # Get the probability of fraud
        pred = pred_proba  # Store the prediction probability

    return render_template('index.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
