import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv('011525updated data.csv')

# Select features and target
# Features (excluding 'ClaimID' and 'BeneID' as they are identifiers)
X = df[['InscClaimAmtReimbursed', 'DeductibleAmtPaid',
       'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
       'AdmissionDays', 'ClaimDays', 'is_dead', 'Num_UniqueClmDiagnosis',
       'Num_UniqueProcedureCode', 'NumPhysicians']]

# Target column
y = df['PotentialFraud']  # Assuming 'Fraud' is your target column


# Create the model instance
model = LogisticRegression(class_weight='balanced', random_state=5)

# Fit the model
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:  # Use 'wb' for writing in binary mode
    pickle.dump(model, f)  # Corrected to dump

