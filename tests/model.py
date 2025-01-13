import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv('Train_withlabel_121624.csv')

# Select features and target
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Create the model instance
model = LogisticRegression(class_weight='balanced', random_state=5)

# Fit the model
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:  # Use 'wb' for writing in binary mode
    pickle.dump(model, f)  # Corrected to dump

