import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("LoanApprovalPrediction.csv")

# Drop the 'Loan_ID' column as it's not useful for prediction
data.drop(['Loan_ID'], axis=1, inplace=True)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = le.fit_transform(data[column])

# Split the data into features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('loan_model.pkl', 'rb') as file:
    model=pickle.load(file)