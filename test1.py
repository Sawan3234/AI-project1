import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras

# Load data
df = pd.read_csv("customer_churn.csv")

# Drop 'customerID' column as it's not useful for modeling
df.drop('customerID', axis='columns', inplace=True)

# Convert TotalCharges column to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with NaN in 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])

# Ensure TotalCharges is numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Visualize tenure distribution for churned and non-churned customers
tenure_churn_no = df[df.Churn == 'No'].tenure
tenure_churn_yes = df[df.Churn == 'Yes'].tenure
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization (Tenure)")
plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.legend()
plt.show()

# Replace 'No internet service' and 'No phone service' with 'No'
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Replace 'Yes'/'No' with 1/0 for categorical columns
yes_no_columns = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'PaperlessBilling', 'Churn'
]
for col in yes_no_columns:
    if col in df.columns:  # Ensure the column exists
        df[col] = df[col].replace({'Yes': 1, 'No': 0})

# Replace 'Female'/'Male' with 1/0 in gender column
df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})

# One-hot encode categorical columns
df = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Scale numerical columns
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Split data into features (X) and target (Y)
X = df.drop('Churn', axis='columns')
Y = df['Churn']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(X.shape[1],), activation='relu'),
    keras.layers.Dense(15, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10)

# Make predictions
yp = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = [1 if prob > 0.5 else 0 for prob in yp]

# Evaluate the model
print(classification_report(Y_test, y_pred))
