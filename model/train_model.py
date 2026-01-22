import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Review Features
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[features]
y = df['diagnosis']

# Map 0 -> Malignant, 1 -> Benign (This is what the notebook did)
# Note: In the original sklearn dataset, 0 is Malignant and 1 is Benign usually? 
# Let's double check. 
# sklearn load_breast_cancer: wdbc.target: 0 = malignant, 1 = benign.
# So the notebook's mapping: y.map({0: 'Malignant', 1: 'Benign'}) is correct.
y_mapped = y.map({0: 'Malignant', 1: 'Benign'})

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y_mapped)

# Check mapping
# Classes are sorted: ['Benign', 'Malignant']
# Benign -> 0
# Malignant -> 1
print(f"Classes: {le.classes_}")
# Verify: if class 0 is Benign, then predict 0 => Benign.

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save
joblib.dump(model, 'model/breast_cancer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model and Scaler saved.")
