# -*- coding: utf-8 -*-
"""Model_Diabetes_Predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C96vRKkjDZyUET-ZCKUd6N68NaHa1cfk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Import Data CSV
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Melihat beberapa baris pertama dari dataset
data.head(10)

# Melihat informasi dataset
data.info()

# Pra-pemrosesan data
data['gender'] = data['gender'].replace({'Female': 0, 'Male': 1, 'Other': 2})
data['smoking_history'] = data['smoking_history'].replace({'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5})

# Memisahkan fitur dan target
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

# Normalisasi fitur numerik
scaler = StandardScaler()
X[['age', 'bmi', 'HbA1c_level']] = scaler.fit_transform(X[['age', 'bmi', 'HbA1c_level']])

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Regresi Logistik
model = LogisticRegression()

# Melatih model dengan data training
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

# Menghitung precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Menghitung recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Menghitung f1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Membuat array nama metrik dan nilai metrik
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

# Membuat grafik batang untuk menampilkan metrik akurasi
plt.bar(metrics, values)
plt.ylim(0, 1)  # Mengatur batas sumbu y antara 0 dan 1
plt.ylabel('Value')
plt.title('Accuracy Metrics')
plt.show()

# Simpan model ke dalam file pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Menampilkan data model
print(model)