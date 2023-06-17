import streamlit as st
import pickle
import pandas as pd

# Muat model yang telah dilatih
filename = 'diabetes-predict-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Pra-pemrosesan data
df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1, 'Other': 2})
df['smoking_history'] = df['smoking_history'].replace({'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5})

# Definisikan fungsi prediksi
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Lakukan pra-pemrosesan pada data input
    data['gender'] = data['gender'].replace({'Female': 0, 'Male': 1, 'Other': 2})
    data['smoking_history'] = data['smoking_history'].replace({'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5})

    # Lakukan prediksi
    prediction = classifier.predict(data)

    return prediction[0]

# Tampilan aplikasi web menggunakan Streamlit
def main():
    st.title("Website Prediksi Diabetes")

    # Tampilkan form input
    st.subheader("Masukkan Data Pasien")
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
    age = st.number_input("Usia", min_value=0.0, step=0.1, format="%.1f")
    hypertension = st.selectbox("Hipertensi", [0, 1])
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
    smoking_history = st.selectbox("Riwayat Merokok", ['No Info', 'never', 'former', 'current', 'not current', 'ever'])
    bmi = st.number_input("BMI", min_value=0.00, format="%.2f")
    HbA1c_level = st.number_input("Tingkat HbA1c", min_value=0.0, step=0.1, format="%.1f")
    blood_glucose_level = st.number_input("Tingkat Glukosa Darah", min_value=0, step=1)

    # Prediksi saat tombol "Prediksi" ditekan
    if st.button("Prediksi"):
        result = predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
        if result == 0:
            st.success("Anda tidak memiliki diabetes.")
        else:
            st.error("Anda memiliki diabetes.")

if __name__ == '__main__':
    main()
