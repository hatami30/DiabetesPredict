import streamlit as st
import pickle
import pandas as pd

# Muat model yang telah dilatih
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Fungsi untuk memproses data input
def preprocess_input(data):
    # Pemetaan nilai gender menjadi kode numerik
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    data['gender'] = data['gender'].map(gender_mapping)

    # Pemetaan nilai smoking history menjadi kode numerik
    smoking_history_mapping = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}
    data['smoking_history'] = data['smoking_history'].map(smoking_history_mapping)

    # Mengonversi 'age', 'bmi', dan 'HbA1c_level' menjadi tipe data float
    data['age'] = data['age'].astype(float)
    data['bmi'] = data['bmi'].astype(float)
    data['HbA1c_level'] = data['HbA1c_level'].astype(float)

    return data

# Fungsi untuk membuat prediksi
def predict_diabetes(data):
    # Lakukan preprocessing terhadap data input
    data = preprocess_input(data)

    # Lakukan prediksi
    predictions = model.predict(data)
    return predictions

# Aplikasi web utama
def main():
    # Atur judul dan deskripsi aplikasi web
    st.title("Aplikasi Prediksi Diabetes")
    st.write("Masukkan informasi yang diperlukan untuk melakukan prediksi diabetes")

    # Buat field input untuk pengguna memasukkan data
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=1.0, max_value=100.0, format="%.1f")
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ["No Info", "never", "former", "current", "not current", "ever"])
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, format="%.2f")
    hba1c_level = st.number_input("HbA1c Level", min_value=1.0, max_value=10.0, format="%.1f")
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=1, max_value=500)

    # Buat tombol untuk melakukan prediksi
    if st.button("Predict"):
        # Buat dictionary dari data input
        input_data = {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c_level],
            "blood_glucose_level": [blood_glucose_level]
        }

        # Konversi dictionary menjadi dataframe
        input_df = pd.DataFrame(input_data)

        # Lakukan prediksi
        predictions = predict_diabetes(input_df)

        # Tampilkan hasil prediksi
        if predictions[0] == 0:
            st.write("Anda tidak memiliki diabetes.")
        else:
            st.write("Anda memiliki diabetes.")



# Jalankan aplikasi web
if __name__ == "__main__":
    main()
