import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Pastikan kolom yang dibutuhkan ada di DataFrame
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code', 'Output']
for column in required_columns:
    if column not in data.columns:
        st.error(f"Kolom {column} tidak ditemukan di data.")
        st.stop()

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {}
    for column in label_encoders:
        if column in user_input:
            input_value = str(user_input[column])
            # Cek apakah nilai input ada dalam kelas yang dikenali
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])[0]
            else:
                # Jika tidak dikenali, Anda bisa memilih untuk mengabaikan nilai atau menggunakan nilai default
                st.warning(f"{input_value} tidak dikenali. Menggunakan nilai default.")
                processed_input[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])[0]  # Menggunakan kelas pertama sebagai nilai default
    processed_input = pd.DataFrame(processed_input, index=[0])
    
    for column in numeric_features:
        if column in user_input:
            processed_input[column] = user_input[column]
        else:
            st.warning(f"Kolom {column} tidak ditemukan dalam input pengguna.")
    
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# Antarmuka Streamlit
st.title("Prediksi Feedback Pelanggan Online Food")

# Tambahkan HTML kustom
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    </style>
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# Input pengguna
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed'])
monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
educational_qualifications = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'])
family_size = st.number_input('Family size', min_value=1, max_value=20)
latitude = st.number_input('Latitude', format="%f")
longitude = st.number_input('Longitude', format="%f")
pin_code = st.number_input('Pin code', min_value=100000, max_value=999999)
feedback = st.selectbox('Feedback', ['Positive', 'Negative'])

user_input = {
    'Age': age,
    'Gender': gender,
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code,
    'Feedback': feedback
}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    prediction = model.predict(user_input_processed)
    st.write(f'Prediction: {prediction[0]}')

# Tambahkan elemen HTML untuk output
st.markdown("""
    <h3>Output Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
