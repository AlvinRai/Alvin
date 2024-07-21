import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('/mnt/data/onlinefoods.csv')

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    for column in label_encoders:
        user_input[column] = label_encoders[column].transform([user_input[column]])[0]
    user_input = pd.DataFrame(user_input, index=[0])
    user_input[numeric_features] = scaler.transform(user_input[numeric_features])
    return user_input

# Antarmuka Streamlit
st.title("Prediksi Feedback Pelanggan Online Food")

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

