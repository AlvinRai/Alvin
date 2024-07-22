# Online Food Feedback Prediction

This is a Streamlit application for predicting customer feedback on online food services using a machine learning model. The application allows users to input customer data and receive a prediction about the feedback.

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone [https://github.com/username/online-food-feedback.git](https://github.com/AlvinRai/Alvin.git)
    cd online-food-feedback
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:
    Place your `onlinefoods.csv` dataset in the `data` directory.

4. **Run the app**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **Input Data**: Enter the required details in the Streamlit interface for features such as Age, Gender, Marital Status, Occupation, Monthly Income, Educational Qualifications, Family size, Latitude, Longitude, and Pin code.

2. **Prediction**: Click the "Predict" button. The app will process the input and provide a prediction for customer feedback.

## Deployment

This app can be deployed on Streamlit Cloud. Follow the instructions on the [Streamlit deployment guide](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html) to deploy your app.

## Files

- **app.py**: Main application file containing Streamlit code.
- **requirements.txt**: Contains the list of dependencies required to run the app.
- **data/onlinefoods.csv**: Dataset file.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-feature-branch`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or contact the repository owner.

## Code Overview

### Data Preprocessing and Model Training

The code includes the following steps for preprocessing and model training:

1. **Handling Missing Values**:
    ```python
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    ```

2. **Encoding Categorical Variables**:
    ```python
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
    numerical_features = ['Age', 'Family size', 'latitude', 'longitude']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    ```

3. **Splitting the Dataset**:
    ```python
    from sklearn.model_selection import train_test_split
    X = df.drop('Output', axis=1)
    y = df['Output']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Training the Models**:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))
    ```

5. **Saving the Best Model**:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    best_model = RandomForestClassifier()
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, 'best_model.pkl')
    ```

### Streamlit Interface

The Streamlit app allows users to input customer data and receive predictions:

```python
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and dataset for encoding and scaling
model = joblib.load('best_model.pkl')
data = pd.read_csv('/mnt/data/onlinefoods.csv')

required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']
data = data[required_columns]

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

def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                processed_input[column] = label_encoders[column].transform(['Unknown'])
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

st.title("Prediksi Feedback Pelanggan Online Food")

st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    </style>
    <h3>Masukkan Data Pelanggan</h3>
""", unsafe_allow_html=True)

# User input fields
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
    'Pin code': pin_code
}

if st.button('Predict'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.write(f'Prediction: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

st.markdown("""
    <h3>Output Prediksi</h3>
    <p>Hasil prediksi akan ditampilkan di sini.</p>
""", unsafe_allow_html=True)
