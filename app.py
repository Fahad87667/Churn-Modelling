import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoded_GEO.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app configuration
st.set_page_config(page_title="Customer Churn Prediction", layout='wide')

# Custom CSS to make images round
st.markdown("""
<style>
    img {
        border-radius: 5%;  /* Adjust to 50% for circular images */
        object-fit: cover;    /* Ensures the image covers the entire container */
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.title('Customer Churn Prediction App')
st.image('churn2.png',width=600)
st.markdown('**:dart: This Streamlit app predicts customer churn for a fictional business.**')



# Create Input Form for Individual Customer Prediction

with st.form(key='customer_details_form'):
    st.subheader("Enter Customer Details")
    
    # Input fields
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0.0, step=0.01)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=0.01)
    tenure = st.slider('Tenure (in years)', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

    # Form submission
    submit_button = st.form_submit_button(label='Predict')
    
st.sidebar.header(':bulb: Prediction Result')
st.sidebar.image('churn.png', use_column_width=True)
# Convert categorical inputs after form submission
if submit_button:
    # Preprocess the data as required for prediction
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display prediction results
    st.sidebar.write(f'**Churn Probability**: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.sidebar.error('The customer is likely to churn.')
    else:
        st.sidebar.success('The customer is not likely to churn.')
