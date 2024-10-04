# Customer Churn Prediction Web App

## Description
The **Customer Churn Prediction Web App** is a web application designed to predict customer churn for a fictional bank or financial institution. By analyzing various customer attributes, the app provides insights into the likelihood of a customer discontinuing their service. It utilizes a sophisticated Artificial Neural Network (ANN) model trained on historical customer data.

## Features
- **User-Friendly Interface**: An intuitive design that allows users to easily input customer details.
- **Real-Time Predictions**: Users can receive instant predictions on customer churn probability based on the provided information.
- **Interactive Visuals**: Results and predictions are displayed dynamically, enhancing user engagement and understanding.
- **Customizable Inputs**: Users can adjust various parameters to see how changes affect churn predictions.

## Feature Engineering
Feature engineering plays a critical role in the performance of the churn prediction model. Key steps include:

1. **Data Collection**: Gathering historical data from customer records to identify relevant features.
2. **Categorical Encoding**: Transforming categorical variables (e.g., gender, geography) into numerical formats suitable for model training. Techniques used include:
   - Label Encoding for binary categories.
   - One-Hot Encoding for multi-class categories.
3. **Scaling**: Normalizing numerical features (e.g., credit score, balance) using techniques like Min-Max scaling to ensure that all input features contribute equally to the model's learning process.
4. **Feature Selection**: Identifying the most significant features that influence customer churn, thereby improving model efficiency and performance.

## Deep Learning
The app employs a **Deep Learning** approach using an **Artificial Neural Network (ANN)** to predict customer churn. Key aspects include:

- **Model Architecture**: The ANN model consists of multiple layers, including:
  - Input Layer: Accepts the customer input features.
  - Hidden Layers: One or more layers with activation functions (e.g., ReLU) to capture complex patterns in the data.
  - Output Layer: A single neuron with a sigmoid activation function to output the probability of churn.
- **Training Process**: The model is trained on a labeled dataset using backpropagation and gradient descent to minimize the loss function. Key hyperparameters include:
  - Learning Rate
  - Number of Epochs
  - Batch Size
- **Evaluation Metrics**: Model performance is assessed using metrics like accuracy, precision, recall, and AUC-ROC.

## Streamlit
The web app is built using **Streamlit**, a powerful framework for creating data-driven applications. Key features include:

- **Interactive Widgets**: Users can input data using various widgets like sliders, dropdowns, and radio buttons.
- **Real-Time Feedback**: The app instantly provides predictions and feedback based on user inputs.
- **Custom Styling**: The app utilizes CSS for styling, ensuring a visually appealing user experience.


