# Food Delivery Time Prediction

This project predicts food delivery times in Bengaluru using restaurant and customer location data, restaurant type, preparation time, and order hour. It uses geocoding, feature engineering, and a machine learning model (Random Forest) to estimate delivery times.

## Features
- Geocodes restaurant addresses using Nominatim.
- Simulates customer locations and order data.
- Trains a Random Forest regression model with hyperparameter tuning.
- Provides a Streamlit web app for easy predictions.

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a Python virtual environment:**
   ```sh
   python -m venv l2_project_env
   l2_project_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the data acquisition and model training scripts:**
   ```sh
   python 01_acquire_and_geocode_data.py
   python 02_train_and_save_model.py
   ```

5. **Launch the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

## Model Details

- **Input features:** Restaurant name, customer location, restaurant type, preparation time, order hour, geodesic distance, peak hour indicator.
- **Output:** Estimated delivery time (minutes).
- **Model:** Random Forest Regressor with GridSearchCV for hyperparameter optimization.
