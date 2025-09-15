# QuickBites: Food Delivery Time Prediction

[![QuickBites UI](quickbites.png)](https://quickbites.streamlit.app/)

## 🚀 Quick Start

Try the app instantly:
- **Live Demo:** [https://quickbites.streamlit.app/](https://quickbites.streamlit.app/)
- No setup required—just click and use!

## 📝 Project Overview
QuickBites predicts food delivery times in Bengaluru based on restaurant and customer locations, restaurant type, preparation time, and order hour. It’s designed for:
- Food delivery enthusiasts
- Data science learners
- Anyone interested in location-based ML applications

The app uses geocoding, feature engineering, and a Random Forest model to estimate delivery times. You can run it locally or use the live demo above.

## Features
- Geocodes restaurant addresses using Nominatim.
- Simulates customer locations and order data.
- Trains a Random Forest regression model with hyperparameter tuning.
- Provides a Streamlit web app for easy predictions.

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/balanivansh/L2-project-QuickBites.git
   cd L2-project-QuickBites
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
