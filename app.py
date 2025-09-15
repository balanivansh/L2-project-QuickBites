"""
app.py

Streamlit app for predicting food delivery time.
"""

import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

def load_components():
    """Load model, encoder, and location data."""
    try:
        model = joblib.load('best_model.joblib')
        encoder = joblib.load('one_hot_encoder.joblib')
        df_restaurants = joblib.load('restaurants_geocoded.joblib')
        df_customers = joblib.load('customers_geocoded.joblib')
        return model, encoder, df_restaurants, df_customers
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run '02_train_and_save_model.py' first.")
        st.stop()

def get_user_input(df_restaurants, df_customers):
    """Render input widgets and return user selections."""
    st.header("Enter Delivery Details")
    col1, col2 = st.columns(2)
    with col1:
        restaurant_name = st.selectbox(
            "Restaurant Location",
            options=df_restaurants['restaurant_name'].unique()
        )
    with col2:
        customer_location = st.selectbox(
            "Customer Location",
            options=df_customers['customer_location'].unique()
        )
    prep_time = st.slider("Restaurant Prep Time (minutes)", min_value=5, max_value=60, value=20)
    order_hour = st.slider("Order Hour (24h format)", min_value=0, max_value=23, value=19)
    return restaurant_name, customer_location, prep_time, order_hour

def predict_delivery_time(model, encoder, df_restaurants, df_customers, restaurant_name, customer_location, prep_time, order_hour):
    """Prepare features and predict delivery time."""
    res_loc = df_restaurants[df_restaurants['restaurant_name'] == restaurant_name].iloc[0]
    cust_loc = df_customers[df_customers['customer_location'] == customer_location].iloc[0]
    dist_km = geodesic(
        (res_loc['latitude'], res_loc['longitude']),
        (cust_loc['latitude'], cust_loc['longitude'])
    ).km
    input_df = pd.DataFrame([{
        'restaurant_name': restaurant_name,
        'customer_location': customer_location,
        'restaurant_type': res_loc['restaurant_type'],
    }])
    encoded_input = encoder.transform(input_df)
    encoded_df = pd.DataFrame(
        encoded_input,
        columns=encoder.get_feature_names_out(['restaurant_name', 'customer_location', 'restaurant_type'])
    )
    is_peak_hour = 1 if 18 <= order_hour <= 21 else 0
    input_data = pd.DataFrame({
        'prep_time_minutes': [prep_time],
        'is_peak_hour': [is_peak_hour],
        'distance_km': [dist_km]
    })
    final_input = pd.concat([input_data, encoded_df], axis=1)
    train_columns = model.feature_names_in_
    final_input = final_input.reindex(columns=train_columns, fill_value=0)
    predicted_time = model.predict(final_input)[0]
    return predicted_time

def main():
    st.set_page_config(page_title="QuickBites")
    st.markdown("<h1 style='text-align: center;'>QuickBites</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Your Fast Food Delivery Time Predictor 🛵</h4>", unsafe_allow_html=True)
    st.markdown("---")
    model, encoder, df_restaurants, df_customers = load_components()
    restaurant_name, customer_location, prep_time, order_hour = get_user_input(df_restaurants, df_customers)
    if st.button("Predict Delivery Time", type="primary"):
        with st.spinner('Calculating...'):
            predicted_time = predict_delivery_time(
                model, encoder, df_restaurants, df_customers,
                restaurant_name, customer_location, prep_time, order_hour
            )
        st.markdown("---")
        st.success(f"### Estimated Delivery Time: **{predicted_time:.2f} minutes**")
        st.balloons()

if __name__ == "__main__":
    main()