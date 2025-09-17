import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import requests
import os

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_real_world_distance_and_time(start_coords, end_coords):
    """
    Calls the Google Maps Directions API to get travel distance and time.
    """
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": "driving",  
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            leg = data["routes"][0]["legs"][0]
            travel_time_minutes = leg["duration"]["value"] / 60
            road_distance_km = leg["distance"]["value"] / 1000
            polyline = data["routes"][0]["overview_polyline"]["points"]
            return road_distance_km, travel_time_minutes, polyline
        else:
            st.warning("Directions API could not find a valid route.")
            return None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Directions API request failed: {e}")
        return None, None, None

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
    start_coords = (res_loc['latitude'], res_loc['longitude'])
    end_coords = (cust_loc['latitude'], cust_loc['longitude'])
    dist_km, travel_time, polyline = get_real_world_distance_and_time(start_coords, end_coords)
    if dist_km is None:
        return None, None, None, None
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
    
    static_map_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?size=600x400"
        f"&markers=color:green|label:S|{start_coords[0]},{start_coords[1]}"
        f"&markers=color:red|label:D|{end_coords[0]},{end_coords[1]}"
        f"&path=enc:{polyline}"
        f"&key={GOOGLE_MAPS_API_KEY}"
    )
    return predicted_time, static_map_url, start_coords, end_coords

def main():
    st.set_page_config(page_title="QuickBites")
    st.markdown("<h1 style='text-align: center;'>QuickBites</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Your Fast Food Delivery Time Predictor 🛵</h4>", unsafe_allow_html=True)
    st.markdown("---")
    model, encoder, df_restaurants, df_customers = load_components()
    restaurant_name, customer_location, prep_time, order_hour = get_user_input(df_restaurants, df_customers)
    if st.button("Predict Delivery Time", type="primary"):
        with st.spinner('Calculating...'):
            predicted_time, static_map_url, start_coords, end_coords = predict_delivery_time(
                model, encoder, df_restaurants, df_customers,
                restaurant_name, customer_location, prep_time, order_hour
            )
        st.markdown("---")
        if predicted_time is not None:
            st.success(f"### Estimated Delivery Time: **{predicted_time:.2f} minutes**")
            st.markdown("#### Route Visualization:")
            st.image(static_map_url, caption="Route from Restaurant to Customer", use_container_width=True)
            st.balloons()

if __name__ == "__main__":
    main()