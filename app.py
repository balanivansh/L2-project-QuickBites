from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import os
import folium
from streamlit_folium import st_folium
import polyline
from datetime import datetime, timedelta

# Load the Google Maps API key from environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

@st.cache_data
def get_traffic_forecast_data(start_coords, end_coords, order_hour):
    # Gets real-time, traffic-aware travel time and the route polyline from the Google Maps Directions API.
    now = datetime.now()
    departure_dt = now.replace(hour=order_hour, minute=0, second=0, microsecond=0)
    if departure_dt < now:
        departure_dt += timedelta(days=1)
    departure_timestamp = int(departure_dt.timestamp())

    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": "driving",
        "departure_time": departure_timestamp,
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            leg = data["routes"][0]["legs"][0]
            road_distance_km = leg["distance"]["value"] / 1000
            if 'duration_in_traffic' in leg:
                travel_time_minutes = leg["duration_in_traffic"]["value"] / 60
            else:
                travel_time_minutes = leg["duration"]["value"] / 60
            encoded_polyline = data["routes"][0]["overview_polyline"]["points"]
            return road_distance_km, travel_time_minutes, encoded_polyline
        else:
            error_status = data.get("status", "UNKNOWN_ERROR")
            error_msg = data.get("error_message", "No error message provided by API.")
            st.error(f"Directions API Error: {error_status} - {error_msg}")
            return None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Directions API request failed: {e}")
        return None, None, None

def calculate_delivery_fee(distance_km, order_hour):
    # Calculates a simple delivery fee based on distance and a peak-hour surge.
    base_fee, per_km_charge, peak_hour_surge = 30, 8, 25
    fee = base_fee + (distance_km * per_km_charge)
    if 18 <= order_hour <= 21:
        fee += peak_hour_surge
    return fee

@st.cache_data
def load_components():
    # Caches the loading of all necessary model and data files.
    try:
        model = joblib.load('best_model.joblib')
        encoder = joblib.load('one_hot_encoder.joblib')
        df_restaurants = joblib.load('restaurants_geocoded.joblib')
        df_customers = joblib.load('customers_geocoded.joblib')
        return model, encoder, df_restaurants, df_customers
    except FileNotFoundError:
        st.error("Error: Model or data files not found. Please run scripts 01 and 02 first.")
        st.stop()

def predict_delivery_time(model, encoder, df_restaurants, df_customers, restaurant_name, customer_location, prep_time, order_hour, day_of_week):
    # Predicts the delivery time using the loaded model and real-time API data.
    res_loc = df_restaurants[df_restaurants['restaurant_name'] == restaurant_name].iloc[0]
    cust_loc = df_customers[df_customers['customer_location'] == customer_location].iloc[0]
    
    start_coords, end_coords = (res_loc['latitude'], res_loc['longitude']), (cust_loc['latitude'], cust_loc['longitude'])
    
    dist_km, travel_time, encoded_polyline = get_traffic_forecast_data(start_coords, end_coords, order_hour)
    
    if dist_km is None:
        return None, None, None, None, None, None
        
    # Prepares the user's input data for the model.
    input_df = pd.DataFrame([{'restaurant_name': restaurant_name, 'customer_location': customer_location,
                              'restaurant_type': res_loc['restaurant_type'], 'day_of_week': day_of_week}])
    
    encoded_input = encoder.transform(input_df)
    encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(input_df.columns))
    
    input_data = pd.DataFrame({
        'prep_time_minutes': [prep_time],
        'distance_km': [dist_km],
        'travel_time_minutes': [travel_time],
        'order_hour': [order_hour]
    })
    
    final_input = pd.concat([input_data, encoded_df], axis=1)
    
    # Ensures the input features are in the same order as the training data.
    train_columns = model.feature_names_in_
    final_input = final_input.reindex(columns=train_columns, fill_value=0)
    
    predicted_time = model.predict(final_input)[0]
    
    return predicted_time, start_coords, end_coords, travel_time, dist_km, encoded_polyline

def main():
    st.set_page_config(page_title="QuickBites Predictor", layout="wide")

    st.title("🛵 Welcome to the QuickBites Predictor!")
    st.markdown("Enter your delivery details below to get a real-time ETA forecast for your food delivery in Bengaluru.", unsafe_allow_html=True)
    st.markdown("---")
    
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    # Load all required files.
    model, encoder, df_restaurants, df_customers = load_components()

    st.header("⚙️ Enter Delivery Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        restaurant_options = df_restaurants['display_name'].unique()
        try:
            bk_index = list(restaurant_options).index('Burger King, Koramangala')
        except ValueError:
            bk_index = 0
        selected_display_name = st.selectbox("Choose a Restaurant", options=restaurant_options, index=bk_index)
        restaurant_name = selected_display_name.split(', ')[0]

    with col2:
        customer_options = df_customers['customer_location'].unique()
        try:
            wf_index = list(customer_options).index('Whitefield')
        except ValueError:
            wf_index = 0
        customer_location = st.selectbox("Choose Customer Location", options=customer_options, index=wf_index)

    with col3:
        day_options = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_name = st.selectbox("Day of the Week", options=day_options.keys(), index=0)
        day_of_week = day_options[day_name]

    c1, c2 = st.columns(2)
    with c1:
        prep_time = st.slider("Restaurant Prep Time (minutes)", 5, 60, 20, 5)
    with c2:
        order_hour = st.slider("Order Hour (24h format)", 0, 23, 17)

    if st.button("Predict Delivery Time 🚀", type="primary", use_container_width=True):
        with st.spinner('Forecasting your delivery time...'):
            st.session_state.prediction_results = predict_delivery_time(
                model, encoder, df_restaurants, df_customers,
                restaurant_name, customer_location, prep_time, order_hour, day_of_week
            )

    if st.session_state.prediction_results:
        predicted_time, start_coords, end_coords, travel_time, dist_km, encoded_polyline = st.session_state.prediction_results

        st.markdown("---")
        if predicted_time is not None:
            st.subheader("📈 Prediction Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total ETA", f"{predicted_time:.0f} min")
            c2.metric("Food Prep Time", f"{prep_time:.0f} min")
            c3.metric("Est. Travel Time", f"{travel_time:.0f} min", help=f"Based on forecasted traffic for {order_hour}:00")
            
            estimated_fee = calculate_delivery_fee(dist_km, order_hour)
            c4.metric("Estimated Delivery Fee", f"₹ {estimated_fee:.2f}")

            st.subheader("🗺️ Route Visualization")
            # Decodes the polyline to get the route coordinates.
            route_points = polyline.decode(encoded_polyline)
            map_center = np.mean(route_points, axis=0)
            
            # Creates an interactive map with start and end markers and the route.
            m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB positron")
            folium.Marker(location=start_coords, popup=f"<b>{selected_display_name}</b>", icon=folium.Icon(color="green", icon="cutlery", prefix="fa")).add_to(m)
            folium.Marker(location=end_coords, popup=f"<b>{customer_location}</b>", icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
            folium.PolyLine(locations=route_points, color='#0078FF', weight=5).add_to(m)
            
            st_folium(m, use_container_width=True, height=500)
        else:
            st.error("Could not retrieve a prediction. Please check your terminal for API error messages.")

if __name__ == "__main__":
    main()