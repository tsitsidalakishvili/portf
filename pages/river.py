import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Function to fetch historical readings for a measure
def fetch_historical_data(measure_id, limit=10, since_datetime=None):
    base_url = f"http://environment.data.gov.uk/flood-monitoring/id/measures/{measure_id}/readings"
    params = {'_sorted': '', '_limit': limit}
    if since_datetime:
        params['since'] = since_datetime
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {e}")
        return None

# Function to convert API data into a DataFrame for both Water Level and Rainfall
def prepare_dataframe(data, column_name):
    if data and 'items' in data:
        readings = data['items']
        df = pd.DataFrame(readings)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.rename(columns={'value': column_name, 'dateTime': 'DateTime'}, inplace=True)
        return df[['DateTime', column_name]]
    else:
        return pd.DataFrame(columns=['DateTime', column_name])

# User Interface in Streamlit
st.title('Environmental Data Visualization')

# Inputs for Water Level
st.header("Water Level Data")
measure_id_level = st.text_input('Water Level Measure ID', '1491TH-level-stage-i-15_min-mASD', key='level_id')
limit_level = st.number_input('Number of Water Level Readings', min_value=10, value=10, key='limit_level')
since_datetime_level = st.text_input('Since Datetime for Water Level (optional)', '', key='since_datetime_level')

# Inputs for Rainfall
st.header("Rainfall Data")
measure_id_rainfall = st.text_input('Rainfall Measure ID', '52203-rainfall-tipping_bucket_raingauge-t-15_min-mm', key='rainfall_id')
limit_rainfall = st.number_input('Number of Rainfall Readings', min_value=10, value=10, key='limit_rainfall')
since_datetime_rainfall = st.text_input('Since Datetime for Rainfall (optional)', '', key='since_datetime_rainfall')

if st.button('Fetch and Plot Data'):
    # Fetch and plot water level data
    data_level = fetch_historical_data(measure_id_level, limit_level, since_datetime_level)
    df_level = prepare_dataframe(data_level, 'Water Level')
    
    # Fetch and plot rainfall data
    data_rainfall = fetch_historical_data(measure_id_rainfall, limit_rainfall, since_datetime_rainfall)
    df_rainfall = prepare_dataframe(data_rainfall, 'Rainfall')
    
    # Check if both datasets are not empty
    if not df_level.empty and not df_rainfall.empty:
        # Merge dataframes on DateTime
        df_combined = pd.merge(df_level, df_rainfall, how='outer', on='DateTime').fillna(0)
        # Plotting
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(go.Scatter(x=df_combined['DateTime'], y=df_combined['Water Level'], name='Water Level', mode='lines+markers'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_combined['DateTime'], y=df_combined['Rainfall'], name='Rainfall', mode='lines+markers'), secondary_y=True)
        
        # Add figure title
        fig.update_layout(title_text="Water Level and Rainfall Over Time")
        
        # Set x-axis title
        fig.update_xaxes(title_text="DateTime")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Water Level", secondary_y=False)
        fig.update_yaxes(title_text="Rainfall", secondary_y=True)
        
        st.plotly_chart(fig)
    else:
        st.write("No data available for plotting.")
