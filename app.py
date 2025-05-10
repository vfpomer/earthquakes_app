#--------------Libraries------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#------------------------------------------------------


#--------------Page Config-----------------------------
st.set_page_config(
    page_title='Earhquake Analysis',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)
#------------------------------------------------------



#--------------Title and Description--------------------
st.title("Earthquake Analysis Dashboard")
st.markdown("""
    This dashboard provides an analysis of earthquake data using various visualizations and clustering techniques.
    The data is sourced from the USGS Earthquake Catalog and includes information on earthquake magnitude, depth, and location.
    The dashboard allows users to explore the data through interactive charts and maps, as well as perform clustering analysis.
""")
#-------------------------------------------------------



#--------------Vars in None-----------------------------
#The objetive of this section is to define the variables taht will be used in the app
filtered_df = None
df = None
#-------------------------------------------------------



#--------------Load Data--------------------------------
#cache(@st.cache)
@st.cache_data(ttl=3600)

def load_data():
    try:
        df = pd.read_csv(r'data\all_month 1.csv')
        #Convert time column to datetime and arase and time zone
        df['time'] = pd.to_datetime(df['time'].dt.tz_localize(None))
        df['updated'] = pd.to_datetime(df['updated'].dt.tz_localize(None))
        #aditional columns
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        #magnitude categories
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Micro','Light','Moderate','Strong']
        #new column (magnitude_category)
        df['magnitude_category'] = np.select(conditions,choices, default='Unknown')
        return df
    except Exception as e:
        st.error(f'Error loading data: {e}')
        return None
#------------------------------------------------------


#-----------------Day of week transformation-----------
#Transform days of week
days_translation = {
    'Monday': 'Mon',
    'Tuesday': 'Tue',
    'Wednesday': 'Wed',
    'Thursay': 'Thu',
    'Friday': 'Fri',
    'Saturday': 'Sat',
    'Sunday': 'Sun'
}
#------------------------------------------------------


#-----------------------Color Scheme--------------------
magnitude_colors = {
    'Micro': 'blue',
    'Light': 'green',
    'Moderate': 'orange',
    'Strong': 'red',
}
#------------------------------------------------------
