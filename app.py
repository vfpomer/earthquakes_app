import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Page configuration
st.set_page_config(
    page_title="Seismic Analysis Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Interactive Seismic Activity Dashboard")
st.markdown("""
This dashboard allows you to explore seismic data for a complete month.
Use the filters and selectors in the sidebar to customize your analysis.
""")

# Global variables for safe initialization
filtered_df = None
df = None

# Function to load data
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(r"data\all_month.csv")
        
        # Convert date columns to datetime AND REMOVE TIMEZONE
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        
        # Create additional useful columns
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        
        # Categorize magnitudes
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
        df['magnitud_categoria'] = np.select(conditions, choices, default='Not classified')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Days of the week translation
days_translation = {
    'Monday': 'Mon',
    'Tuesday': 'Tue',
    'Wednesday': 'Wed',
    'Thursday': 'Thu',
    'Friday': 'Fri',
    'Saturday': 'Sat',
    'Sunday': 'Sun'
}

# Color scheme for magnitudes
magnitude_colors = {
    'Minor (<2)': 'blue',
    'Light (2-4)': 'green',
    'Moderate (4-6)': 'orange',
    'Strong (6+)': 'red'
}

# Function to ensure positive sizes for markers
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# Load data
try:
    with st.spinner('Loading data...'):
        df = load_data()
        
    if df is not None and not df.empty:
        # Sidebar for filters
        st.sidebar.header("Filters")
        
        # Date filter
        min_date = df['time'].min().date()
        max_date = df['time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Convert selected dates to datetime for filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # Convert to datetime objects without timezone
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Filter the dataframe (now both are of the same type)
            filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)].copy()
        else:
            filtered_df = df.copy()
        
        # Magnitude filter
        min_mag, max_mag = st.sidebar.slider(
            "Magnitude range",
            min_value=float(df['mag'].min()),
            max_value=float(df['mag'].max()),
            value=(float(df['mag'].min()), float(df['mag'].max())),
            step=0.1
        )
        filtered_df = filtered_df[(filtered_df['mag'] >= min_mag) & (filtered_df['mag'] <= max_mag)]
        
        # Depth filter
        min_depth, max_depth = st.sidebar.slider(
            "Depth range (km)",
            min_value=float(df['depth'].min()),
            max_value=float(df['depth'].max()),
            value=(float(df['depth'].min()), float(df['depth'].max())),
            step=5.0
        )
        filtered_df = filtered_df[(filtered_df['depth'] >= min_depth) & (filtered_df['depth'] <= max_depth)]
        
        # Event type filter
        event_types = df['type'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Event types",
            options=event_types,
            default=event_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        # Region filter (optional)
        all_regions = sorted(df['place'].str.split(', ').str[-1].unique().tolist())
        selected_regions = st.sidebar.multiselect(
            "Filter by region",
            options=all_regions,
            default=[]
        )
        if selected_regions:
            region_mask = filtered_df['place'].str.contains('|'.join(selected_regions), case=False)
            filtered_df = filtered_df[region_mask]
        
        # Show count of filtered events
        st.sidebar.metric("Selected events", len(filtered_df))
        
        # Advanced options in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Advanced Options")
        
        show_clusters = st.sidebar.checkbox("Show Cluster Analysis", value=False)
        show_advanced_charts = st.sidebar.checkbox("Show Advanced Charts", value=False)
        
        # Check if there is data after filtering
        if len(filtered_df) == 0:
            st.warning("No data available with the selected filters. Please adjust the filters.")
        else:
            # Main tabs to organize the dashboard
            main_tabs = st.tabs(["📊 General Summary", "🌐 Geographic Analysis", "⏱️ Temporal Analysis", "📈 Advanced Analysis"])
            
            # Tab 1: General Summary
            with main_tabs[0]:
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Events", len(filtered_df))
                col2.metric("Average Magnitude", f"{filtered_df['mag'].mean():.2f}")
                col3.metric("Maximum Magnitude", f"{filtered_df['mag'].max():.2f}")
                col4.metric("Average Depth", f"{filtered_df['depth'].mean():.2f} km")
                
                # Distribution of magnitudes and depths
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.subheader("Magnitude Distribution")
                    
                    fig_mag = px.histogram(
                        filtered_df,
                        x="mag",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"mag": "Magnitude", "count": "Frequency"},
                        title="Magnitude Distribution by Category"
                    )
                    fig_mag.update_layout(bargap=0.1)
                    st.plotly_chart(fig_mag, use_container_width=True)
                
                with col_dist2:
                    st.subheader("Depth Distribution")
                    
                    fig_depth = px.histogram(
                        filtered_df,
                        x="depth",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"depth": "Depth (km)", "count": "Frequency"},
                        title="Depth Distribution by Magnitude Category"
                    )
                    fig_depth.update_layout(bargap=0.1)
                    st.plotly_chart(fig_depth, use_container_width=True)
                
                # Relationship Magnitude vs Depth
                st.subheader("Relationship between Magnitude and Depth")
                
                # Ensure positive values for size
                size_values = ensure_positive(filtered_df['mag'])
                
                fig_scatter = px.scatter(
                    filtered_df,
                    x="depth",
                    y="mag",
                    color="magnitud_categoria",
                    size=size_values,  # Use guaranteed positive values
                    size_max=15,
                    opacity=0.7,
                    hover_name="place",
                    color_discrete_map=magnitude_colors,
                    labels={"depth": "Depth (km)", "mag": "Magnitude"}
                )
                
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Top 10 Regions
                st.subheader("Top 10 Regions with Highest Seismic Activity")
                
                top_places = filtered_df['place'].value_counts().head(10).reset_index()
                top_places.columns = ['Region', 'Number of Events']
                
                fig_top = px.bar(
                    top_places,
                    x='Number of Events',
                    y='Region',
                    orientation='h',
                    text='Number of Events',
                    color='Number of Events',
                    color_continuous_scale='Viridis'
                )
                
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            # Tab 2: Geographic Analysis
            with main_tabs[1]:
                geo_tabs = st.tabs(["Event Map", "Heat Map", "Cluster Analysis"])
                
                # Tab 1: Event Map
                with geo_tabs[0]:
                    st.subheader("Geographic Distribution of Earthquakes")
                    
                    # Create a basic map with px.scatter_geo instead of scatter_map
                    fig_map = px.scatter_geo(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        color="magnitud_categoria",
                        size=ensure_positive(filtered_df['mag']),  # Ensure positive values
                        size_max=15,
                        hover_name="place",
                        hover_data={
                            "latitude": False,
                            "longitude": False,
                            "magnitud_categoria": False,
                            "mag": ":.2f",
                            "depth": ":.2f km",
                            "time": True,
                            "type": True
                        },
                        color_discrete_map=magnitude_colors,
                        projection="natural earth"
                    )
                    
                    fig_map.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600,
                        geo=dict(
                            showland=True,
                            landcolor="lightgray",
                            showocean=True,
                            oceancolor="lightblue",
                            showcountries=True,
                            countrycolor="white",
                            showcoastlines=True,
                            coastlinecolor="white"
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # List of significant events
                    st.subheader("Significant Events (Magnitude ≥ 4.0)")
                    significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not significant_events.empty:
                        st.dataframe(
                            significant_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("There are no events with magnitude ≥ 4.0 in the selected range.")
                
                # Tab 2: Heat Map
                with geo_tabs[1]:
                    st.subheader("Heat Map of Seismic Activity")
                    st.markdown("""
                    This heat map shows the areas with the highest concentration of seismic activity.
                    Brighter areas indicate higher density of events.
                    """)
                    
                    # Use a heatmap approach with scatter_geo and markersize for the heat map
                    fig_heat = px.density_mapbox(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        z=ensure_positive(filtered_df['mag']),  # Ensure z is positive
                        radius=10,
                        center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()),
                        zoom=1,
                        mapbox_style="open-street-map",
                        opacity=0.8
                    )
                    
                    fig_heat.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        height=600
                    )
                    
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # Show significant events as a table instead of an additional map
                    st.subheader("Significant Events (Magnitude ≥ 4.0)")
                    strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not strong_events.empty:
                        st.dataframe(
                            strong_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("There are no events with magnitude ≥ 4.0 in the selected range.")
                
                # Tab 3: Cluster Analysis
                with geo_tabs[2]:
                    st.subheader("Geographic Cluster Analysis")
                    st.markdown("""
                    This analysis identifies groups of earthquakes that may be geographically related.
                    It uses the DBSCAN algorithm that groups events based on their spatial proximity.
                    """)
                    
                    # Prepare data for clustering
                    if len(filtered_df) > 10:  # Ensure there is enough data
                        # Select columns for clustering
                        cluster_df = filtered_df[['latitude', 'longitude']].copy()
                        
                        # Scale the data
                        scaler = StandardScaler()
                        cluster_data = scaler.fit_transform(cluster_df)
                        
                        # Slider to adjust DBSCAN parameters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            eps_distance = st.slider(
                                "Maximum distance between events to consider them neighbors (eps)",
                                min_value=0.05,
                                max_value=1.0,
                                value=0.2,
                                step=0.05
                            )
                        
                        with col2:
                            min_samples = st.slider(
                                "Minimum number of events to form a cluster",
                                min_value=2,
                                max_value=20,
                                value=5,
                                step=1
                            )
                        
                        # Run DBSCAN
                        dbscan = DBSCAN(eps=eps_distance, min_samples=min_samples)
                        cluster_result = dbscan.fit_predict(cluster_data)
                        filtered_df['cluster'] = cluster_result
                        
                        # Count the number of clusters (excluding noise, which is -1)
                        n_clusters = len(set(filtered_df['cluster'])) - (1 if -1 in filtered_df['cluster'] else 0)
                        n_noise = list(filtered_df['cluster']).count(-1)
                        
                        # Show metrics
                        col1, col2 = st.columns(2)
                        col1.metric("Number of identified clusters", n_clusters)
                        col2.metric("Ungrouped events (noise)", n_noise)
                        
                        # Visualize clusters on a map
                        st.markdown("### Cluster Map")
                        
                        # Create a column to map the cluster to string for better visualization
                        filtered_df['cluster_str'] = filtered_df['cluster'].apply(
                            lambda x: f'Cluster {x}' if x >= 0 else 'No Cluster'
                        )
                        
                        # Use scatter_geo for the cluster map
                        fig_cluster = px.scatter_geo(
                            filtered_df,
                            lat="latitude",
                            lon="longitude",
                            color="cluster_str",
                            size=ensure_positive(filtered_df['mag']),  # Ensure positive values
                            size_max=15,
                            hover_name="place",
                            hover_data={
                                "latitude": False,
                                "longitude": False,
                                "cluster_str": False,
                                "mag": ":.2f",
                                "depth": ":.2f km",
                                "time": True
                            },
                            projection="natural earth"
                        )
                        
                        fig_cluster.update_layout(
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=500,
                            geo=dict(
                                showland=True,
                                landcolor="lightgray",
                                showocean=True,
                                oceancolor="lightblue",
                                showcountries=True,
                                countrycolor="white",
                                showcoastlines=True,
                                coastlinecolor="white"
                            )
                        )
                        
                        st.plotly_chart(fig_cluster, use_container_width=True)
                        
                        # Cluster analysis
                        if n_clusters > 0:
                            st.markdown("### Cluster Analysis")
                            
                            # Cluster summary table
                            cluster_summary = filtered_df[filtered_df['cluster'] >= 0].groupby('cluster_str').agg({
                                'mag': ['count', 'mean', 'max'],
                                'depth': ['mean', 'min', 'max']
                            }).reset_index()
                            
                            # Flatten the table for better visualization
                            cluster_summary.columns = [
                                'Cluster', 'Number of Events', 'Average Magnitude', 'Maximum Magnitude',
                                'Average Depth', 'Minimum Depth', 'Maximum Depth'
                            ]
                            
                            st.dataframe(cluster_summary, use_container_width=True)
                            
                            # Select a cluster for detailed analysis
                            if n_clusters > 0:
                                cluster_options = [f'Cluster {i}' for i in range(n_clusters)]
                                if cluster_options:
                                    selected_cluster = st.selectbox(
                                        "Select a cluster to see details",
                                        options=cluster_options
                                    )
                                    
                                    # Filter data for the selected cluster
                                    cluster_data = filtered_df[filtered_df['cluster_str'] == selected_cluster]
                                    
                                    if not cluster_data.empty:
                                        # Show events in the selected cluster
                                        st.markdown(f"### Events in {selected_cluster}")
                                        st.dataframe(
                                            cluster_data[['time', 'place', 'mag', 'depth']].sort_values(by='time'),
                                            use_container_width=True
                                        )
                                        
                                        # Temporal evolution of the cluster
                                        st.markdown(f"### Temporal evolution of {selected_cluster}")
                                        
                                        fig_timeline = px.scatter(
                                            cluster_data.sort_values('time'),
                                            x='time',
                                            y='mag',
                                            size=ensure_positive(cluster_data['mag']),  # Ensure positive values
                                            color='depth',
                                            hover_name='place',
                                            title=f"Temporal evolution of events in {selected_cluster}",
                                            labels={'time': 'Date and Time', 'mag': 'Magnitude', 'depth': 'Depth (km)'}
                                        )
                                        
                                        st.plotly_chart(fig_timeline, use_container_width=True)
                                        
                                        # Interpretation suggestion
                                        st.info("""
                                        **Cluster Interpretation:**
                                        Clusters may represent aftershocks of a main earthquake, activity on a specific fault, 
                                        or patterns of seismic activity in a specific region.
                                        
                                        Observe the temporal evolution to identify whether these are simultaneous or sequential events.
                                        """)
                    else:
                        st.warning("There is not enough data to perform cluster analysis with the current filters.")
            
            # Tab 3: Temporal Analysis
            with main_tabs[2]:
                st.subheader("Temporal Pattern Analysis")
                
                # Create tabs for different temporal analyses
                temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                    "Daily Evolution", 
                    "Weekly Patterns",
                    "Hourly Patterns"
                ])
                
                # Tab 1: Daily Evolution
                with temp_tab1:
                    st.subheader("Daily Evolution of Seismic Activity")
                    
                    # Group by day
                    try:
                        daily_counts = filtered_df.groupby('day').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        daily_counts.columns = ['Date', 'Count', 'Mean Magnitude', 'Maximum Magnitude']
                        daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
                        
                        # Create chart
                        fig_daily = go.Figure()
                        
                        # Add bars for event count
                        fig_daily.add_trace(go.Bar(
                            x=daily_counts['Date'],
                            y=daily_counts['Count'],
                            name='Number of Events',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Add line for maximum magnitude
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Date'],
                            y=daily_counts['Maximum Magnitude'],
                            name='Maximum Magnitude',
                            mode='lines+markers',
                            marker=dict(color='red', size=6),
                            line=dict(width=2, dash='solid'),
                            yaxis='y2'
                        ))
                        
                        # Add line for mean magnitude
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Date'],
                            y=daily_counts['Mean Magnitude'],
                            name='Mean Magnitude',
                            mode='lines',
                            marker=dict(color='orange'),
                            line=dict(width=2, dash='dot'),
                            yaxis='y2'
                        ))
                        
                        # Configure axes and layout
                        fig_daily.update_layout(
                            title='Daily Evolution of Seismic Events',
                            xaxis=dict(title='Date', tickformat='%d-%b'),
                            yaxis=dict(title='Number of Events', side='left'),
                            yaxis2=dict(
                                title='Magnitude',
                                side='right',
                                overlaying='y',
                                range=[0, max(daily_counts['Maximum Magnitude']) + 0.5]
                            ),
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # Add trend analysis
                        if len(daily_counts) > 5:
                            st.subheader("Trend Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Calculate trend of events per day
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Count']
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                
                                trend_direction = "increasing" if z[0] > 0 else "decreasing"
                                trend_value = abs(z[0])
                                
                                st.metric(
                                    "Event Trend", 
                                    f"{trend_direction} ({trend_value:.2f} events/day)",
                                    delta=f"{trend_value:.2f}" if z[0] > 0 else f"-{trend_value:.2f}"
                                )
                            
                            with col2:
                                # Calculate trend of magnitude per day
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Mean Magnitude']
                                z_mag = np.polyfit(x, y, 1)
                                p_mag = np.poly1d(z_mag)
                                
                                trend_direction_mag = "increasing" if z_mag[0] > 0 else "decreasing"
                                trend_value_mag = abs(z_mag[0])
                                
                                st.metric(
                                    "Magnitude Trend", 
                                    f"{trend_direction_mag} ({trend_value_mag:.3f} mag/day)",
                                    delta=f"{trend_value_mag:.3f}" if z_mag[0] > 0 else f"-{trend_value_mag:.3f}"
                                )
                    except Exception as e:
                        st.error(f"Error in daily evolution analysis: {e}")
                
                # Tab 2: Weekly Patterns
                with temp_tab2:
                    try:
                        # Translate days of the week
                        filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Correctly order days of the week
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ordered_days = [days_translation[day] for day in days_order]
                        
                        # Group by day of the week
                        dow_data = filtered_df.groupby('day_name').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Rename columns
                        dow_data.columns = ['Day', 'Count', 'Mean Magnitude', 'Maximum Magnitude']
                        
                        # Order days
                        dow_data['Day_ordered'] = pd.Categorical(dow_data['Day'], categories=ordered_days, ordered=True)
                        dow_data = dow_data.sort_values('Day_ordered')
                        
                        # Create chart
                        fig_dow = px.bar(
                            dow_data,
                            x='Day',
                            y='Count',
                            color='Mean Magnitude',
                            text='Count',
                            title='Distribution of Events by Day of the Week',
                            color_continuous_scale='Viridis',
                            labels={'Count': 'Number of Events', 'Mean Magnitude': 'Average Magnitude'}
                        )
                        
                        fig_dow.update_traces(textposition='outside')
                        fig_dow.update_layout(height=400)
                        
                        st.plotly_chart(fig_dow, use_container_width=True)
                        
                        # Add an interpretation
                        st.markdown("""
                        ### Weekly pattern analysis
                        
                        This chart shows how seismic events are distributed throughout the week.
                        Significant patterns could indicate:
                        
                        - Possible influence of human activities (e.g., controlled explosions on workdays)
                        - Trends that merit additional investigation
                        - Note that weekly patterns are generally not expected in natural phenomena
                        """)
                        
                        # Create a heatmap of activity by day of the week and week of the month
                        st.subheader("Heat Map: Activity by Week and Day")
                        
                        # Add column of relative week number within the period
                        filtered_df['week_num'] = filtered_df['time'].dt.isocalendar().week
                        min_week = filtered_df['week_num'].min()
                        filtered_df['rel_week'] = filtered_df['week_num'] - min_week + 1
                        
                        # Group by relative week and day of the week
                        heatmap_weekly = filtered_df.groupby(['rel_week', 'day_name']).size().reset_index(name='count')
                        
                        # Pivot to create the format for the heatmap
                        pivot_weekly = pd.pivot_table(
                            heatmap_weekly, 
                            values='count', 
                            index='day_name', 
                            columns='rel_week',
                            fill_value=0
                        )
                        
                        # Reorder days
                        if set(ordered_days).issubset(set(pivot_weekly.index)):
                            pivot_weekly = pivot_weekly.reindex(ordered_days)
                        
                        # Create heatmap
                        fig_weekly_heat = px.imshow(
                            pivot_weekly,
                            labels=dict(x="Week", y="Day of the Week", color="Number of Events"),
                            x=[f"Week {i}" for i in pivot_weekly.columns],
                            y=pivot_weekly.index,
                            color_continuous_scale="YlOrRd",
                            title="Heat Map: Seismic Activity by Week and Day"
                        )
                        
                        st.plotly_chart(fig_weekly_heat, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in weekly pattern analysis: {e}")
                
                # Tab 3: Hourly Patterns
                with temp_tab3:
                    try:
                        st.subheader("Distribution of Events by Hour of Day")
                        
                        # Group by hour
                        hourly_counts = filtered_df.groupby('hour').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Rename columns
                        hourly_counts.columns = ['Hour', 'Count', 'Mean Magnitude', 'Maximum Magnitude']
                        
                        # Create bar chart for distribution by hour
                        fig_hourly = px.bar(
                            hourly_counts,
                            x='Hour',
                            y='Count',
                            color='Mean Magnitude',
                            title="Distribution of seismic events by hour of day",
                            labels={"Hour": "Hour of day (UTC)", "Count": "Number of events"},
                            color_continuous_scale='Viridis',
                            text='Count'
                        )
                        
                        fig_hourly.update_traces(textposition='outside')
                        fig_hourly.update_layout(height=400)
                        
                        st.plotly_chart(fig_hourly, use_container_width=True)
                        
                        # Heat map by hour and day of the week
                        st.subheader("Heat Map: Activity by Hour and Day of the Week")
                        
                        # Make sure 'day_name' exists
                        if 'day_name' not in filtered_df.columns:
                            filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Group by hour and day of the week
                        heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                        
                        # Pivot to create the format for the heatmap
                        pivot_data = pd.pivot_table(
                            heatmap_data, 
                            values='count', 
                            index='day_name', 
                            columns='hour',
                            fill_value=0
                        )
                        
                        # Reorder days
                        ordered_days = [days_translation[day] for day in days_order]
                        if set(ordered_days).issubset(set(pivot_data.index)):
                            pivot_data = pivot_data.reindex(ordered_days)
                        
                        # Create heatmap
                        fig_heatmap = px.imshow(
                            pivot_data,
                            labels=dict(x="Hour of Day (UTC)", y="Day of the Week", color="Number of Events"),
                            x=[f"{h}:00" for h in range(24)],
                            y=pivot_data.index,
                            color_continuous_scale="YlOrRd",
                            title="Heat Map: Seismic Activity by Hour and Day of the Week"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.markdown("""
                        ### Heat map interpretation
                        
                        This heat map shows the distribution of seismic events by hour and day of the week.
                        
                        - Darker cells indicate times with more seismic activity
                        - Horizontal patterns suggest hours of the day with more activity
                        - Vertical patterns indicate days of the week with more events
                        - Isolated cells of intense color may indicate special events or temporal clusters
                        """)
                    except Exception as e:
                        st.error(f"Error in hourly pattern analysis: {e}")
            
            # Tab 4: Advanced Analysis
            with main_tabs[3]:
                adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                    "Correlations", 
                    "Magnitude by Region", 
                    "Comparisons"
                ])
                
                # Tab 1: Correlations
                with adv_tab1:
                    try:
                        st.subheader("Correlation Matrix")
                        
                        # Select variables for correlation
                        corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                        
                        # Filter columns that exist in the DataFrame
                        valid_cols = [col for col in corr_cols if col in filtered_df.columns]
                        
                        if len(valid_cols) > 1:
                            corr_df = filtered_df[valid_cols].dropna()
                            
                            if len(corr_df) > 1:  # Ensure there is enough data for correlation
                                corr_matrix = corr_df.corr()
                                
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale="RdBu_r",
                                    title="Correlation Matrix",
                                    aspect="auto"
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                st.markdown("""
                                **Correlation matrix interpretation:**
                                - Values close to 1 indicate strong positive correlation
                                - Values close to -1 indicate strong negative correlation
                                - Values close to 0 indicate little or no correlation
                                """)
                                
                                # Add detailed correlation analysis
                                st.subheader("Detailed Correlation Analysis")
                                
                                # Find significant correlations
                                significant_corr = []
                                for i in range(len(valid_cols)):
                                    for j in range(i+1, len(valid_cols)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.3:  # Threshold for significant correlation
                                            significant_corr.append({
                                                'Variables': f"{valid_cols[i]} vs {valid_cols[j]}",
                                                'Correlation': corr_val,
                                                'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.5 else 'Weak',
                                                'Type': 'Positive' if corr_val > 0 else 'Negative'
                                            })
                                
                                if significant_corr:
                                    significant_df = pd.DataFrame(significant_corr)
                                    significant_df = significant_df.sort_values('Correlation', key=abs, ascending=False)
                                    
                                    st.dataframe(significant_df, use_container_width=True)
                                    
                                    # Visualize the strongest correlation
                                    if len(significant_df) > 0:
                                        top_corr = significant_df.iloc[0]
                                        var1, var2 = top_corr['Variables'].split(' vs ')
                                        
                                        st.subheader(f"Correlation Visualization: {top_corr['Variables']}")
                                        
                                        fig_scatter_corr = px.scatter(
                                            filtered_df,
                                            x=var1,
                                            y=var2,
                                            color='magnitud_categoria',
                                            size=ensure_positive(filtered_df['mag']),  # Use guaranteed positive values
                                            hover_name='place',
                                            title=f"{top_corr['Type']} {top_corr['Strength']} Correlation (r={top_corr['Correlation']:.2f})",
                                            color_discrete_map=magnitude_colors
                                        )
                                        
                                        fig_scatter_corr.update_layout(height=500)
                                        
                                        st.plotly_chart(fig_scatter_corr, use_container_width=True)
                                else:
                                    st.info("No significant correlations were found between the analyzed variables.")
                            else:
                                st.warning("There is not enough data to calculate correlations.")
                        else:
                            st.warning("There are not enough numerical columns to calculate correlations.")
                    except Exception as e:
                        st.error(f"Error in correlation analysis: {e}")
                
                # Tab 2: Magnitude by Region
                with adv_tab2:
                    try:
                        st.subheader("Magnitude Analysis by Region")
                        
                        # Extract main regions
                        filtered_df['region'] = filtered_df['place'].str.split(', ').str[-1]
                        region_stats = filtered_df.groupby('region').agg({
                            'id': 'count',
                            'mag': ['mean', 'max', 'min'],
                            'depth': 'mean'
                        }).reset_index()
                        
                        # Flatten multi-level columns
                        region_stats.columns = ['Region', 'Count', 'Mean Magnitude', 'Maximum Magnitude', 'Minimum Magnitude', 'Mean Depth']
                        
                        # Filter regions with enough events
                        min_events = st.slider("Minimum events per region", 1, 50, 5)
                        filtered_regions = region_stats[region_stats['Count'] >= min_events].sort_values('Mean Magnitude', ascending=False)
                        
                        # Visualize
                        if not filtered_regions.empty:
                            fig_regions = px.bar(
                                filtered_regions.head(15),  # Top 15 regions
                                x='Region',
                                y='Mean Magnitude',
                                error_y=filtered_regions.head(15)['Maximum Magnitude'] - filtered_regions.head(15)['Mean Magnitude'],
                                color='Count',
                                hover_data=['Count', 'Maximum Magnitude', 'Mean Depth'],
                                title='Mean Magnitude by Region (Top 15)',
                                color_continuous_scale='Viridis'
                            )
                            
                            fig_regions.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_regions, use_container_width=True)
                            
                            # Show detailed table
                            st.dataframe(
                                filtered_regions.sort_values('Count', ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.warning(f"There are no regions with at least {min_events} events. Try reducing the minimum.")
                    except Exception as e:
                        st.error(f"Error in magnitude analysis by region: {e}")
                
                # Tab 3: Comparisons
                with adv_tab3:
                    try:
                        st.subheader("Comparative Analysis")
                        
                        # Available numerical columns
                        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'rel_week', 'week_num']]
                        
                        # Select variables to compare
                        if len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_variable = st.selectbox(
                                    "X Variable",
                                    options=numeric_cols,
                                    index=numeric_cols.index('mag') if 'mag' in numeric_cols else 0
                                )
                            
                            with col2:
                                y_variable = st.selectbox(
                                    "Y Variable",
                                    options=numeric_cols,
                                    index=numeric_cols.index('depth') if 'depth' in numeric_cols else min(1, len(numeric_cols)-1)
                                )
                            
                            # Create custom scatter plot
                            fig_custom = px.scatter(
                                filtered_df,
                                x=x_variable,
                                y=y_variable,
                                color='magnitud_categoria',
                                size=ensure_positive(filtered_df['mag']),  # Positive values
                                hover_name='place',
                                title=f"Relationship between {x_variable} and {y_variable}",
                                color_discrete_map=magnitude_colors,
                                trendline='ols'  # Add trend line
                            )
                            
                            fig_custom.update_layout(height=500)
                            st.plotly_chart(fig_custom, use_container_width=True)
                            
                            # Analysis by category
                            st.subheader("Statistics by Magnitude Category")
                            
                            # Group by magnitude category
                            cat_stats = filtered_df.groupby('magnitud_categoria').agg({
                                'id': 'count',
                                'mag': ['mean', 'std'],
                                'depth': ['mean', 'std'],
                                'rms': 'mean'
                            }).reset_index()
                            
                            # Flatten columns
                            cat_stats.columns = [
                                'Category', 'Count', 'Mean Magnitude', 'Magnitude Std', 
                                'Mean Depth', 'Depth Std', 'Mean RMS'
                            ]
                            
                            # Order categories
                            cat_order = ['Minor (<2)', 'Light (2-4)', 'Moderate (4-6)', 'Strong (6+)']
                            cat_stats['Order'] = cat_stats['Category'].map({cat: i for i, cat in enumerate(cat_order)})
                            cat_stats = cat_stats.sort_values('Order').drop('Order', axis=1)
                            
                            # Visualize statistics
                            st.dataframe(cat_stats, use_container_width=True)
                            
                            # Comparative bar chart
                            fig_cats = go.Figure()
                            
                            # Add bars for count
                            fig_cats.add_trace(go.Bar(
                                x=cat_stats['Category'],
                                y=cat_stats['Count'],
                                name='Count',
                                marker_color='lightskyblue',
                                opacity=0.7
                            ))
                            
                            # Add line for mean depth
                            fig_cats.add_trace(go.Scatter(
                                x=cat_stats['Category'],
                                y=cat_stats['Mean Depth'],
                                name='Mean Depth (km)',
                                mode='lines+markers',
                                marker=dict(color='darkred', size=8),
                                line=dict(width=2),
                                yaxis='y2'
                            ))
                            
                            # Configure axes and layout
                            fig_cats.update_layout(
                                title='Comparison of Count and Depth by Category',
                                xaxis=dict(title='Magnitude Category'),
                                yaxis=dict(title='Number of Events', side='left'),
                                yaxis2=dict(
                                    title='Mean Depth (km)',
                                    side='right',
                                    overlaying='y'
                                ),
                                legend=dict(x=0.01, y=0.99),
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cats, use_container_width=True)
                        else:
                            st.warning("There are not enough numerical columns to perform comparative analysis.")
                    except Exception as e:
                        st.error(f"Error in comparative analysis: {e}")
            
            # Data table (expandable)
            with st.expander("View data in tabular format"):
                try:
                    # Available columns to display
                    display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
                    
                    # Sorting options
                    sort_col = st.selectbox(
                        "Sort by",
                        options=display_cols,
                        index=0
                    )
                    
                    sort_order = st.radio(
                        "Order",
                        options=['Descending', 'Ascending'],
                        index=0,
                        horizontal=True
                    )
                    
                    # Sort data
                    sorted_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == 'Ascending')
                    )
                    
                    # Show table
                    st.dataframe(
                        sorted_df[display_cols],
                        use_container_width=True
                    )
                    
                    # Option to download filtered data
                    csv = sorted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download filtered data (CSV)",
                        data=csv,
                        file_name="filtered_seismic_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error displaying data table: {e}")
    else:
        st.error("Could not load seismic data. Verify that the 'all_month.csv' file exists and has the correct format.")

except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.info("Verify that the 'all_month.csv' file is available and has the correct format.")

# Dashboard information
st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard**

This dashboard displays seismic data for approximately one month of activity.
Developed with Streamlit and Plotly Express.
""")