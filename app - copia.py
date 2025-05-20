#------------Libraries-----------------
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#---------------------------------------

#-----------------Configuración de la página------------
st.set_page_config(
    page_title="Panel de Análisis Sísmico",
    page_icon="🌍",
    layout="wide"
)

#--------------Título y descripción--------------------------------------
st.title("🌍 Panel Interactivo de Actividad Sísmica")
st.markdown("""
Este panel te permite explorar datos sísmicos de un mes completo.
Utiliza los filtros y selectores en la barra lateral para personalizar tu análisis.
""")

#------------------ Variables globales para inicialización segura----------------
filtered_df = None
df = None

# Función para cargar datos
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("data/all_month.csv")
        
        # Convertir columnas de fecha a datetime Y REMOVER ZONA HORARIA
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None)
        
        # Crear columnas adicionales útiles
        df['day'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.day_name()
        df['week'] = df['time'].dt.isocalendar().week
        
        # Categorizar magnitudes
        conditions = [
            (df['mag'] < 2.0),
            (df['mag'] >= 2.0) & (df['mag'] < 4.0),
            (df['mag'] >= 4.0) & (df['mag'] < 6.0),
            (df['mag'] >= 6.0)
        ]
        choices = ['Menor (<2)', 'Ligero (2-4)', 'Moderado (4-6)', 'Fuerte (6+)']
        df['magnitud_categoria'] = np.select(conditions, choices, default='No clasificado')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Traducción de días de la semana
days_translation = {
    'Monday': 'Lun',
    'Tuesday': 'Mar',
    'Wednesday': 'Mié',
    'Thursday': 'Jue',
    'Friday': 'Vie',
    'Saturday': 'Sáb',
    'Sunday': 'Dom'
}

# Esquema de colores para magnitudes
magnitude_colors = {
    'Menor (<2)': 'blue',
    'Ligero (2-4)': 'green',
    'Moderado (4-6)': 'orange',
    'Fuerte (6+)': 'red'
}

# Función para asegurar tamaños positivos para los marcadores
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# Cargar datos
try:
    with st.spinner('Cargando datos...'):
        df = load_data()
        
    if df is not None and not df.empty:
        # Barra lateral para filtros
        st.sidebar.header("Filtros")
        
        # Filtro de fecha
        min_date = df['time'].min().date()
        max_date = df['time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Rango de fechas",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Convertir las fechas seleccionadas a datetime para filtrar
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # Convertir a objetos datetime sin zona horaria
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Filtrar el dataframe (ahora ambos son del mismo tipo)
            filtered_df = df[(df['time'] >= start_datetime) & (df['time'] <= end_datetime)].copy()
        else:
            filtered_df = df.copy()
        
        # Filtro de magnitud
        min_mag, max_mag = st.sidebar.slider(
            "Rango de magnitud",
            min_value=float(df['mag'].min()),
            max_value=float(df['mag'].max()),
            value=(float(df['mag'].min()), float(df['mag'].max())),
            step=0.1
        )
        filtered_df = filtered_df[(filtered_df['mag'] >= min_mag) & (filtered_df['mag'] <= max_mag)]
        
        # Filtro de profundidad
        min_depth, max_depth = st.sidebar.slider(
            "Rango de profundidad (km)",
            min_value=float(df['depth'].min()),
            max_value=float(df['depth'].max()),
            value=(float(df['depth'].min()), float(df['depth'].max())),
            step=5.0
        )
        filtered_df = filtered_df[(filtered_df['depth'] >= min_depth) & (filtered_df['depth'] <= max_depth)]
        
        # Filtro por tipo de evento
        event_types = df['type'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Tipos de evento",
            options=event_types,
            default=event_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        # Filtro por región (opcional)
        all_regions = sorted(df['place'].str.split(', ').str[-1].unique().tolist())
        selected_regions = st.sidebar.multiselect(
            "Filtrar por región",
            options=all_regions,
            default=[]
        )
        if selected_regions:
            region_mask = filtered_df['place'].str.contains('|'.join(selected_regions), case=False)
            filtered_df = filtered_df[region_mask]
        
        # Mostrar cantidad de eventos filtrados
        st.sidebar.metric("Eventos seleccionados", len(filtered_df))
        
        # Opciones avanzadas en la barra lateral
        st.sidebar.markdown("---")
        st.sidebar.header("Opciones avanzadas")
        
        show_clusters = st.sidebar.checkbox("Mostrar análisis de clústeres", value=False)
        show_advanced_charts = st.sidebar.checkbox("Mostrar gráficos avanzados", value=False)
        
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
                
                col1.metric("Eventos totales", len(filtered_df))
                col2.metric("Magnitud promedio", f"{filtered_df['mag'].mean():.2f}")
                col3.metric("Magnitud máxima", f"{filtered_df['mag'].max():.2f}")
                col4.metric("Profundidad promedio", f"{filtered_df['depth'].mean():.2f} km")
                
                # Distribución de magnitudes y profundidades
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.subheader("Distribución de magnitudes")
                    
                    fig_mag = px.histogram(
                        filtered_df,
                        x="mag",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"mag": "Magnitud", "count": "Frecuencia"},
                        title="Distribución de magnitudes por categoría"
                    )
                    fig_mag.update_layout(bargap=0.1)
                    st.plotly_chart(fig_mag, use_container_width=True)
                
                with col_dist2:
                    st.subheader("Distribución de profundidades")
                    
                    fig_depth = px.histogram(
                        filtered_df,
                        x="depth",
                        nbins=30,
                        color="magnitud_categoria",
                        color_discrete_map=magnitude_colors,
                        labels={"depth": "Profundidad (km)", "count": "Frecuencia"},
                        title="Distribución de profundidades por categoría de magnitud"
                    )
                    fig_depth.update_layout(bargap=0.1)
                    st.plotly_chart(fig_depth, use_container_width=True)
                
                # Relación Magnitud vs Profundidad
                st.subheader("Relación entre Magnitud y Profundidad")
                
                # Asegurar valores positivos para el tamaño
                size_values = ensure_positive(filtered_df['mag'])
                
                fig_scatter = px.scatter(
                    filtered_df,
                    x="depth",
                    y="mag",
                    color="magnitud_categoria",
                    size=size_values,  # Usar valores positivos garantizados
                    size_max=15,
                    opacity=0.7,
                    hover_name="place",
                    color_discrete_map=magnitude_colors,
                    labels={"depth": "Profundidad (km)", "mag": "Magnitud"}
                )
                
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Top 10 regiones
                st.subheader("Top 10 regiones con mayor actividad sísmica")
                
                top_places = filtered_df['place'].value_counts().head(10).reset_index()
                top_places.columns = ['Región', 'Número de eventos']
                
                fig_top = px.bar(
                    top_places,
                    x='Número de eventos',
                    y='Región',
                    orientation='h',
                    text='Número de eventos',
                    color='Número de eventos',
                    color_continuous_scale='Viridis'
                )
                
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            # Pestaña 2: Análisis geográfico
            with main_tabs[1]:
                geo_tabs = st.tabs(["Mapa de eventos", "Mapa de calor", "Análisis de clústeres"])
                
                # Pestaña 1: Mapa de eventos
                with geo_tabs[0]:
                    st.subheader("Distribución geográfica de los sismos")
                    
                    # Crear un mapa básico con px.scatter_geo en vez de scatter_map
                    fig_map = px.scatter_geo(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        color="magnitud_categoria",
                        size=ensure_positive(filtered_df['mag']),  # Asegurar valores positivos
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
                    
                    # Lista de eventos significativos
                    st.subheader("Eventos significativos (Magnitud ≥ 4.0)")
                    significant_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not significant_events.empty:
                        st.dataframe(
                            significant_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No hay eventos con magnitud ≥ 4.0 en el rango seleccionado.")
                
                # Pestaña 2: Mapa de calor
                with geo_tabs[1]:
                    st.subheader("Mapa de calor de la actividad sísmica")
                    st.markdown("""
                    Este mapa de calor muestra las áreas con mayor concentración de actividad sísmica.
                    Las zonas más brillantes indican mayor densidad de eventos.
                    """)
                    
                    # Usar un enfoque de heatmap con density_mapbox
                    fig_heat = px.density_mapbox(
                        filtered_df,
                        lat="latitude",
                        lon="longitude",
                        z=ensure_positive(filtered_df['mag']),  # Asegurar que z sea positivo
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
                    
                    # Mostrar eventos significativos como tabla en vez de un mapa adicional
                    st.subheader("Eventos significativos (Magnitud ≥ 4.0)")
                    strong_events = filtered_df[filtered_df['mag'] >= 4.0].sort_values(by='mag', ascending=False)
                    
                    if not strong_events.empty:
                        st.dataframe(
                            strong_events[['time', 'place', 'mag', 'depth', 'type']],
                            use_container_width=True
                        )
                    else:
                        st.info("No hay eventos con magnitud ≥ 4.0 en el rango seleccionado.")
                
                # Pestaña 3: Análisis de clústeres
                with geo_tabs[2]:
                    st.subheader("Análisis geográfico de clústeres")
                    st.markdown("""
                    Este análisis identifica grupos de sismos que pueden estar relacionados geográficamente.
                    Utiliza el algoritmo DBSCAN, que agrupa eventos según su proximidad espacial.
                    """)
                    
                    # Preparar datos para clustering
                    if len(filtered_df) > 10:  # Asegurar que haya suficientes datos
                        # Seleccionar columnas para clustering
                        cluster_df = filtered_df[['latitude', 'longitude']].copy()
                        
                        # Escalar los datos
                        scaler = StandardScaler()
                        cluster_data = scaler.fit_transform(cluster_df)
                        
                        # Sliders para ajustar parámetros de DBSCAN
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            eps_distance = st.slider(
                                "Distancia máxima entre eventos para considerarlos vecinos (eps)",
                                min_value=0.05,
                                max_value=1.0,
                                value=0.2,
                                step=0.05
                            )
                        
                        with col2:
                            min_samples = st.slider(
                                "Número mínimo de eventos para formar un clúster",
                                min_value=2,
                                max_value=20,
                                value=5,
                                step=1
                            )
                        
                        # Ejecutar DBSCAN
                        dbscan = DBSCAN(eps=eps_distance, min_samples=min_samples)
                        cluster_result = dbscan.fit_predict(cluster_data)
                        filtered_df['cluster'] = cluster_result
                        
                        # Contar el número de clústeres (excluyendo ruido, que es -1)
                        n_clusters = len(set(filtered_df['cluster'])) - (1 if -1 in filtered_df['cluster'] else 0)
                        n_noise = list(filtered_df['cluster']).count(-1)
                        
                        # Mostrar métricas
                        col1, col2 = st.columns(2)
                        col1.metric("Número de clústeres identificados", n_clusters)
                        col2.metric("Eventos no agrupados (ruido)", n_noise)
                        
                        # Visualizar clústeres en un mapa
                        st.markdown("### Mapa de Clústeres")
                        
                        # Crear una columna para mapear el clúster a string para mejor visualización
                        filtered_df['cluster_str'] = filtered_df['cluster'].apply(
                            lambda x: f'Clúster {x}' if x >= 0 else 'Sin Clúster'
                        )
                        
                        # Usar scatter_geo para el mapa de clústeres
                        fig_cluster = px.scatter_geo(
                            filtered_df,
                            lat="latitude",
                            lon="longitude",
                            color="cluster_str",
                            size=ensure_positive(filtered_df['mag']),  # Asegurar valores positivos
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
                        # Análisis de clústeres
                        if n_clusters > 0:
                            st.markdown("### Análisis de Clústeres")
                            
                            # Tabla resumen de clústeres
                            cluster_summary = filtered_df[filtered_df['cluster'] >= 0].groupby('cluster_str').agg({
                                'mag': ['count', 'mean', 'max'],
                                'depth': ['mean', 'min', 'max']
                            }).reset_index()
                            
                            # Aplanar la tabla para mejor visualización
                            cluster_summary.columns = [
                                'Clúster', 'Número de eventos', 'Magnitud promedio', 'Magnitud máxima',
                                'Profundidad promedio', 'Profundidad mínima', 'Profundidad máxima'
                            ]
                            
                            st.dataframe(cluster_summary, use_container_width=True)
                            
                            # Seleccionar un clúster para análisis detallado
                            if n_clusters > 0:
                                cluster_options = [f'Clúster {i}' for i in range(n_clusters)]
                                if cluster_options:
                                    selected_cluster = st.selectbox(
                                        "Selecciona un clúster para ver detalles",
                                        options=cluster_options
                                    )
                                    
                                    # Filtrar datos para el clúster seleccionado
                                    cluster_data = filtered_df[filtered_df['cluster_str'] == selected_cluster]
                                    
                                    if not cluster_data.empty:
                                        # Mostrar eventos del clúster seleccionado
                                        st.markdown(f"### Eventos en {selected_cluster}")
                                        st.dataframe(
                                            cluster_data[['time', 'place', 'mag', 'depth']].sort_values(by='time'),
                                            use_container_width=True
                                        )
                                        
                                        # Evolución temporal del clúster
                                        st.markdown(f"### Evolución temporal de {selected_cluster}")
                                        
                                        fig_timeline = px.scatter(
                                            cluster_data.sort_values('time'),
                                            x='time',
                                            y='mag',
                                            size=ensure_positive(cluster_data['mag']),  # Asegurar valores positivos
                                            color='depth',
                                            hover_name='place',
                                            title=f"Evolución temporal de los eventos en {selected_cluster}",
                                            labels={'time': 'Fecha y hora', 'mag': 'Magnitud', 'depth': 'Profundidad (km)'}
                                        )
                                        
                                        st.plotly_chart(fig_timeline, use_container_width=True)
                                        
                                        # Sugerencia de interpretación
                                        st.info("""
                                        **Interpretación del clúster:**
                                        Los clústeres pueden representar réplicas de un sismo principal, actividad en una falla específica,
                                        o patrones de actividad sísmica en una región concreta.
                                        
                                        Observa la evolución temporal para identificar si se trata de eventos simultáneos o secuenciales.
                                        """)
                        else:
                            st.warning("No hay suficientes datos para realizar análisis de clústeres con los filtros actuales.")
            # Pestaña 3: Análisis Temporal
            with main_tabs[2]:
                st.subheader("Análisis de Patrones Temporales")
                
                # Crear pestañas para diferentes análisis temporales
                temp_tab1, temp_tab2, temp_tab3 = st.tabs([
                    "Evolución Diaria", 
                    "Patrones Semanales",
                    "Patrones por Hora"
                ])
                
                # Pestaña 1: Evolución Diaria
                with temp_tab1:
                    st.subheader("Evolución Diaria de la Actividad Sísmica")
                    
                    # Agrupar por día
                    try:
                        daily_counts = filtered_df.groupby('day').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        daily_counts.columns = ['Fecha', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        daily_counts['Fecha'] = pd.to_datetime(daily_counts['Fecha'])
                        
                        # Crear gráfico
                        fig_daily = go.Figure()
                        
                        # Añadir barras para el conteo de eventos
                        fig_daily.add_trace(go.Bar(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Cantidad'],
                            name='Número de Eventos',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # Añadir línea para la magnitud máxima
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Magnitud Máxima'],
                            name='Magnitud Máxima',
                            mode='lines+markers',
                            marker=dict(color='red', size=6),
                            line=dict(width=2, dash='solid'),
                            yaxis='y2'
                        ))
                        
                        # Añadir línea para la magnitud media
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Magnitud Media'],
                            name='Magnitud Media',
                            mode='lines',
                            marker=dict(color='orange'),
                            line=dict(width=2, dash='dot'),
                            yaxis='y2'
                        ))
                        
                        # Configurar ejes y diseño
                        fig_daily.update_layout(
                            title='Evolución Diaria de los Eventos Sísmicos',
                            xaxis=dict(title='Fecha', tickformat='%d-%b'),
                            yaxis=dict(title='Número de Eventos', side='left'),
                            yaxis2=dict(
                                title='Magnitud',
                                side='right',
                                overlaying='y',
                                range=[0, max(daily_counts['Magnitud Máxima']) + 0.5]
                            ),
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # Añadir análisis de tendencia
                        if len(daily_counts) > 5:
                            st.subheader("Análisis de Tendencia")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Calcular tendencia de eventos por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Cantidad']
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                
                                trend_direction = "en aumento" if z[0] > 0 else "en descenso"
                                trend_value = abs(z[0])
                                
                                st.metric(
                                    "Tendencia de Eventos", 
                                    f"{trend_direction} ({trend_value:.2f} eventos/día)",
                                    delta=f"{trend_value:.2f}" if z[0] > 0 else f"-{trend_value:.2f}"
                                )
                            
                            with col2:
                                # Calcular tendencia de magnitud por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Magnitud Media']
                                z_mag = np.polyfit(x, y, 1)
                                p_mag = np.poly1d(z_mag)
                                
                                trend_direction_mag = "en aumento" if z_mag[0] > 0 else "en descenso"
                                trend_value_mag = abs(z_mag[0])
                                
                                st.metric(
                                    "Tendencia de Magnitud", 
                                    f"{trend_direction_mag} ({trend_value_mag:.3f} mag/día)",
                                    delta=f"{trend_value_mag:.3f}" if z_mag[0] > 0 else f"-{trend_value_mag:.3f}"
                                )
                    except Exception as e:
                        st.error(f"Error en el análisis de evolución diaria: {e}")
                        # Añadir línea para la magnitud media
                        fig_daily.add_trace(go.Scatter(
                            x=daily_counts['Fecha'],
                            y=daily_counts['Magnitud Media'],
                            name='Magnitud Media',
                            mode='lines',
                            marker=dict(color='orange'),
                            line=dict(width=2, dash='dot'),
                            yaxis='y2'
                        ))
                        
                        # Configurar ejes y diseño
                        fig_daily.update_layout(
                            title='Evolución Diaria de los Eventos Sísmicos',
                            xaxis=dict(title='Fecha', tickformat='%d-%b'),
                            yaxis=dict(title='Número de Eventos', side='left'),
                            yaxis2=dict(
                                title='Magnitud',
                                side='right',
                                overlaying='y',
                                range=[0, max(daily_counts['Magnitud Máxima']) + 0.5]
                            ),
                            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig_daily, use_container_width=True)
                        
                        # Añadir análisis de tendencia
                        if len(daily_counts) > 5:
                            st.subheader("Análisis de Tendencia")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Calcular tendencia de eventos por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Cantidad']
                                z = np.polyfit(x, y, 1)
                                p = np.poly1d(z)
                                
                                trend_direction = "en aumento" if z[0] > 0 else "en descenso"
                                trend_value = abs(z[0])
                                
                                st.metric(
                                    "Tendencia de Eventos", 
                                    f"{trend_direction} ({trend_value:.2f} eventos/día)",
                                    delta=f"{trend_value:.2f}" if z[0] > 0 else f"-{trend_value:.2f}"
                                )
                            
                            with col2:
                                # Calcular tendencia de magnitud por día
                                x = np.arange(len(daily_counts))
                                y = daily_counts['Magnitud Media']
                                z_mag = np.polyfit(x, y, 1)
                                p_mag = np.poly1d(z_mag)
                                
                                trend_direction_mag = "en aumento" if z_mag[0] > 0 else "en descenso"
                                trend_value_mag = abs(z_mag[0])
                                
                                st.metric(
                                    "Tendencia de Magnitud", 
                                    f"{trend_direction_mag} ({trend_value_mag:.3f} mag/día)",
                                    delta=f"{trend_value_mag:.3f}" if z_mag[0] > 0 else f"-{trend_value_mag:.3f}"
                                )
                    except Exception as e:
                        st.error(f"Error en el análisis de evolución diaria: {e}")
                
                # Pestaña 2: Patrones Semanales
                with temp_tab2:
                    try:
                        # Traducir días de la semana
                        filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Orden correcto de los días de la semana
                        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        ordered_days = [days_translation[day] for day in days_order]
                        
                        # Agrupar por día de la semana
                        dow_data = filtered_df.groupby('day_name').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Renombrar columnas
                        dow_data.columns = ['Día', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        
                        # Ordenar días
                        dow_data['Día_ordenado'] = pd.Categorical(dow_data['Día'], categories=ordered_days, ordered=True)
                        dow_data = dow_data.sort_values('Día_ordenado')
                        
                        # Crear gráfico
                        fig_dow = px.bar(
                            dow_data,
                            x='Día',
                            y='Cantidad',
                            color='Magnitud Media',
                            text='Cantidad',
                            title='Distribución de eventos por día de la semana',
                            color_continuous_scale='Viridis',
                            labels={'Cantidad': 'Número de eventos', 'Magnitud Media': 'Magnitud promedio'}
                        )
                        
                        fig_dow.update_traces(textposition='outside')
                        fig_dow.update_layout(height=400)
                        
                        st.plotly_chart(fig_dow, use_container_width=True)
                        
                        # Añadir interpretación
                        st.markdown("""
                        ### Análisis de patrones semanales
                        
                        Este gráfico muestra cómo se distribuyen los eventos sísmicos a lo largo de la semana.
                        Patrones significativos podrían indicar:
                        
                        - Posible influencia de actividades humanas (por ejemplo, explosiones controladas en días laborables)
                        - Tendencias que merecen investigación adicional
                        - Nota: en fenómenos naturales no suelen esperarse patrones semanales
                        """)
                        # Crear un mapa de calor de la actividad por día de la semana y semana del mes
                        st.subheader("Mapa de calor: Actividad por semana y día")
                        
                        # Añadir columna de número de semana relativa dentro del periodo
                        filtered_df['week_num'] = filtered_df['time'].dt.isocalendar().week
                        min_week = filtered_df['week_num'].min()
                        filtered_df['rel_week'] = filtered_df['week_num'] - min_week + 1
                        
                        # Agrupar por semana relativa y día de la semana
                        heatmap_weekly = filtered_df.groupby(['rel_week', 'day_name']).size().reset_index(name='count')
                        
                        # Pivotear para crear el formato del mapa de calor
                        pivot_weekly = pd.pivot_table(
                            heatmap_weekly, 
                            values='count', 
                            index='day_name', 
                            columns='rel_week',
                            fill_value=0
                        )
                        
                        # Reordenar días
                        if set(ordered_days).issubset(set(pivot_weekly.index)):
                            pivot_weekly = pivot_weekly.reindex(ordered_days)
                        
                        # Crear mapa de calor
                        fig_weekly_heat = px.imshow(
                            pivot_weekly,
                            labels=dict(x="Semana", y="Día de la semana", color="Número de eventos"),
                            x=[f"Semana {i}" for i in pivot_weekly.columns],
                            y=pivot_weekly.index,
                            color_continuous_scale="YlOrRd",
                            title="Mapa de calor: Actividad sísmica por semana y día"
                        )
                        
                        st.plotly_chart(fig_weekly_heat, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error en el análisis de patrones semanales: {e}")
                
                # Pestaña 3: Patrones por hora
                with temp_tab3:
                    try:
                        st.subheader("Distribución de eventos por hora del día")
                        
                        # Agrupar por hora
                        hourly_counts = filtered_df.groupby('hour').agg({
                            'id': 'count',
                            'mag': ['mean', 'max']
                        }).reset_index()
                        
                        # Renombrar columnas
                        hourly_counts.columns = ['Hora', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima']
                        
                        # Crear gráfico de barras para la distribución por hora
                        fig_hourly = px.bar(
                            hourly_counts,
                            x='Hora',
                            y='Cantidad',
                            color='Magnitud Media',
                            title="Distribución de eventos sísmicos por hora del día",
                            labels={"Hora": "Hora del día (UTC)", "Cantidad": "Número de eventos"},
                            color_continuous_scale='Viridis',
                            text='Cantidad'
                        )
                        
                        fig_hourly.update_traces(textposition='outside')
                        fig_hourly.update_layout(height=400)
                        
                        st.plotly_chart(fig_hourly, use_container_width=True)
                        
                        # Mapa de calor por hora y día de la semana
                        st.subheader("Mapa de calor: Actividad por hora y día de la semana")
                        
                        # Asegurarse de que 'day_name' existe
                        if 'day_name' not in filtered_df.columns:
                            filtered_df['day_name'] = filtered_df['day_of_week'].map(days_translation)
                        
                        # Agrupar por hora y día de la semana
                        heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                        
                        # Pivotear para crear el formato del mapa de calor
                        pivot_data = pd.pivot_table(
                            heatmap_data, 
                            values='count', 
                            index='day_name', 
                            columns='hour',
                            fill_value=0
                        )
                        
                        # Reordenar días
                        ordered_days = [days_translation[day] for day in days_order]
                        if set(ordered_days).issubset(set(pivot_data.index)):
                            pivot_data = pivot_data.reindex(ordered_days)
                        
                        # Crear mapa de calor
                        fig_heatmap = px.imshow(
                            pivot_data,
                            labels=dict(x="Hora del día (UTC)", y="Día de la semana", color="Número de eventos"),
                            x=[f"{h}:00" for h in range(24)],
                            y=pivot_data.index,
                            color_continuous_scale="YlOrRd",
                            title="Mapa de calor: Actividad sísmica por hora y día de la semana"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.markdown("""
                        ### Interpretación del mapa de calor
                        
                        Este mapa de calor muestra la distribución de los eventos sísmicos por hora y día de la semana.
                        
                        - Las celdas más oscuras indican horas con mayor actividad sísmica
                        - Patrones horizontales sugieren horas del día con más actividad
                        - Patrones verticales indican días de la semana con más eventos
                        - Celdas aisladas de color intenso pueden indicar eventos especiales o clústeres temporales
                        """)
                    except Exception as e:
                        st.error(f"Error en el análisis de patrones horarios: {e}")
            
            # Pestaña 4: Análisis Avanzado
            with main_tabs[3]:
                adv_tab1, adv_tab2, adv_tab3 = st.tabs([
                    "Correlaciones", 
                    "Magnitud por Región", 
                    "Comparaciones"
                ])
                
                # Pestaña 1: Correlaciones
                with adv_tab1:
                    try:
                        st.subheader("Matriz de Correlación")
                        
                        # Seleccionar variables para la correlación
                        corr_cols = ['mag', 'depth', 'rms', 'gap', 'horizontalError', 'depthError']
                        
                        # Filtrar columnas que existen en el DataFrame
                        valid_cols = [col for col in corr_cols if col in filtered_df.columns]
                        
                        if len(valid_cols) > 1:
                            corr_df = filtered_df[valid_cols].dropna()
                            
                            if len(corr_df) > 1:  # Asegurar que haya suficientes datos para la correlación
                                corr_matrix = corr_df.corr()
                                
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    color_continuous_scale="RdBu_r",
                                    title="Matriz de Correlación",
                                    aspect="auto"
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                                
                                st.markdown("""
                                **Interpretación de la matriz de correlación:**
                                - Valores cercanos a 1 indican fuerte correlación positiva
                                - Valores cercanos a -1 indican fuerte correlación negativa
                                - Valores cercanos a 0 indican poca o ninguna correlación
                                """)
                                
                                # Análisis detallado de correlaciones
                                st.subheader("Análisis Detallado de Correlaciones")
                                
                                # Buscar correlaciones significativas
                                significant_corr = []
                                for i in range(len(valid_cols)):
                                    for j in range(i+1, len(valid_cols)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.3:  # Umbral para correlación significativa
                                            significant_corr.append({
                                                'Variables': f"{valid_cols[i]} vs {valid_cols[j]}",
                                                'Correlación': corr_val,
                                                'Fuerza': 'Fuerte' if abs(corr_val) > 0.7 else 'Moderada' if abs(corr_val) > 0.5 else 'Débil',
                                                'Tipo': 'Positiva' if corr_val > 0 else 'Negativa'
                                            })
                                
                                if significant_corr:
                                    significant_df = pd.DataFrame(significant_corr)
                                    significant_df = significant_df.sort_values('Correlación', key=abs, ascending=False)
                                    
                                    st.dataframe(significant_df, use_container_width=True)
                                    
                                    # Visualizar la correlación más fuerte
                                    if len(significant_df) > 0:
                                        top_corr = significant_df.iloc[0]
                                        var1, var2 = top_corr['Variables'].split(' vs ')
                                        
                                        st.subheader(f"Visualización de Correlación: {top_corr['Variables']}")
                                        
                                        fig_scatter_corr = px.scatter(
                                            filtered_df,
                                            x=var1,
                                            y=var2,
                                            color='magnitud_categoria',
                                            size=ensure_positive(filtered_df['mag']),  # Usar valores positivos garantizados
                                            hover_name='place',
                                            title=f"{top_corr['Tipo']} {top_corr['Fuerza']} Correlación (r={top_corr['Correlación']:.2f})",
                                            color_discrete_map=magnitude_colors
                                        )
                                        
                                        fig_scatter_corr.update_layout(height=500)
                                        
                                        st.plotly_chart(fig_scatter_corr, use_container_width=True)
                                else:
                                    st.info("No se encontraron correlaciones significativas entre las variables analizadas.")
                            else:
                                st.warning("No hay suficientes datos para calcular correlaciones.")
                        else:
                            st.warning("No hay suficientes columnas numéricas para calcular correlaciones.")
                    except Exception as e:
                        st.error(f"Error en el análisis de correlaciones: {e}")
                # Pestaña 2: Magnitud por Región
                with adv_tab2:
                    try:
                        st.subheader("Análisis de Magnitud por Región")
                        
                        # Extraer regiones principales
                        filtered_df['region'] = filtered_df['place'].str.split(', ').str[-1]
                        region_stats = filtered_df.groupby('region').agg({
                            'id': 'count',
                            'mag': ['mean', 'max', 'min'],
                            'depth': 'mean'
                        }).reset_index()
                        
                        # Aplanar columnas multinivel
                        region_stats.columns = ['Región', 'Cantidad', 'Magnitud Media', 'Magnitud Máxima', 'Magnitud Mínima', 'Profundidad Media']
                        
                        # Filtrar regiones con suficientes eventos
                        min_events = st.slider("Mínimo de eventos por región", 1, 50, 5)
                        filtered_regions = region_stats[region_stats['Cantidad'] >= min_events].sort_values('Magnitud Media', ascending=False)
                        
                        # Visualizar
                        if not filtered_regions.empty:
                            fig_regions = px.bar(
                                filtered_regions.head(15),  # Top 15 regiones
                                x='Región',
                                y='Magnitud Media',
                                error_y=filtered_regions.head(15)['Magnitud Máxima'] - filtered_regions.head(15)['Magnitud Media'],
                                color='Cantidad',
                                hover_data=['Cantidad', 'Magnitud Máxima', 'Profundidad Media'],
                                title='Magnitud Media por Región (Top 15)',
                                color_continuous_scale='Viridis'
                            )
                            
                            fig_regions.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_regions, use_container_width=True)
                            
                            # Mostrar tabla detallada
                            st.dataframe(
                                filtered_regions.sort_values('Cantidad', ascending=False),
                                use_container_width=True
                            )
                        else:
                            st.warning(f"No hay regiones con al menos {min_events} eventos. Prueba reduciendo el mínimo.")
                    except Exception as e:
                        st.error(f"Error en el análisis de magnitud por región: {e}")
                
                # Pestaña 3: Comparaciones
                with adv_tab3:
                    try:
                        st.subheader("Análisis Comparativo")
                        
                        # Columnas numéricas disponibles
                        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'rel_week', 'week_num']]
                        
                        # Seleccionar variables a comparar
                        if len(numeric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                x_variable = st.selectbox(
                                    "Variable X",
                                    options=numeric_cols,
                                    index=numeric_cols.index('mag') if 'mag' in numeric_cols else 0
                                )
                            
                            with col2:
                                y_variable = st.selectbox(
                                    "Variable Y",
                                    options=numeric_cols,
                                    index=numeric_cols.index('depth') if 'depth' in numeric_cols else min(1, len(numeric_cols)-1)
                                )
                            
                            # Crear gráfico de dispersión personalizado
                            fig_custom = px.scatter(
                                filtered_df,
                                x=x_variable,
                                y=y_variable,
                                color='magnitud_categoria',
                                size=ensure_positive(filtered_df['mag']),  # Valores positivos
                                hover_name='place',
                                title=f"Relación entre {x_variable} y {y_variable}",
                                color_discrete_map=magnitude_colors,
                                trendline='ols'  # Línea de tendencia
                            )
                            
                            fig_custom.update_layout(height=500)
                            st.plotly_chart(fig_custom, use_container_width=True)
                            
                            # Análisis por categoría
                            st.subheader("Estadísticas por Categoría de Magnitud")
                            
                            # Agrupar por categoría de magnitud
                            cat_stats = filtered_df.groupby('magnitud_categoria').agg({
                                'id': 'count',
                                'mag': ['mean', 'std'],
                                'depth': ['mean', 'std'],
                                'rms': 'mean'
                            }).reset_index()
                            
                            # Aplanar columnas
                            cat_stats.columns = [
                                'Categoría', 'Cantidad', 'Magnitud Media', 'Desv. Est. Magnitud', 
                                'Profundidad Media', 'Desv. Est. Profundidad', 'RMS Medio'
                            ]
                            
                            # Ordenar categorías
                            cat_order = ['Menor (<2)', 'Ligero (2-4)', 'Moderado (4-6)', 'Fuerte (6+)']
                            cat_stats['Orden'] = cat_stats['Categoría'].map({cat: i for i, cat in enumerate(cat_order)})
                            cat_stats = cat_stats.sort_values('Orden').drop('Orden', axis=1)
                            
                            # Visualizar estadísticas
                            st.dataframe(cat_stats, use_container_width=True)
                            
                            # Gráfico comparativo de barras
                            fig_cats = go.Figure()
                            
                            # Añadir barras para cantidad
                            fig_cats.add_trace(go.Bar(
                                x=cat_stats['Categoría'],
                                y=cat_stats['Cantidad'],
                                name='Cantidad',
                                marker_color='lightskyblue',
                                opacity=0.7
                            ))
                            
                            # Añadir línea para profundidad media
                            fig_cats.add_trace(go.Scatter(
                                x=cat_stats['Categoría'],
                                y=cat_stats['Profundidad Media'],
                                name='Profundidad Media (km)',
                                mode='lines+markers',
                                marker=dict(color='darkred', size=8),
                                line=dict(width=2),
                                yaxis='y2'
                            ))
                            
                            # Configurar ejes y diseño
                            fig_cats.update_layout(
                                title='Comparación de Cantidad y Profundidad por Categoría',
                                xaxis=dict(title='Categoría de Magnitud'),
                                yaxis=dict(title='Número de Eventos', side='left'),
                                yaxis2=dict(
                                    title='Profundidad Media (km)',
                                    side='right',
                                    overlaying='y'
                                ),
                                legend=dict(x=0.01, y=0.99),
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig_cats, use_container_width=True)
                        else:
                            st.warning("No hay suficientes columnas numéricas para realizar el análisis comparativo.")
                    except Exception as e:
                        st.error(f"Error en el análisis comparativo: {e}")
            
            # Tabla de datos (expandible)
            with st.expander("Ver datos en formato tabla"):
                try:
                    # Columnas disponibles para mostrar
                    display_cols = [col for col in ['time', 'place', 'mag', 'depth', 'type', 'magType', 'rms'] if col in filtered_df.columns]
                    
                    # Opciones de ordenamiento
                    sort_col = st.selectbox(
                        "Ordenar por",
                        options=display_cols,
                        index=0
                    )
                    
                    sort_order = st.radio(
                        "Orden",
                        options=['Descendente', 'Ascendente'],
                        index=0,
                        horizontal=True
                    )
                    
                    # Ordenar datos
                    sorted_df = filtered_df.sort_values(
                        by=sort_col,
                        ascending=(sort_order == 'Ascendente')
                    )
                    
                    # Mostrar tabla
                    st.dataframe(
                        sorted_df[display_cols],
                        use_container_width=True
                    )
                    
                    # Opción para descargar datos filtrados
                    csv = sorted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Descargar datos filtrados (CSV)",
                        data=csv,
                        file_name="datos_sismicos_filtrados.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error al mostrar la tabla de datos: {e}")
    else:
        st.error("No se pudieron cargar los datos sísmicos. Verifica que el archivo 'all_month.csv' exista y tenga el formato correcto.")

except Exception as e:
    st.error(f"Error al cargar o procesar los datos: {e}")
    st.info("Verifica que el archivo 'all_month.csv' esté disponible y tenga el formato correcto.")

# Información del dashboard
st.sidebar.markdown("---")
st.sidebar.info("""
**Acerca de este Panel**

Este panel muestra datos sísmicos de aproximadamente un mes de actividad.
Desarrollado con Streamlit y Plotly Express.
""")