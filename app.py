import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from retry_requests import retry
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(
    page_title="Clima Sierra de los Padres",
    page_icon="🌤️",
    layout="wide"
)

# Título y descripción de la aplicación
st.title("Panel del Clima - Sierra de los Padres")
st.markdown("Datos de pronóstico meteorológico para Sierra de los Padres, Buenos Aires, Argentina")

# Función para obtener datos meteorológicos
@st.cache_data(ttl=3600)  # Almacenar en caché los datos durante 1 hora
def obtener_datos_clima():
    # Configurar el cliente API de Open-Meteo con caché y reintento en caso de error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Coordenadas de Sierra de los Padres, Buenos Aires, Argentina
    latitude = -37.9527
    longitude = -57.7716

    # URL y parámetros para la API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "rain", "windspeed_10m", "winddirection_10m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum"],
        "timezone": "America/Argentina/Buenos_Aires",
        "forecast_days": 7,  # Pronóstico de 7 días
        "models": ["best_match", "ecmwf_ifs04", "gfs_seamless", "icon_seamless", "gem_seamless"],
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Procesar datos por hora
    hourly = response.Hourly()
    hourly_time_utc = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    # Convertir a zona horaria local
    hourly_time_local = hourly_time_utc.tz_convert("America/Argentina/Buenos_Aires")

    hourly_data = {
        "fecha_hora": hourly_time_local,
        "temperatura_c": hourly.Variables(0).ValuesAsNumpy(),
        "lluvia_mm": hourly.Variables(1).ValuesAsNumpy(),
        "viento_kmh": hourly.Variables(2).ValuesAsNumpy(),
        "direccion_viento": hourly.Variables(3).ValuesAsNumpy()
    }

    # Crear DataFrame por hora
    df_hourly = pd.DataFrame(data=hourly_data)

    # Procesar datos diarios
    daily = response.Daily()
    daily_time_utc = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    # Convertir a zona horaria local
    daily_time_local = daily_time_utc.tz_convert("America/Argentina/Buenos_Aires")

    daily_data = {
        "fecha": daily_time_local,
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "lluvia_total": daily.Variables(2).ValuesAsNumpy()
    }

    # Crear DataFrame diario
    df_daily = pd.DataFrame(data=daily_data)
    
    return df_hourly, df_daily, response

# Función para crear tablas pivote para visualización
def crear_tablas_pivote(df_hourly):
    # Crear una tabla pivote para facilitar la visualización
    df_pivot = df_hourly.copy()
    df_pivot['fecha'] = df_pivot['fecha_hora'].dt.date
    df_pivot['hora'] = df_pivot['fecha_hora'].dt.hour
    
    # Filtrar solo algunas horas para una visualización más limpia (cada 3 horas)
    df_pivot = df_pivot[df_pivot['hora'] % 3 == 0]
    
    # Crear tabla pivote para temperatura
    tabla_temp = pd.pivot_table(df_pivot, values='temperatura_c', 
                               index='hora', columns='fecha', 
                               aggfunc='mean').round(1)
    
    # Crear tabla pivote para lluvia
    tabla_lluvia = pd.pivot_table(df_pivot, values='lluvia_mm', 
                                 index='hora', columns='fecha', 
                                 aggfunc='sum').round(1)
    
    # Crear tabla pivote para viento
    tabla_viento = pd.pivot_table(df_pivot, values='viento_kmh', 
                                 index='hora', columns='fecha', 
                                 aggfunc='mean').round(1)
    
    return tabla_temp, tabla_lluvia, tabla_viento

# Función para crear mapa de calor de temperatura
def crear_mapa_temperatura(tabla_temp):
    fig = px.imshow(
        tabla_temp,
        labels=dict(x="Fecha", y="Hora", color="Temperatura (°C)"),
        x=tabla_temp.columns,
        y=tabla_temp.index,
        color_continuous_scale=[
            (0, "darkblue"),    # Frío
            (0.2, "blue"),      # Fresco
            (0.4, "lightgreen"), # Templado
            (0.6, "yellow"),    # Cálido
            (0.8, "orange"),    # Caluroso
            (1, "red")          # Muy caluroso
        ],
        aspect="auto",
        title="Pronóstico de Temperatura (°C)"
    )
    
    # Añadir anotaciones de texto
    for i in range(len(tabla_temp.index)):
        for j in range(len(tabla_temp.columns)):
            if not np.isnan(tabla_temp.iloc[i, j]):
                fig.add_annotation(
                    x=tabla_temp.columns[j],
                    y=tabla_temp.index[i],
                    text=str(tabla_temp.iloc[i, j]),
                    showarrow=False,
                    font=dict(color="black", size=10)
                )
    
    fig.update_layout(height=500)
    return fig

# Función para crear mapa de calor de lluvia
def crear_mapa_lluvia(tabla_lluvia):
    # Reemplazar ceros con NaN para mejor visualización
    tabla_lluvia_viz = tabla_lluvia.copy()
    tabla_lluvia_viz = tabla_lluvia_viz.replace(0, np.nan)
    
    fig = px.imshow(
        tabla_lluvia_viz,
        labels=dict(x="Fecha", y="Hora", color="Lluvia (mm)"),
        x=tabla_lluvia_viz.columns,
        y=tabla_lluvia_viz.index,
        color_continuous_scale=[
            (0, "lightblue"),
            (0.3, "royalblue"),
            (0.6, "blue"),
            (0.8, "darkblue"),
            (1, "navy")
        ],
        aspect="auto",
        title="Pronóstico de Lluvia (mm)"
    )
    
    # Añadir anotaciones de texto
    for i in range(len(tabla_lluvia.index)):
        for j in range(len(tabla_lluvia.columns)):
            if not np.isnan(tabla_lluvia.iloc[i, j]) and tabla_lluvia.iloc[i, j] > 0:
                fig.add_annotation(
                    x=tabla_lluvia.columns[j],
                    y=tabla_lluvia.index[i],
                    text=str(tabla_lluvia.iloc[i, j]),
                    showarrow=False,
                    font=dict(color="white" if tabla_lluvia.iloc[i, j] > 2 else "black", size=10)
                )
    
    fig.update_layout(height=500)
    return fig

# Función para crear mapa de calor de viento
def crear_mapa_viento(tabla_viento, df_hourly):
    fig = px.imshow(
        tabla_viento,
        labels=dict(x="Fecha", y="Hora", color="Velocidad del Viento (km/h)"),
        x=tabla_viento.columns,
        y=tabla_viento.index,
        color_continuous_scale=[
            (0, "lightgreen"),
            (0.3, "green"),
            (0.6, "darkgreen"),
            (0.8, "orange"),
            (1, "red")
        ],
        aspect="auto",
        title="Pronóstico de Velocidad del Viento (km/h)"
    )
    
    # Añadir anotaciones de texto
    for i in range(len(tabla_viento.index)):
        for j in range(len(tabla_viento.columns)):
            if not np.isnan(tabla_viento.iloc[i, j]):
                fig.add_annotation(
                    x=tabla_viento.columns[j],
                    y=tabla_viento.index[i],
                    text=str(tabla_viento.iloc[i, j]),
                    showarrow=False,
                    font=dict(color="white" if tabla_viento.iloc[i, j] > 30 else "black", size=10)
                )
    
    fig.update_layout(height=500)
    return fig

# Función para crear gráfico de dirección del viento
def crear_grafico_direccion_viento(df_hourly):
    # Filtrar datos cada 6 horas para evitar sobrecarga
    df_wind = df_hourly.copy()
    df_wind = df_wind[(df_wind['fecha_hora'].dt.hour % 6 == 0)]
    
    # Convertir dirección del viento a radianes y ajustar para graficar
    df_wind['direction_rad'] = np.radians(90 - df_wind['direccion_viento'])
    df_wind['u'] = -df_wind['viento_kmh'] * np.cos(df_wind['direction_rad'])
    df_wind['v'] = -df_wind['viento_kmh'] * np.sin(df_wind['direction_rad'])
    
    fig = go.Figure()
    
    # Añadir línea de velocidad del viento
    fig.add_trace(go.Scatter(
        x=df_wind['fecha_hora'],
        y=df_wind['viento_kmh'],
        mode='lines+markers',
        name='Velocidad del Viento',
        line=dict(color='green', width=2),
        marker=dict(size=8, color='darkgreen')
    ))
    
    # Añadir flechas de dirección del viento
    for i, row in df_wind.iterrows():
        fig.add_annotation(
            x=row['fecha_hora'],
            y=row['viento_kmh'],
            ax=0,
            ay=-15,
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="black",
            standoff=5,
            startstandoff=5
        )
    
    fig.update_layout(
        title="Velocidad y Dirección del Viento",
        xaxis_title="Fecha y Hora",
        yaxis_title="Velocidad del Viento (km/h)",
        height=400
    )
    
    return fig

# Función para crear resumen diario
def crear_resumen_diario(df_daily):
    fig = go.Figure()
    
    # Añadir barras de rango de temperatura
    fig.add_trace(go.Bar(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_max'] - df_daily['temp_min'],
        base=df_daily['temp_min'],
        name='Rango de Temperatura',
        marker_color='orange'
    ))
    
    # Añadir puntos de temperatura mínima
    fig.add_trace(go.Scatter(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_min'],
        mode='markers',
        name='Temp. Mínima',
        marker=dict(color='blue', size=8)
    ))
    
    # Añadir puntos de temperatura máxima
    fig.add_trace(go.Scatter(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_max'],
        mode='markers',
        name='Temp. Máxima',
        marker=dict(color='red', size=8)
    ))
    
    # Añadir anotaciones de lluvia como texto
    for i, row in df_daily.iterrows():
        if row['lluvia_total'] > 0:
            fig.add_annotation(
                x=row['fecha'].date(),
                y=row['temp_max'] + 1,
                text=f"{row['lluvia_total']:.1f} mm",
                showarrow=False,
                font=dict(color="blue", size=12)
            )
    
    fig.update_layout(
        title="Rango de Temperatura Diaria y Lluvia",
        xaxis_title="Fecha",
        yaxis_title="Temperatura (°C)",
        height=400
    )
    
    return fig

# Lógica principal de la aplicación
try:
    # Mostrar spinner mientras se obtienen los datos
    with st.spinner("Obteniendo datos meteorológicos..."):
        df_hourly, df_daily, response = obtener_datos_clima()
    
    # Mostrar información de ubicación
    st.subheader("Información de Ubicación")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latitud", f"{response.Latitude()}°S")
    with col2:
        st.metric("Longitud", f"{-response.Longitude()}°O")
    with col3:
        st.metric("Elevación", f"{response.Elevation()} m")
    
    # Crear tablas pivote
    tabla_temp, tabla_lluvia, tabla_viento = crear_tablas_pivote(df_hourly)
    
    # Crear pestañas para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["Temperatura", "Lluvia", "Viento", "Resumen Diario"])
    
    with tab1:
        st.subheader("Pronóstico de Temperatura (°C)")
        st.plotly_chart(crear_mapa_temperatura(tabla_temp), use_container_width=True)
        
        # Mostrar tabla de temperatura
        st.subheader("Tabla de Temperatura (°C)")
        st.dataframe(tabla_temp.style.background_gradient(
            cmap="RdYlBu_r", 
            axis=None
        ), use_container_width=True)
    
    with tab2:
        st.subheader("Pronóstico de Lluvia (mm)")
        st.plotly_chart(crear_mapa_lluvia(tabla_lluvia), use_container_width=True)
        
        # Mostrar tabla de lluvia
        st.subheader("Tabla de Lluvia (mm)")
        # Aplicar estilo personalizado para resaltar valores de lluvia
        def color_lluvia(val):
            color = 'white'
            if pd.notnull(val) and val > 0:
                if val <= 0.5:
                    color = 'lightblue'
                elif val <= 2:
                    color = 'skyblue'
                elif val <= 5:
                    color = 'deepskyblue'
                elif val <= 10:
                    color = 'royalblue'
                else:
                    color = 'navy'
            return f'background-color: {color}'
        
        st.dataframe(tabla_lluvia.style.applymap(color_lluvia), use_container_width=True)
        
        # Mostrar totales diarios de lluvia
        st.subheader("Totales Diarios de Lluvia (mm)")
        lluvia_diaria = pd.DataFrame({
            'Fecha': df_daily['fecha'].dt.date,
            'Lluvia Total (mm)': df_daily['lluvia_total']
        })
        st.dataframe(lluvia_diaria, use_container_width=True)
    
    with tab3:
        st.subheader("Pronóstico de Velocidad del Viento (km/h)")
        st.plotly_chart(crear_mapa_viento(tabla_viento, df_hourly), use_container_width=True)
        
        # Mostrar gráfico de dirección del viento
        st.plotly_chart(crear_grafico_direccion_viento(df_hourly), use_container_width=True)
        
        # Mostrar tabla de viento
        st.subheader("Tabla de Velocidad del Viento (km/h)")
        # Aplicar estilo personalizado para resaltar valores de viento
        def color_viento(val):
            color = 'white'
            if pd.notnull(val):
                if val <= 10:
                    color = 'palegreen'
                elif val <= 20:
                    color = 'lightgreen'
                elif val <= 30:
                    color = 'limegreen'
                elif val <= 40:
                    color = 'green'
                else:
                    color = 'darkgreen'
            return f'background-color: {color}'
        
        st.dataframe(tabla_viento.style.applymap(color_viento), use_container_width=True)
    
    with tab4:
        st.subheader("Resumen Meteorológico Diario")
        st.plotly_chart(crear_resumen_diario(df_daily), use_container_width=True)
        
        # Mostrar tabla de resumen diario
        st.subheader("Tabla de Resumen Diario")
        resumen_diario = pd.DataFrame({
            'Fecha': df_daily['fecha'].dt.date,
            'Temp. Mín. (°C)': df_daily['temp_min'],
            'Temp. Máx. (°C)': df_daily['temp_max'],
            'Rango Temp. (°C)': df_daily['temp_max'] - df_daily['temp_min'],
            'Lluvia Total (mm)': df_daily['lluvia_total']
        })
        st.dataframe(resumen_diario, use_container_width=True)
    
    # Añadir botón de actualización y hora de última actualización
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Actualizar Datos"):
            st.cache_data.clear()
            st.experimental_rerun()
    with col2:
        st.write(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Añadir pie de página con información de la fuente de datos
    st.markdown("---")
    st.markdown("Fuente de datos: [API de Open-Meteo](https://open-meteo.com/)")
    
except Exception as e:
    st.error(f"Ocurrió un error: {e}")
    st.error("Por favor, intente actualizar la página o verifique su conexión a internet.")
