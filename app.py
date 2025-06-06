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

# Diccionario con información de los modelos disponibles
MODELOS_DISPONIBLES = {
    "best_match": {
        "nombre": "Mejor Coincidencia (Automático)",
        "descripcion": "Selección automática del mejor modelo para la ubicación",
        "resolucion": "Variable",
        "origen": "Open-Meteo"
    },
    "ecmwf_ifs04": {
        "nombre": "ECMWF IFS",
        "descripcion": "Centro Europeo de Pronósticos - Alta precisión",
        "resolucion": "0.4° (~44km)",
        "origen": "Europa"
    },
    "gfs_seamless": {
        "nombre": "GFS NOAA",
        "descripcion": "Sistema de Pronóstico Global de NOAA",
        "resolucion": "0.25° (~28km)",
        "origen": "Estados Unidos"
    },
    "icon_seamless": {
        "nombre": "ICON DWD",
        "descripcion": "Modelo Icosaédrico del Servicio Meteorológico Alemán",
        "resolucion": "0.125° (~13km)",
        "origen": "Alemania"
    },
    "gem_seamless": {
        "nombre": "GEM CMC",
        "descripcion": "Modelo de Ambiente Global de Canadá",
        "resolucion": "0.24° (~25km)",
        "origen": "Canadá"
    },
    "meteofrance_seamless": {
        "nombre": "ARPEGE Météo-France",
        "descripcion": "Modelo de Alta Resolución de Francia",
        "resolucion": "0.1° (~10km)",
        "origen": "Francia"
    },
    "jma_seamless": {
        "nombre": "JMA GSM",
        "descripcion": "Modelo Global de la Agencia Meteorológica de Japón",
        "resolucion": "0.5° (~55km)",
        "origen": "Japón"
    },
    "metno_seamless": {
        "nombre": "NORDIC NWP",
        "descripcion": "Modelo Nórdico de Predicción Numérica",
        "resolucion": "1.0° (~111km)",
        "origen": "Noruega"
    }
}

# Sidebar para selección de opciones
st.sidebar.header("⚙️ Configuración")

# Selector de modelo meteorológico
modelo_seleccionado = st.sidebar.selectbox(
    "🛰️ Seleccionar Modelo Meteorológico:",
    options=list(MODELOS_DISPONIBLES.keys()),
    format_func=lambda x: MODELOS_DISPONIBLES[x]["nombre"],
    index=0,
    help="Selecciona el modelo meteorológico para el pronóstico"
)

# Mostrar información del modelo seleccionado
st.sidebar.info(f"""
**📊 Información del Modelo:**
- **Nombre:** {MODELOS_DISPONIBLES[modelo_seleccionado]['nombre']}
- **Descripción:** {MODELOS_DISPONIBLES[modelo_seleccionado]['descripcion']}
- **Resolución:** {MODELOS_DISPONIBLES[modelo_seleccionado]['resolucion']}
- **Origen:** {MODELOS_DISPONIBLES[modelo_seleccionado]['origen']}
""")

# Opción para comparar modelos
comparar_modelos = st.sidebar.checkbox(
    "🔄 Comparar con otros modelos",
    help="Muestra comparación con modelos adicionales"
)

modelos_comparacion = []
if comparar_modelos:
    modelos_comparacion = st.sidebar.multiselect(
        "Modelos para comparar:",
        options=[k for k in MODELOS_DISPONIBLES.keys() if k != modelo_seleccionado],
        default=["ecmwf_ifs04", "gfs_seamless"] if modelo_seleccionado != "ecmwf_ifs04" else ["gfs_seamless", "icon_seamless"],
        format_func=lambda x: MODELOS_DISPONIBLES[x]["nombre"]
    )

# Función para obtener datos meteorológicos con modelo específico
@st.cache_data(ttl=3600)  # Almacenar en caché los datos durante 1 hora
def obtener_datos_clima(modelo="best_match"):
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
        "hourly": ["temperature_2m", "rain", "windspeed_10m", "winddirection_10m", "pressure_msl", "relativehumidity_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "windspeed_10m_max"],
        "timezone": "America/Argentina/Buenos_Aires",
        "forecast_days": 7,  # Pronóstico de 7 días
        "models": [modelo],
    }

    try:
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
            "direccion_viento": hourly.Variables(3).ValuesAsNumpy(),
            "presion_hpa": hourly.Variables(4).ValuesAsNumpy(),
            "humedad": hourly.Variables(5).ValuesAsNumpy()
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
            "lluvia_total": daily.Variables(2).ValuesAsNumpy(),
            "viento_max": daily.Variables(3).ValuesAsNumpy()
        }

        # Crear DataFrame diario
        df_daily = pd.DataFrame(data=daily_data)
        
        return df_hourly, df_daily, response, True, None
        
    except Exception as e:
        return None, None, None, False, str(e)

# Función para comparar modelos
def crear_comparacion_modelos(modelos_lista, variable="temperatura_c", titulo="Comparación de Temperatura"):
    fig = go.Figure()
    
    colores = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    
    for i, modelo in enumerate(modelos_lista):
        df_hourly, _, _, success, error = obtener_datos_clima(modelo)
        
        if success and df_hourly is not None:
            # Filtrar datos cada 6 horas para evitar sobrecarga
            df_filtered = df_hourly[df_hourly['fecha_hora'].dt.hour % 6 == 0].copy()
            
            fig.add_trace(go.Scatter(
                x=df_filtered['fecha_hora'],
                y=df_filtered[variable],
                mode='lines+markers',
                name=MODELOS_DISPONIBLES[modelo]["nombre"],
                line=dict(color=colores[i % len(colores)], width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha y Hora",
        yaxis_title=titulo.split(" de ")[-1] if " de " in titulo else "Valor",
        height=500,
        hovermode='x unified'
    )
    
    return fig

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

# Función para crear resumen diario mejorado
def crear_resumen_diario(df_daily):
    fig = go.Figure()
    
    # Añadir barras de rango de temperatura
    fig.add_trace(go.Bar(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_max'] - df_daily['temp_min'],
        base=df_daily['temp_min'],
        name='Rango de Temperatura',
        marker_color='orange',
        opacity=0.7
    ))
    
    # Añadir puntos de temperatura mínima
    fig.add_trace(go.Scatter(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_min'],
        mode='markers+lines',
        name='Temp. Mínima',
        marker=dict(color='blue', size=10),
        line=dict(color='blue', width=2)
    ))
    
    # Añadir puntos de temperatura máxima
    fig.add_trace(go.Scatter(
        x=df_daily['fecha'].dt.date,
        y=df_daily['temp_max'],
        mode='markers+lines',
        name='Temp. Máxima',
        marker=dict(color='red', size=10),
        line=dict(color='red', width=2)
    ))
    
    # Añadir anotaciones de lluvia como texto
    for i, row in df_daily.iterrows():
        if row['lluvia_total'] > 0:
            fig.add_annotation(
                x=row['fecha'].date(),
                y=row['temp_max'] + 2,
                text=f"💧 {row['lluvia_total']:.1f} mm",
                showarrow=False,
                font=dict(color="blue", size=11, family="Arial Black")
            )
    
    fig.update_layout(
        title="Rango de Temperatura Diaria y Lluvia",
        xaxis_title="Fecha",
        yaxis_title="Temperatura (°C)",
        height=400,
        showlegend=True
    )
    
    return fig

# Lógica principal de la aplicación
try:
    # Mostrar información del modelo seleccionado
    st.info(f"🛰️ **Modelo Activo:** {MODELOS_DISPONIBLES[modelo_seleccionado]['nombre']} | "
            f"**Resolución:** {MODELOS_DISPONIBLES[modelo_seleccionado]['resolucion']} | "
            f"**Origen:** {MODELOS_DISPONIBLES[modelo_seleccionado]['origen']}")
    
    # Mostrar spinner mientras se obtienen los datos
    with st.spinner(f"Obteniendo datos del modelo {MODELOS_DISPONIBLES[modelo_seleccionado]['nombre']}..."):
        df_hourly, df_daily, response, success, error = obtener_datos_clima(modelo_seleccionado)
    
    if not success:
        st.error(f"Error al obtener datos del modelo {modelo_seleccionado}: {error}")
        st.stop()
    
    # Mostrar información de ubicación
    st.subheader("📍 Información de Ubicación")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latitud", f"{response.Latitude()}°S")
    with col2:
        st.metric("Longitud", f"{-response.Longitude()}°O")
    with col3:
        st.metric("Elevación", f"{response.Elevation()} m")
    with col4:
        st.metric("Modelo", MODELOS_DISPONIBLES[modelo_seleccionado]['nombre'])
    
    # Crear tablas pivote
    tabla_temp, tabla_lluvia, tabla_viento = crear_tablas_pivote(df_hourly)
    
    # Crear pestañas para diferentes visualizaciones
    if comparar_modelos and modelos_comparacion:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌡️ Temperatura", "🌧️ Lluvia", "💨 Viento", "📊 Resumen Diario", "🔄 Comparación"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperatura", "🌧️ Lluvia", "💨 Viento", "📊 Resumen Diario"])
    
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
            'Lluvia Total (mm)': df_daily['lluvia_total'],
            'Viento Máximo (km/h)': df_daily['viento_max']
        })
        st.dataframe(lluvia_diaria, use_container_width=True)
    
    with tab3:
        st.subheader("Datos de Viento")
        
        # Crear gráfico de velocidad del viento
        df_wind_filtered = df_hourly[df_hourly['fecha_hora'].dt.hour % 3 == 0].copy()
        
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(
            x=df_wind_filtered['fecha_hora'],
            y=df_wind_filtered['viento_kmh'],
            mode='lines+markers',
            name='Velocidad del Viento',
            line=dict(color='green', width=2),
            marker=dict(size=6, color='darkgreen')
        ))
        
        fig_wind.update_layout(
            title="Velocidad del Viento a lo largo del tiempo",
            xaxis_title="Fecha y Hora",
            yaxis_title="Velocidad del Viento (km/h)",
            height=400
        )
        
        st.plotly_chart(fig_wind, use_container_width=True)
    
    with tab4:
        st.subheader("Resumen Meteorológico Diario")
        st.plotly_chart(crear_resumen_diario(df_daily), use_container_width=True)
        
        # Mostrar tabla de resumen diario
        st.subheader("Tabla de Resumen Diario")
        resumen_diario = pd.DataFrame({
            'Fecha': df_daily['fecha'].dt.date,
            'Temp. Mín. (°C)': df_daily['temp_min'].round(1),
            'Temp. Máx. (°C)': df_daily['temp_max'].round(1),
            'Rango Temp. (°C)': (df_daily['temp_max'] - df_daily['temp_min']).round(1),
            'Lluvia Total (mm)': df_daily['lluvia_total'].round(1),
            'Viento Máx. (km/h)': df_daily['viento_max'].round(1)
        })
        st.dataframe(resumen_diario, use_container_width=True)
    
    # Pestaña de comparación (solo si está habilitada)
    if comparar_modelos and modelos_comparacion:
        with tab5:
            st.subheader("🔄 Comparación entre Modelos Meteorológicos")
            
            # Mostrar modelos que se están comparando
            modelos_texto = [MODELOS_DISPONIBLES[m]["nombre"] for m in [modelo_seleccionado] + modelos_comparacion]
            st.info(f"**Modelos comparados:** {', '.join(modelos_texto)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Comparación de Temperatura")
                fig_temp_comp = crear_comparacion_modelos(
                    [modelo_seleccionado] + modelos_comparacion, 
                    "temperatura_c", 
                    "Comparación de Temperatura (°C)"
                )
                st.plotly_chart(fig_temp_comp, use_container_width=True)
            
            with col2:
                st.subheader("Comparación de Lluvia")
                fig_lluvia_comp = crear_comparacion_modelos(
                    [modelo_seleccionado] + modelos_comparacion, 
                    "lluvia_mm", 
                    "Comparación de Lluvia (mm)"
                )
                st.plotly_chart(fig_lluvia_comp, use_container_width=True)
            
            # Tabla comparativa de diferencias
            st.subheader("📊 Análisis de Diferencias entre Modelos")
            
            # Calcular estadísticas comparativas
            try:
                stats_comparison = []
                
                def safe_round(value, decimals=1):
                    """Función auxiliar para manejar NaN y valores nulos"""
                    if pd.isna(value) or value is None:
                        return "N/A"
                    try:
                        return round(float(value), decimals)
                    except (ValueError, TypeError):
                        return "N/A"
                
                for modelo in [modelo_seleccionado] + modelos_comparacion:
                    df_temp, df_d_temp, _, success_temp, _ = obtener_datos_clima(modelo)
                    if success_temp and df_temp is not None:
                        # Calcular estadísticas con manejo seguro de NaN
                        temp_mean = df_temp['temperatura_c'].mean()
                        temp_max = df_temp['temperatura_c'].max()
                        temp_min = df_temp['temperatura_c'].min()
                        lluvia_sum = df_temp['lluvia_mm'].sum()
                        viento_mean = df_temp['viento_kmh'].mean()
                        
                        stats_comparison.append({
                            'Modelo': MODELOS_DISPONIBLES[modelo]["nombre"],
                            'Temp. Promedio (°C)': safe_round(temp_mean),
                            'Temp. Máx. (°C)': safe_round(temp_max),
                            'Temp. Mín. (°C)': safe_round(temp_min),
                            'Lluvia Total (mm)': safe_round(lluvia_sum),
                            'Viento Promedio (km/h)': safe_round(viento_mean)
                        })
                
                if stats_comparison:
                    df_stats = pd.DataFrame(stats_comparison)
                    st.dataframe(df_stats, use_container_width=True)
                else:
                    st.warning("No se pudieron obtener datos de los modelos seleccionados.")
                    
            except Exception as e:
                st.warning(f"No se pudo generar la tabla comparativa: {e}")
                st.error("Detalles del error para debugging:")
                st.code(str(e))
    
    # Información adicional en el sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Datos Adicionales")
    
    # Mostrar promedios del modelo actual
    if df_hourly is not None:
        st.sidebar.metric("🌡️ Temp. Promedio", f"{df_hourly['temperatura_c'].mean():.1f}°C")
        st.sidebar.metric("🌧️ Lluvia Total", f"{df_hourly['lluvia_mm'].sum():.1f} mm")
        st.sidebar.metric("💨 Viento Promedio", f"{df_hourly['viento_kmh'].mean():.1f} km/h")
        st.sidebar.metric("💧 Humedad Promedio", f"{df_hourly['humedad'].mean():.0f}%")
    
    # Añadir botón de actualización y hora de última actualización
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        if st.button("🔄 Actualizar Datos"):
            st.cache_data.clear()
            st.experimental_rerun()
    with col2:
        st.write(f"**🕒 Última actualización:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    with col3:
        st.write(f"**🛰️ Modelo activo:** {MODELOS_DISPONIBLES[modelo_seleccionado]['nombre']}")
    
    # Añadir pie de página con información de la fuente de datos
    st.markdown("---")
    st.markdown("""
    **📡 Fuentes de datos:**
    - [API de Open-Meteo](https://open-meteo.com/) - Datos meteorológicos
    - **ECMWF**: Centro Europeo de Pronósticos Meteorológicos a Plazo Medio
    - **GFS**: Sistema de Pronóstico Global de NOAA (Estados Unidos)
    - **ICON**: Modelo del Servicio Meteorológico Alemán (DWD)
    - **GEM**: Modelo de Ambiente Global de Canadá (CMC)
    """)
    
except Exception as e:
    st.error(f"❌ Ocurrió un error: {e}")
    st.error("Por favor, intente actualizar la página o verifique su conexión a internet.")
    st.info("💡 **Sugerencia:** Prueba seleccionando un modelo diferente en la barra lateral.")
