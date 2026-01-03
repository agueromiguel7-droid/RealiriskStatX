import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Configuración de página y Título (Branding Realirisk StatX)
st.set_page_config(page_title="Realirisk StatX", layout="wide")

# ==========================================
# LOGO DE LA EMPRESA
# ==========================================
# Se intenta cargar el logo en la barra lateral. 
# Si el archivo "mi_logo.png" no existe, simplemente no muestra nada (evita errores).
if os.path.exists("mi_logo.png"):
    st.sidebar.image("mi_logo.png", width=200, caption="Realirisk StatX")
else:
    # Opcional: Mostrar advertencia si estás probando y no encuentras el logo
    # st.sidebar.warning("Nota: Sube 'mi_logo.png' para ver el logo aquí.")
    pass

st.title("Realirisk StatX")
st.markdown("""
**Prototipo de Liderazgo en Caracterización Estadística** *Diseñado para análisis de fiabilidad e ingeniería de procesos.*
""")
st.markdown("---")

# ==========================================
# 4.0 FASE 1: INGESTA Y PREPARACIÓN DE DATOS
# ==========================================
st.sidebar.header("1. Ingesta de Datos")

# 4.1 Asistente de Importación
uploaded_file = st.sidebar.file_uploader("Cargar archivo (CSV o Excel)", type=['csv', 'xlsx'])

def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")
            return None
    return None

df = load_data(uploaded_file)

if df is not None:
    # Selección de columnas (Simulando selección de rango)
    col_name = st.sidebar.selectbox("Seleccione la variable a analizar", df.columns)
    data = df[col_name].dropna() # 4.2 Manejo de NaN por defecto (ignorar)
    
    # 4.2 Datos Censurados (Placeholder para funcionalidad futura)
    censored_option = st.sidebar.checkbox("¿Contiene datos censurados?", value=False)
    if censored_option:
        st.sidebar.info("La funcionalidad avanzada de datos censurados se habilitará en la versión completa (Prioridad 4).")

    # ==========================================
    # 5.0 FASE 2: ANÁLISIS EXPLORATORIO (EDA)
    # ==========================================
    st.header("2. Análisis Exploratorio de Datos (EDA)")
    
    # 69-70 Estadísticos Descriptivos Fundamentales
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("N (Muestra)", f"{len(data)}")
    col2.metric("Media", f"{np.mean(data):.4f}")
    col3.metric("Desv. Estándar", f"{np.std(data):.4f}")
    col4.metric("Asimetría (Skew)", f"{stats.skew(data):.4f}")
    col5.metric("Curtosis", f"{stats.kurtosis(data):.4f}")

    # ==========================================
    # 6.0 FASE 3: MOTOR CENTRAL DE PRUEBAS DE BONDAD DE AJUSTE
    # ==========================================
    st.header("3. Motor de Ajuste y Comparación")
    
    # 87-88 Biblioteca de Distribuciones Soportadas
    dist_names = ['norm', 'lognorm', 'weibull_min', 'expon', 'gamma', 'uniform']
    results = []

    progress_bar = st.progress(0)
    
    for i, name in enumerate(dist_names):
        dist = getattr(stats, name)
        
        # Ajuste de parámetros (MLE)
        params = dist.fit(data)
        
        # 78 Cálculo del Estadístico Anderson-Darling (AD)
        # Nota: Scipy 'anderson' tiene soporte limitado. Calculamos el estadístico 'ad_stat' manualmente 
        # o usamos la prueba KS como proxy para p-value genérico en este prototipo rápido.
        
        # Kolmogorov-Smirnov para obtener un p-value genérico
        ks_stat, ks_p = stats.kstest(data, name, args=params)
        
        # Anderson-Darling (A2) custom calculation approximation for ranking
        n = len(data)
        sorted_data = np.sort(data)
        cdf_vals = dist.cdf(sorted_data, *params)
        
        # Evitar log(0)
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
        
        # Fórmula A^2
        s = np.sum((2*np.arange(1, n+1) - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
        ad_stat = -n - s/n
        
        results.append({
            "Distribución": name.capitalize(),
            "Estadístico AD": ad_stat, # 103 Menor es mejor
            "Valor p (KS)": ks_p,     # Proxy hasta implementar Monte Carlo completo
            "Parámetros": str([round(p, 4) for p in params]),
            "Object": dist,
            "Params": params
        })
        progress_bar.progress((i + 1) / len(dist_names))

    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    # 104 Ordenar por Estadístico AD (Mejor ajuste primero)
    results_df = results_df.sort_values(by="Estadístico AD", ascending=True).reset_index(drop=True)
    
    # Mejor distribución detectada
    best_dist_row = results_df.iloc[0]
    best_dist_name = best_dist_row["Distribución"]
    
    # ==========================================
    # 7.0 FASE 4: PRESENTACIÓN E INTERPRETACIÓN
    # ==========================================
    
    # 105-111 Vista Dividida (Split View)
    st.subheader(f"Mejor Ajuste Identificado: {best_dist_name}")
    
    row1_col1, row1_col2 = st.columns([1, 2])
    
    with row1_col1:
        st.markdown("### Tabla Comparativa")
        # 101 Tabla con columnas requeridas
        st.dataframe(results_df[["Distribución", "Estadístico AD", "Valor p (KS)"]].style.highlight_min(subset=["Estadístico AD"], color="lightgreen"))
        
        # 113-116 Guía de Interpretación
        with st.expander("ℹ️ Ayuda de Interpretación"):
            st.markdown("""
            * **Estadístico AD:** Mide la distancia entre los datos y la distribución, pesando más las colas. *Menor valor = Mejor ajuste*.
            * **Valor p:** Probabilidad de que los datos provengan de esta distribución. *Valor bajo (<0.05) sugiere rechazo*.
            """)

    with row1_col2:
        st.markdown("### Diagnóstico Visual")
        # Gráficos combinados
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 110 Histograma + PDF
        sns.histplot(data, kde=False, stat="density", bins='auto', color="skyblue", alpha=0.6, ax=axes[0], label="Datos")
        x_plot = np.linspace(min(data), max(data), 1000)
        pdf_plot = best_dist_row["Object"].pdf(x_plot, *best_dist_row["Params"])
        axes[0].plot(x_plot, pdf_plot, 'r-', lw=2, label=f"Ajuste {best_dist_name}")
        axes[0].set_title("Histograma y Densidad (PDF)")
        axes[0].legend()
        
        # 109 Gráfico P-P
        stats.probplot(data, dist=best_dist_row["Object"], sparams=best_dist_row["Params"], plot=axes[1], fit=False)
        axes[1].plot([min(data), max(data)], [min(data), max(data)], 'r--', lw=2)
        axes[1].set_title("Gráfico de Probabilidad (P-P Plot)")
        
        st.pyplot(fig)

    # ==========================================
    # 8.0 FASE 5: REPORTE Y EXPORTACIÓN
    # ==========================================
    st.header("5. Exportación")
    
    # 122 Exportar Tabla de Resultados
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Tabla de Resultados (CSV)",
        data=csv,
        file_name='realirisk_statx_results.csv',
        mime='text/csv',
    )
    
    # 124 Generar Informe
    report_text = f"""
    # Informe Realirisk StatX
    Variable Analizada: {col_name}
    Mejor Ajuste: {best_dist_name}
    Estadístico AD: {best_dist_row['Estadístico AD']:.4f}
    
    Este informe fue generado automáticamente por Realirisk StatX.
    """
    st.download_button(
        label="📄 Descargar Informe Resumido (TXT)",
        data=report_text,
        file_name='informe_statx.txt',
        mime='text/plain'
    )

else:
    st.info("👋 Por favor, carga un archivo CSV o Excel en la barra lateral para comenzar el análisis.")