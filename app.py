import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="Realirisk StatX", layout="wide")

# Estilo personalizado para tablas y métricas
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOGO Y TÍTULO
# ==========================================
if os.path.exists("mi_logo.png"):
    st.sidebar.image("mi_logo.png", width=200, caption="Grupo Reliarisk")

st.title("Realirisk StatX")
st.markdown("**Plataforma de Caracterización Estadística Avanzada** | *Versión 2.0*")
st.markdown("---")

# ==========================================
# 1. INGESTA Y VISUALIZACIÓN DE DATOS (REQ 5)
# ==========================================
st.sidebar.header("1. Configuración de Datos")
uploaded_file = st.sidebar.file_uploader("Cargar archivo de datos", type=['csv', 'xlsx'])

# Opción para alinearse con Crystal Ball (2 parámetros vs 3 parámetros)
force_loc_zero = st.sidebar.checkbox("Forzar origen en cero (Loc=0)", value=True, 
    help="Si se activa, las distribuciones (Weibull, Gamma, Lognormal) comenzarán estrictamente en 0. Desactívalo para permitir ubicaciones negativas o desplazadas (3 parámetros).")

def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return None

df = load_data(uploaded_file)

if df is not None:
    # Selección de variable
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col_name = st.sidebar.selectbox("Variable a analizar", cols)
    
    # Limpieza de datos
    raw_data = df[col_name]
    data = raw_data.dropna()
    data = data[np.isfinite(data)] # Eliminar inf
    
    # REQ 5: Tabla para validar datos usados
    with st.expander("🔍 Verificación de Datos Fuente (Clic para desplegar)"):
        st.markdown(f"**Variable:** {col_name} | **Registros Válidos:** {len(data)}")
        st.dataframe(data.to_frame().T, height=150) # Transpuesta para vista compacta
        st.caption("Estos son los datos exactos que están entrando al motor de cálculo.")

    # ==========================================
    # 2. MOTOR DE CÁLCULO (REQ 1)
    # ==========================================
    
    # Diccionario de distribuciones
    dist_names = ['norm', 'lognorm', 'weibull_min', 'expon', 'gamma', 'uniform']
    results = []

    # Barra de progreso
    progress_text = "Ajustando distribuciones..."
    my_bar = st.progress(0, text=progress_text)

    for i, name in enumerate(dist_names):
        dist = getattr(stats, name)
        
        # Lógica de Ajuste (Fit)
        # Si force_loc_zero es True, forzamos floc=0 para distribuciones que lo soportan
        try:
            if force_loc_zero and name in ['weibull_min', 'gamma', 'lognorm', 'expon']:
                params = dist.fit(data, floc=0)
            else:
                params = dist.fit(data)
                
            # Formateo de parámetros para lectura humana (REQ 1)
            param_str = ""
            if name == 'norm':
                param_str = f"Media={params[0]:.2f}, Desv={params[1]:.2f}"
            elif name == 'weibull_min':
                param_str = f"Forma={params[0]:.2f}, Escala={params[2]:.2f}, Loc={params[1]:.2f}"
            elif name == 'lognorm':
                # Scipy lognorm: s=sigma, scale=exp(mu). Convertimos para claridad.
                s, loc, scale = params
                param_str = f"Forma(s)={s:.2f}, Escala={scale:.2f} (Mu={np.log(scale):.2f})"
            elif name == 'expon':
                param_str = f"Loc={params[0]:.2f}, Escala={params[1]:.2f}"
            elif name == 'gamma':
                param_str = f"Alpha={params[0]:.2f}, Beta={params[2]:.2f}, Loc={params[1]:.2f}"
            else:
                param_str = ", ".join([f"{p:.2f}" for p in params])

            # Cálculo Anderson-Darling (A2)
            n = len(data)
            sorted_data = np.sort(data)
            cdf_vals = dist.cdf(sorted_data, *params)
            
            # Corrección para extremos logarítmicos
            cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
            
            s_val = np.sum((2*np.arange(1, n+1) - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
            ad_stat = -n - s_val/n
            
            # Cálculo KS para P-Value proxy
            ks_stat, ks_p = stats.kstest(data, name, args=params)

            results.append({
                "Distribución": name.capitalize(),
                "Estadístico AD": ad_stat,
                "Valor P (KS)": ks_p,
                "Parámetros Detectados": param_str, # Columna nueva
                "Object": dist,
                "Params": params
            })
        except Exception as e:
            pass
        
        my_bar.progress((i + 1) / len(dist_names))

    my_bar.empty()
    
    # Crear DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Estadístico AD", ascending=True).reset_index(drop=True)

    # ==========================================
    # 3. INTERFAZ DE RESULTADOS E INTERACCIÓN (REQ 3)
    # ==========================================
    
    col_master_1, col_master_2 = st.columns([4, 6])

    with col_master_1:
        st.subheader("Tabla de Resultados")
        # Mostrar tabla con la nueva columna de parámetros
        st.dataframe(
            results_df[["Distribución", "Estadístico AD", "Valor P (KS)", "Parámetros Detectados"]].style.background_gradient(subset=["Estadístico AD"], cmap="Greens_r"),
            use_container_width=True
        )
        
        st.info("💡 Nota: Un 'Estadístico AD' más bajo indica un mejor ajuste.")

        # REQ 3: Selector de Distribución
        st.markdown("### 🛠️ Análisis Detallado")
        selected_dist_name = st.selectbox(
            "Seleccione la distribución a visualizar:", 
            results_df["Distribución"].tolist()
        )
        
        # Obtener datos de la distribución seleccionada
        selected_row = results_df[results_df["Distribución"] == selected_dist_name].iloc[0]
        sel_dist = selected_row["Object"]
        sel_params = selected_row["Params"]

        # REQ 4: Calculadora de Percentiles
        st.markdown("---")
        st.markdown("### 🧮 Calculadora de Probabilidad")
        
        calc_tab1, calc_tab2 = st.tabs(["Valor ⮕ Percentil", "Percentil ⮕ Valor"])
        
        with calc_tab1:
            val_input = st.number_input(f"Ingresar valor de '{col_name}':", value=float(np.mean(data)))
            perc_result = sel_dist.cdf(val_input, *sel_params) * 100
            st.metric("Percentil (Prob. Acumulada)", f"{perc_result:.2f}%")
            
        with calc_tab2:
            perc_input = st.number_input("Ingresar Percentil (0-100%):", value=50.0, min_value=0.01, max_value=99.99)
            val_result = sel_dist.ppf(perc_input/100, *sel_params)
            st.metric(f"Valor estimado de '{col_name}'", f"{val_result:.4f}")

    with col_master_2:
        # REQ 2: Menú de Gráficos
        st.subheader(f"Visualización: {selected_dist_name}")
        
        plot_type = st.radio(
            "Tipo de Gráfico:",
            ["Densidad (PDF)", "Acumulada (CDF)", "Acumulada Inversa (1-CDF)"],
            horizontal=True
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Datos comunes para plotear líneas
        x_plot = np.linspace(min(data), max(data), 1000)
        
        if plot_type == "Densidad (PDF)":
            # Histograma + PDF
            sns.histplot(data, stat="density", bins='auto', color="#87CEEB", alpha=0.5, label="Datos Reales", ax=ax)
            y_plot = sel_dist.pdf(x_plot, *sel_params)
            ax.plot(x_plot, y_plot, 'r-', lw=2.5, label=f"Ajuste {selected_dist_name}")
            ax.set_title("Función de Densidad de Probabilidad")
            ax.set_ylabel("Densidad")
            
        elif plot_type == "Acumulada (CDF)":
            # Histograma Acumulado + CDF
            sns.histplot(data, stat="density", bins='auto', cumulative=True, element="step", fill=False, color="gray", label="Datos Empíricos", ax=ax)
            y_plot = sel_dist.cdf(x_plot, *sel_params)
            ax.plot(x_plot, y_plot, 'g-', lw=2.5, label=f"CDF {selected_dist_name}")
            ax.set_title("Función de Distribución Acumulada")
            ax.set_ylabel("Probabilidad Acumulada")
            
        elif plot_type == "Acumulada Inversa (1-CDF)":
            # Supervivencia
            # Para datos empíricos de supervivencia:
            sorted_data_inv = np.sort(data)
            y_emp = 1.0 - np.arange(1, len(data)+1) / len(data)
            ax.step(sorted_data_inv, y_emp, where='post', color='gray', label='Datos Empíricos (Survival)')
            
            # Curva teórica
            # Scipy usa .sf (survival function) = 1 - cdf
            y_plot = sel_dist.sf(x_plot, *sel_params) 
            ax.plot(x_plot, y_plot, 'purple', lw=2.5, label=f"1-CDF {selected_dist_name}")
            ax.set_title("Función de Supervivencia (Acumulada Inversa)")
            ax.set_ylabel("Probabilidad (1 - P)")

        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Gráfico adicional P-P Plot siempre útil para bondad de ajuste
        with st.expander("Ver Gráfico P-P (Diagnóstico de Linealidad)"):
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            stats.probplot(data, dist=sel_dist, sparams=sel_params, plot=ax2, fit=False)
            ax2.plot([min(data), max(data)], [min(data), max(data)], 'r--', lw=2)
            st.pyplot(fig2)

else:
    st.info("👋 Sube un archivo CSV o Excel en la barra lateral para comenzar.")