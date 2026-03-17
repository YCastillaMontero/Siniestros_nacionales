"""
================================================================================
ANÁLISIS DE SINIESTROS NACIONALES - APLICACIÓN STREAMLIT
================================================================================
Autor: Análisis de Datos Avanzado
Dataset: Siniestros Nacionales (Sector Recolección de Residuos)
Tecnologías: Streamlit, Pandas, Plotly, NumPy
================================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Análisis de Siniestros Nacionales",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS PERSONALIZADOS — DISEÑO MODERNO PROFESIONAL
# ============================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
    /* ── Variables de diseño ── */
    :root {
        --navy:     #0d1b2a;
        --navy-mid: #1b2e45;
        --navy-light: #243b55;
        --accent:   #00c2ff;
        --accent2:  #ff6b35;
        --text:     #e8edf3;
        --text-muted: #8ba3bc;
        --card-bg:  #162333;
        --border:   rgba(0,194,255,0.15);
        --radius:   14px;
        --shadow:   0 8px 32px rgba(0,0,0,0.35);
    }

    /* ── Reset general ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    .main, .block-container {
        background-color: var(--navy) !important;
        padding: 3.5rem 2rem 2rem 2rem!important;
    }
    /* Evita que el h1 quede recortado por el header de Streamlit */
    .block-container > div:first-child {
        padding-top: 0.25rem;
    } 
            
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--navy-mid) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* ── Botones de navegación ── */
    .stButton > button {
        width: 100%;
        background: transparent;
        color: var(--text) !important;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        text-align: left;
        transition: all 0.22s ease;
        margin-bottom: 4px;
    }
    .stButton > button:hover {
        background: rgba(0,194,255,0.1) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        transform: translateX(4px);
        box-shadow: 0 0 16px rgba(0,194,255,0.12);
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(0,194,255,0.35) !important;
    }

    /* ── Encabezados generales ── */
    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif !important;
    }

    /* ── Títulos h1 de sección ── */
    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem !important;
        padding: 0 !important;
        background: none !important;
        box-shadow: none !important;
    }

    /* Línea decorativa bajo h1 */
    h1::after {
        content: '';
        display: block;
        width: 56px;
        height: 3px;
        background: var(--accent);
        margin: 0.55rem auto 1.5rem;
        border-radius: 2px;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        border-bottom: 2px solid var(--border) !important;
        padding-bottom: 0.4rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        margin-top: 1.2rem !important;
    }

    h4 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--accent) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Métricas ── */
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
    }
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.3rem !important;
        box-shadow: var(--shadow);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(0,194,255,0.3);
    }

    /* ── Cajas de información personalizadas ── */
    .info-box {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .info-box h3 {
        color: var(--accent) !important;
        margin-top: 0 !important;
        font-size: 1rem !important;
    }
    .info-box p, .info-box li {
        color: var(--text-muted) !important;
        line-height: 1.75;
    }

    .warning-box {
        background: var(--card-bg);
        border: 1px solid rgba(255,107,53,0.25);
        border-left: 4px solid var(--accent2);
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .warning-box h3 {
        color: var(--accent2) !important;
        margin-top: 0 !important;
        font-size: 1rem !important;
    }
    .warning-box p, .warning-box li {
        color: var(--text-muted) !important;
        line-height: 1.75;
    }

    .success-box {
        background: var(--card-bg);
        border: 1px solid rgba(0,210,150,0.2);
        border-left: 4px solid #00d296;
        border-radius: var(--radius);
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }
    .success-box h3 {
        color: #00d296 !important;
        margin-top: 0 !important;
        font-size: 1rem !important;
    }
    .success-box p, .success-box li {
        color: var(--text-muted) !important;
        line-height: 1.75;
    }

    /* ── Divisores ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.8rem 0 !important;
    }

    /* ── Tablas / DataFrames ── */
    .dataframe, [data-testid="stDataFrame"] {
        font-size: 0.85rem !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stDataFrame"] th {
        background: var(--navy-light) !important;
        color: var(--accent) !important;
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    [data-testid="stDataFrame"] td {
        color: var(--text) !important;
        background: var(--card-bg) !important;
    }

    /* ── Selectboxes / Multiselect ── */
    [data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
        background: var(--card-bg) !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.2s;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: rgba(0,194,255,0.07) !important;
    }
    [data-testid="stTabs"] button:hover {
        color: var(--text) !important;
    }

    /* ── Info / Warning nativa de Streamlit ── */
    [data-testid="stAlert"] {
        background: var(--card-bg) !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.8rem 2rem;
        margin-top: 3rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .footer strong {
        color: var(--accent);
        font-family: 'Syne', sans-serif;
    }

    /* ── Sidebar logo area ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }
    .sidebar-brand-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.05rem;
        color: #ffffff;
        line-height: 1.2;
    }
    .sidebar-brand-sub {
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Nav section label ── */
    .nav-label {
        font-size: 0.67rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.2rem 0 0.4rem 0.3rem;
    }

    /* ── Nav item con imagen ── */
    .nav-item-img {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.55rem 0.85rem;
        border-radius: 10px;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all 0.22s ease;
        margin-bottom: 3px;
        text-decoration: none;
    }
    .nav-item-img:hover {
        background: rgba(0,194,255,0.08);
        border-color: rgba(0,194,255,0.25);
    }
    .nav-icon {
        font-size: 1.2rem;
        width: 28px;
        text-align: center;
    }
    .nav-text {
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--text);
    }

    /* ── Sidebar stats card ── */
    .stats-card {
        background: rgba(0,194,255,0.07);
        border: 1px solid rgba(0,194,255,0.18);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
    }
    .stats-card-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 0.28rem 0;
        border-bottom: 1px solid rgba(0,194,255,0.08);
    }
    .stats-card-row:last-child { border-bottom: none; }
    .stats-card-label {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    .stats-card-value {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        color: var(--accent);
    }

    /* ── Hero section (inicio) ── */
    .hero-section {
        background: linear-gradient(135deg, var(--navy-mid) 0%, var(--navy-light) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(0,194,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(0,194,255,0.1);
        border: 1px solid rgba(0,194,255,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 0.8rem;
    }
    .hero-title span {
        color: var(--accent);
    }
    .hero-desc {
        font-size: 0.96rem;
        color: var(--text-muted);
        line-height: 1.75;
        max-width: 680px;
    }

    /* ── Section header chip ── */
    .section-chip {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent);
        background: rgba(0,194,255,0.08);
        border: 1px solid rgba(0,194,255,0.2);
        border-radius: 20px;
        padding: 0.22rem 0.75rem;
        margin-bottom: 0.4rem;
        display: block;
        width: fit-content;
    }

    /* ── Plotly chart fix: fondo transparente ── */
    .js-plotly-plot .plotly, .js-plotly-plot .plot-container {
        background: transparent !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--navy); }
    ::-webkit-scrollbar-thumb { background: var(--navy-light); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

@st.cache_data
def cargar_datos():
    """
    Carga y prepara el dataset de siniestros nacionales.
    
    Returns:
        pd.DataFrame: DataFrame limpio y procesado
    """
    try:
        # Cargar datos
        df = pd.read_csv("Siniestros_nacionales_aleatorio_trim6.csv", encoding='latin-1', sep=';')
        
        # Limpieza básica
        df.columns = df.columns.str.strip()
        
        # Convertir fecha a datetime si no lo está
        if df['Fecha Siniestro'].dtype != 'datetime64[ns]':
            df['Fecha Siniestro'] = pd.to_datetime(df['Fecha Siniestro'])
        
        # Crear columnas derivadas
        df['Año'] = df['Fecha Siniestro'].dt.year
        df['Mes'] = df['Fecha Siniestro'].dt.month
        df['Mes_Nombre'] = df['Fecha Siniestro'].dt.month_name()
        df['Día_Semana'] = df['Fecha Siniestro'].dt.day_name()
        df['Trimestre'] = df['Fecha Siniestro'].dt.quarter
        df['Semestre'] = df['Mes'].apply(lambda x: 'S1' if x <= 6 else 'S2')
        
        # Categoría de gravedad
        df['Gravedad'] = pd.cut(
            df['Tiempo Inoperativo ( HORAS )'],
            bins=[-np.inf, 0, 2, 24, 100, np.inf],
            labels=['Sin tiempo', 'Leve', 'Moderado', 'Grave', 'Crítico']
        )
        
        return df
    
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Cargar datos
df = cargar_datos()

# ============================================================================
# FUNCIONES DE NAVEGACIÓN
# ============================================================================

def cambiar_vista(vista_activa):
    """Cambia entre diferentes vistas de la aplicación."""
    vistas = [
        'inicio', 'resumen_general', 'analisis_temporal', 'analisis_geografico',
        'analisis_causas', 'analisis_vehiculos', 'analisis_conductores',
        'analisis_gravedad', 'consulta_especifica', 'conclusiones'
    ]
    for vista in vistas:
        st.session_state[vista] = (vista == vista_activa)

# Inicializar estado de sesión
if 'inicio' not in st.session_state:
    st.session_state.inicio = True
    for vista in ['resumen_general', 'analisis_temporal', 'analisis_geografico',
                  'analisis_causas', 'analisis_vehiculos', 'analisis_conductores',
                  'analisis_gravedad', 'consulta_especifica', 'conclusiones']:
        st.session_state[vista] = False

# ============================================================================
# FUNCIONES AUXILIARES PARA ANÁLISIS
# ============================================================================

def crear_metrica_card(label, value, delta=None, delta_color="normal"):
    """Crea una métrica visual mejorada."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ── Paleta y layout base para todos los gráficos ──
_COLORS = ['#00c2ff', '#ff6b35', '#00d296', '#a78bfa', '#f59e0b',
           '#ec4899', '#34d399', '#60a5fa', '#fb923c', '#a3e635']

_LAYOUT_BASE = dict(
    font=dict(family="DM Sans, sans-serif", color="#8ba3bc"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,35,51,0.6)",
    title_font=dict(family="Syne, sans-serif", size=15, color="#e8edf3"),
    title_x=0.5,
    title_pad=dict(t=8, b=4),
    margin=dict(l=28, r=28, t=60, b=32),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1b2e45",
        bordercolor="#00c2ff",
        font=dict(family="DM Sans, sans-serif", color="#e8edf3")
    ),
    xaxis=dict(
        gridcolor="rgba(0,194,255,0.07)",
        linecolor="rgba(0,194,255,0.15)",
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        gridcolor="rgba(0,194,255,0.07)",
        linecolor="rgba(0,194,255,0.15)",
        tickfont=dict(size=11)
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,194,255,0.15)",
        borderwidth=1,
        font=dict(size=11)
    )
)

def _apply_base(fig, height=480):
    layout = dict(_LAYOUT_BASE)
    layout['height'] = height
    fig.update_layout(**layout)
    return fig

def crear_grafico_barras(df_data, x, y, title, color=None, orientation='v'):
    """Crea un gráfico de barras con tema oscuro moderno."""
    fig = px.bar(
        df_data, x=x, y=y, title=title, color=color,
        orientation=orientation,
        color_discrete_sequence=_COLORS
    )
    _apply_base(fig)
    fig.update_traces(marker_line_width=0)
    return fig

def crear_grafico_lineas(df_data, x, y, title, group=None):
    """Crea un gráfico de líneas con tema oscuro moderno."""
    kwargs = dict(x=x, y=y, title=title, markers=True,
                  color_discrete_sequence=_COLORS)
    if group:
        kwargs['color'] = group
    fig = px.line(df_data, **kwargs)
    _apply_base(fig)
    fig.update_traces(line_width=2.5)
    return fig

def crear_grafico_pie(labels, values, title):
    """Crea un gráfico de pastel con tema oscuro moderno."""
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.42,
        textposition='inside', textinfo='percent+label',
        marker=dict(colors=_COLORS, line=dict(color='rgba(13,27,42,0.8)', width=2)),
        textfont=dict(size=11, family="DM Sans, sans-serif")
    )])
    _apply_base(fig)
    fig.update_layout(title_text=title, showlegend=True)
    return fig

# ============================================================================
# SECCIÓN: INICIO
# ============================================================================

if st.session_state.inicio:
    st.markdown("<h1>🚛 Análisis de Siniestros Nacionales</h1>", unsafe_allow_html=True)
 
    # ── Cálculos para el hero ──
    _total        = len(df)
    _desde        = df['Fecha Siniestro'].min().strftime('%B %Y')
    _hasta        = df['Fecha Siniestro'].max().strftime('%B %Y')
    _depts        = df['DEPARTAMENTO'].nunique()
    _cities       = df['CIUDAD'].nunique()
    _pct_culp     = (df['Culpabilidad Conductor'] == 'SI').sum() / _total * 100
    _tiempo_prom  = df['Tiempo Inoperativo ( HORAS )'].mean()
    _causa_top    = df['Causa Siniestro'].value_counts().idxmax()
    _causa_top_pct= df['Causa Siniestro'].value_counts().iloc[0] / _total * 100
    _dept_top     = df['DEPARTAMENTO'].value_counts().idxmax()
    _dept_top_pct = df['DEPARTAMENTO'].value_counts().iloc[0] / _total * 100
    _turno_top    = df['Turno'].value_counts().idxmax()
    _turno_pct    = df['Turno'].value_counts().iloc[0] / _total * 100
    _pct_grua     = (df['Usó Grua'] == 'SI').sum() / _total * 100
 
    hero_html = f"""
    <div class=\"hero-section\">
        <div class=\"hero-tag\">• Sector Recolección de Residuos — Colombia •</div>
        <div class=\"hero-title\">Plataforma de <span>Análisis Integral</span><br>de Siniestros Viales</div>
        <div class=\"hero-desc\">
            Esta plataforma centraliza y analiza <strong style=\"color:#e8edf3;\">{_total:,}</strong> siniestros
            viales ocurridos en la flota de recolección de residuos entre
            <strong style=\"color:#e8edf3;\">{_desde}</strong> y <strong style=\"color:#e8edf3;\">{_hasta}</strong>,
            con cobertura en <strong style=\"color:#00c2ff;\">{_depts} departamentos</strong>
            y <strong style=\"color:#00c2ff;\">{_cities} ciudades</strong> de Colombia.
            Su propósito es convertir los datos de accidentalidad operativa en inteligencia accionable
            para reducir riesgos, optimizar rutas y fortalecer la seguridad vial de la operación.
        </div>
        <div style=\"border-top:1px solid rgba(0,194,255,0.15); margin:1.4rem 0 1.2rem 0;\"></div>
        <div style=\"display:flex; flex-wrap:wrap; gap:1.2rem;\">
            <div style=\"background:rgba(0,194,255,0.08); border:1px solid rgba(0,194,255,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Causa principal</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#00c2ff;\">{_causa_top}</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">{_causa_top_pct:.1f}% del total</div>
            </div>
            <div style=\"background:rgba(255,107,53,0.08); border:1px solid rgba(255,107,53,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Departamento crítico</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#ff6b35;\">{_dept_top}</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">{_dept_top_pct:.1f}% de los siniestros</div>
            </div>
            <div style=\"background:rgba(0,210,150,0.08); border:1px solid rgba(0,210,150,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Culpabilidad conductor</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#00d296;\">{_pct_culp:.1f}%</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">de casos con responsabilidad</div>
            </div>
            <div style=\"background:rgba(167,139,250,0.08); border:1px solid rgba(167,139,250,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Tiempo inop. promedio</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#a78bfa;\">{_tiempo_prom:.1f} h</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">por siniestro registrado</div>
            </div>
            <div style=\"background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Turno predominante</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#f59e0b;\">{_turno_top}</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">{_turno_pct:.1f}% de ocurrencias</div>
            </div>
            <div style=\"background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.2); border-radius:10px; padding:0.7rem 1.1rem; min-width:160px;\">
                <div style=\"font-size:0.65rem; color:#8ba3bc; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.2rem;\">Uso de grúa</div>
                <div style=\"font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#ec4899;\">{_pct_grua:.1f}%</div>
                <div style=\"font-size:0.75rem; color:#8ba3bc;\">de siniestros requirieron grúa</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # Imagen adicional (mantener estética del inicio)
    st.markdown("### 📌 Imagen Destacada")
    st.image("Imagen1.png", caption="Flota de recolección y contexto operativo", width='stretch')
    st.markdown("""
    <div style='background: rgba(22,35,51,0.7); border: 1px solid rgba(0,194,255,0.2); border-radius: 14px; padding: 1rem; margin-top: 0.5rem;'>
        <h4 style='color:#00c2ff; margin-bottom:0.5rem;'>Contexto visual</h4>
        <p style='color:#d3deef; margin:0;'>Imagen representativa de siniestros en la operación de residuos, útil para comunicar el alcance del análisis visual en campo.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs principales
    st.markdown("### 📈 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Siniestros",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        pct_culpable = (df['Culpabilidad Conductor'] == 'SI').sum() / len(df) * 100
        st.metric(
            label="Conductor Culpable",
            value=f"{pct_culpable:.1f}%",
            delta=f"{(df['Culpabilidad Conductor'] == 'SI').sum()} casos"
        )
    
    with col3:
        promedio_tiempo = df['Tiempo Inoperativo ( HORAS )'].mean()
        st.metric(
            label="Tiempo Inoperativo Promedio",
            value=f"{promedio_tiempo:.1f}h",
            delta=None
        )
    
    with col4:
        pct_reportado = (df['Reportado Aseguradora'] == 'SI').sum() / len(df) * 100
        st.metric(
            label="Reportado a Aseguradora",
            value=f"{pct_reportado:.1f}%",
            delta=None
        )
    
    # Gráficos principales de resumen
    st.markdown("### 📊 Distribuciones Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Siniestros por departamento
        dept_counts = df['DEPARTAMENTO'].value_counts()
        fig = crear_grafico_pie(dept_counts.index, dept_counts.values, "Distribución por Departamento")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Causas de siniestros
        causa_counts = df['Causa Siniestro'].value_counts()
        fig = crear_grafico_pie(causa_counts.index, causa_counts.values, "Distribución por Causa")
        st.plotly_chart(fig, use_container_width=True)
    
    # Información adicional
    st.markdown("""
    <div class="success-box">
        <h3>🎯 ¿Qué puedes hacer con esta plataforma?</h3>
        <ul style="font-size: 1.05rem; line-height: 2;">
            <li><strong>Resumen General:</strong> Visualiza estadísticas globales y KPIs principales</li>
            <li><strong>Análisis Temporal:</strong> Explora tendencias por año, mes y día de la semana</li>
            <li><strong>Análisis Geográfico:</strong> Identifica zonas críticas por departamento y ciudad</li>
            <li><strong>Análisis de Causas:</strong> Comprende factores que generan siniestros</li>
            <li><strong>Análisis de Vehículos:</strong> Estudia el desempeño por tipo de vehículo</li>
            <li><strong>Análisis de Conductores:</strong> Evalúa patrones de culpabilidad</li>
            <li><strong>Análisis de Gravedad:</strong> Clasifica siniestros por impacto operativo</li>
            <li><strong>Consultas Específicas:</strong> Filtra y busca información personalizada</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Objetivo del Análisis</h3>
        <p style="font-size: 1.05rem; line-height: 1.8;">
            Esta plataforma busca identificar patrones, tendencias y áreas de mejora en la operación 
            de vehículos de recolección de residuos. Los insights generados pueden ayudar a:
        </p>
        <ul style="font-size: 1.05rem; line-height: 2;">
            <li>Reducir la tasa de siniestros mediante capacitación focalizada</li>
            <li>Optimizar rutas y horarios operativos</li>
            <li>Mejorar el mantenimiento preventivo de vehículos</li>
            <li>Implementar políticas de seguridad vial más efectivas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SECCIÓN: RESUMEN GENERAL
# ============================================================================

elif st.session_state.resumen_general:
    st.markdown("<h1>📊 Resumen General de Siniestros</h1>", unsafe_allow_html=True)
    
    # Filtros globales
    st.sidebar.markdown("### 🔍 Filtros Globales")
    
    # Filtro por año
    años_disponibles = sorted(df['Año'].unique())
    año_seleccionado = st.sidebar.multiselect(
        "Selecciona Año(s)",
        options=años_disponibles,
        default=años_disponibles
    )
    
    # Filtro por departamento
    departamentos_disponibles = sorted(df['DEPARTAMENTO'].unique())
    dept_seleccionado = st.sidebar.multiselect(
        "Selecciona Departamento(s)",
        options=departamentos_disponibles,
        default=departamentos_disponibles
    )
    
    # Aplicar filtros
    df_filtrado = df[
        (df['Año'].isin(año_seleccionado)) &
        (df['DEPARTAMENTO'].isin(dept_seleccionado))
    ]
    
    if df_filtrado.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados")
    else:
        # KPIs con datos filtrados
        st.markdown("### 📈 Métricas Principales")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Siniestros", f"{len(df_filtrado):,}")
        
        with col2:
            conductor_culpable = (df_filtrado['Culpabilidad Conductor'] == 'SI').sum()
            pct = conductor_culpable / len(df_filtrado) * 100
            st.metric("Conductor Culpable", f"{conductor_culpable:,}", f"{pct:.1f}%")
        
        with col3:
            tiempo_promedio = df_filtrado['Tiempo Inoperativo ( HORAS )'].mean()
            st.metric("Tiempo Inoper. Promedio", f"{tiempo_promedio:.1f}h")
        
        with col4:
            con_grua = (df_filtrado['Usó Grua'] == 'SI').sum()
            pct_grua = con_grua / len(df_filtrado) * 100
            st.metric("Uso de Grúa", f"{con_grua:,}", f"{pct_grua:.1f}%")
        
        with col5:
            reportados = (df_filtrado['Reportado Aseguradora'] == 'SI').sum()
            pct_reportado = reportados / len(df_filtrado) * 100
            st.metric("Reportados", f"{reportados:,}", f"{pct_reportado:.1f}%")
        
        # Gráficos de distribución
        st.markdown("---")
        st.markdown("### 📊 Distribuciones y Comparativas")
        
        tab1, tab2, tab3 = st.tabs(["🏙️ Por Ubicación", "🚗 Por Tipo", "⏰ Por Turno"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 ciudades
                top_ciudades = df_filtrado['CIUDAD'].value_counts().head(10)
                fig = crear_grafico_barras(
                    pd.DataFrame({'Ciudad': top_ciudades.index, 'Siniestros': top_ciudades.values}),
                    x='Siniestros',
                    y='Ciudad',
                    title="Top 10 Ciudades con Más Siniestros",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribución por departamento
                dept_data = df_filtrado.groupby('DEPARTAMENTO').agg({
                    'CIUDAD': 'count',
                    'Tiempo Inoperativo ( HORAS )': 'mean'
                }).round(2)
                dept_data.columns = ['Cantidad', 'Tiempo Promedio (h)']
                st.markdown("#### Resumen por Departamento")
                st.dataframe(dept_data, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Tipo de vehículo
                vehiculo_counts = df_filtrado['Tipo de vehículo'].value_counts().head(8)
                fig = crear_grafico_barras(
                    pd.DataFrame({'Vehículo': vehiculo_counts.index, 'Cantidad': vehiculo_counts.values}),
                    x='Vehículo',
                    y='Cantidad',
                    title="Siniestros por Tipo de Vehículo"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Tipo de accidente
                accidente_counts = df_filtrado['Tipo accidente'].value_counts().head(8)
                fig = crear_grafico_barras(
                    pd.DataFrame({'Tipo': accidente_counts.index, 'Cantidad': accidente_counts.values}),
                    x='Cantidad',
                    y='Tipo',
                    title="Tipos de Accidente Más Comunes",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Siniestros por turno
                turno_counts = df_filtrado['Turno'].value_counts()
                fig = crear_grafico_pie(
                    turno_counts.index,
                    turno_counts.values,
                    "Distribución por Turno"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Culpabilidad por turno
                culpa_turno = pd.crosstab(
                    df_filtrado['Turno'],
                    df_filtrado['Culpabilidad Conductor'],
                    normalize='index'
                ) * 100
                
                fig = go.Figure()
                for i, col in enumerate(culpa_turno.columns):
                    fig.add_trace(go.Bar(
                        name=col,
                        x=culpa_turno.index,
                        y=culpa_turno[col],
                        text=culpa_turno[col].round(1),
                        textposition='inside',
                        marker_color=_COLORS[i % len(_COLORS)],
                        marker_line_width=0
                    ))

                fig.update_layout(
                    **_LAYOUT_BASE,
                    title="Culpabilidad del Conductor por Turno (%)",
                    barmode='stack',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS TEMPORAL
# ============================================================================

elif st.session_state.analisis_temporal:
    st.markdown("<h1>📅 Análisis Temporal de Siniestros</h1>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Evolución en el Tiempo")
    
    # Siniestros por año
    col1, col2 = st.columns([2, 1])
    
    with col1:
        siniestros_año = df.groupby('Año').size().reset_index(name='Cantidad')
        fig = crear_grafico_lineas(
            siniestros_año,
            x='Año',
            y='Cantidad',
            title="Evolución de Siniestros por Año"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Estadísticas por Año")
        resumen_anual = df.groupby('Año').agg({
            'CIUDAD': 'count',
            'Tiempo Inoperativo ( HORAS )': 'mean',
            'Culpabilidad Conductor': lambda x: (x == 'SI').sum()
        }).round(2)
        resumen_anual.columns = ['Total', 'Tiempo Prom (h)', 'Conductor Culpable']
        st.dataframe(resumen_anual, use_container_width=True)
    
    # Análisis mensual
    st.markdown("---")
    st.markdown("### 📆 Patrones Mensuales")
    
    # Selector de año para análisis mensual
    año_analisis = st.selectbox(
        "Selecciona un año para análisis detallado",
        options=sorted(df['Año'].unique(), reverse=True)
    )
    
    df_año = df[df['Año'] == año_analisis]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Siniestros por mes
        siniestros_mes = df_año.groupby('Mes').size().reset_index(name='Cantidad')
        meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        siniestros_mes['Mes_Nombre'] = siniestros_mes['Mes'].apply(lambda x: meses_nombres[x-1])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=siniestros_mes['Mes_Nombre'],
            y=siniestros_mes['Cantidad'],
            marker_color='#00c2ff',
            marker_line_width=0,
            text=siniestros_mes['Cantidad'],
            textposition='outside',
            textfont=dict(color='#8ba3bc', size=10)
        ))
        fig.update_layout(
            **_LAYOUT_BASE,
            title=f"Siniestros por Mes — {año_analisis}",
            xaxis_title="Mes",
            yaxis_title="Cantidad",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Siniestros por día de la semana
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        dia_counts = df_año['Día_Semana'].value_counts().reindex(orden_dias, fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dias_es,
            y=dia_counts.values,
            marker_color='#ff6b35',
            marker_line_width=0,
            text=dia_counts.values,
            textposition='outside',
            textfont=dict(color='#8ba3bc', size=10)
        ))
        fig.update_layout(
            **_LAYOUT_BASE,
            title=f"Siniestros por Día de la Semana — {año_analisis}",
            xaxis_title="Día",
            yaxis_title="Cantidad",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de trimestres
    st.markdown("---")
    st.markdown("### 📊 Análisis Trimestral")
    
    trimestre_data = df_año.groupby('Trimestre').agg({
        'CIUDAD': 'count',
        'Tiempo Inoperativo ( HORAS )': 'mean',
        'Culpabilidad Conductor': lambda x: (x == 'SI').sum() / len(x) * 100
    }).round(2)
    trimestre_data.columns = ['Total Siniestros', 'Tiempo Promedio (h)', '% Culpabilidad']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Total Siniestros',
        x=['Q1', 'Q2', 'Q3', 'Q4'],
        y=trimestre_data['Total Siniestros'],
        yaxis='y',
        offsetgroup=1,
        marker_color='#00c2ff',
        marker_line_width=0
    ))
    fig.add_trace(go.Scatter(
        name='% Culpabilidad',
        x=['Q1', 'Q2', 'Q3', 'Q4'],
        y=trimestre_data['% Culpabilidad'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#ff6b35', width=2.5),
        marker=dict(size=8, color='#ff6b35')
    ))

    base = dict(_LAYOUT_BASE)
    base.update(dict(
        title=f'Siniestros y Culpabilidad por Trimestre — {año_analisis}',
        yaxis=dict(title='Total Siniestros', gridcolor='rgba(0,194,255,0.07)',
                   linecolor='rgba(0,194,255,0.15)', tickfont=dict(size=11)),
        yaxis2=dict(title='% Culpabilidad', overlaying='y', side='right',
                    gridcolor='rgba(0,0,0,0)', tickfont=dict(size=11)),
        height=500
    ))
    fig.update_layout(**base)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS GEOGRÁFICO
# ============================================================================

elif st.session_state.analisis_geografico:
    st.markdown("<h1>🗺️ Análisis Geográfico de Siniestros</h1>", unsafe_allow_html=True)
    
    st.markdown("### 📍 Distribución por Departamento y Ciudad")
    
    # Seleccionar departamento
    departamento_seleccionado = st.selectbox(
        "Selecciona un Departamento",
        options=['TODOS'] + sorted(df['DEPARTAMENTO'].unique().tolist())
    )
    
    if departamento_seleccionado == 'TODOS':
        df_geo = df.copy()
    else:
        df_geo = df[df['DEPARTAMENTO'] == departamento_seleccionado]
    
    # KPIs geográficos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Siniestros", f"{len(df_geo):,}")
    
    with col2:
        ciudades_afectadas = df_geo['CIUDAD'].nunique()
        st.metric("Ciudades Afectadas", ciudades_afectadas)
    
    with col3:
        tiempo_promedio = df_geo['Tiempo Inoperativo ( HORAS )'].mean()
        st.metric("Tiempo Inoper. Prom.", f"{tiempo_promedio:.1f}h")
    
    with col4:
        pct_culpa = (df_geo['Culpabilidad Conductor'] == 'SI').sum() / len(df_geo) * 100
        st.metric("% Culpabilidad", f"{pct_culpa:.1f}%")
    
    st.markdown("---")
    
    # Visualizaciones geográficas
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 15 ciudades
        top_ciudades = df_geo['CIUDAD'].value_counts().head(15)
        fig = crear_grafico_barras(
            pd.DataFrame({'Ciudad': top_ciudades.index, 'Siniestros': top_ciudades.values}),
            x='Siniestros',
            y='Ciudad',
            title=f"Top 15 Ciudades - {departamento_seleccionado}",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mapa de calor por ciudad y causa
        top_10_ciudades = df_geo['CIUDAD'].value_counts().head(10).index
        df_top_ciudades = df_geo[df_geo['CIUDAD'].isin(top_10_ciudades)]

        heatmap_data = pd.crosstab(
            df_top_ciudades['CIUDAD'],
            df_top_ciudades['Causa Siniestro']
        ).loc[top_10_ciudades]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, '#0d1b2a'], [0.5, '#004d80'], [1, '#00c2ff']],
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 10, "color": "#e8edf3"}
        ))
        fig.update_layout(
            **_LAYOUT_BASE,
            title='Causas de Siniestros por Ciudad (Top 10)',
            xaxis_title='Causa',
            yaxis_title='Ciudad',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla detallada por ciudad
    st.markdown("---")
    st.markdown("### 📋 Análisis Detallado por Ciudad")
    
    ciudad_detail = df_geo.groupby('CIUDAD').agg({
        'DEPARTAMENTO': 'first',
        'Fecha Siniestro': 'count',
        'Tiempo Inoperativo ( HORAS )': ['mean', 'max'],
        'Culpabilidad Conductor': lambda x: (x == 'SI').sum(),
        'Reportado Aseguradora': lambda x: (x == 'SI').sum()
    }).round(2)
    
    ciudad_detail.columns = ['Departamento', 'Total', 'Tiempo Prom (h)', 'Tiempo Max (h)', 
                             'Conductor Culpable', 'Reportados']
    ciudad_detail = ciudad_detail.sort_values('Total', ascending=False).head(20)
    
    st.dataframe(ciudad_detail, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS DE CAUSAS
# ============================================================================

elif st.session_state.analisis_causas:
    st.markdown("<h1>🔍 Análisis de Causas de Siniestros</h1>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Distribución de Causas")
    
    # KPIs de causas
    col1, col2, col3 = st.columns(3)
    
    causas_count = df['Causa Siniestro'].value_counts()
    
    with col1:
        fallas_humanas = (df['Causa Siniestro'] == 'FALLAS HUMANAS').sum()
        pct_fh = fallas_humanas / len(df) * 100
        st.metric("Fallas Humanas", f"{fallas_humanas:,}", f"{pct_fh:.1f}%")
    
    with col2:
        factores_externos = (df['Causa Siniestro'] == 'FACTORES EXTERNOS').sum()
        pct_fe = factores_externos / len(df) * 100
        st.metric("Factores Externos", f"{factores_externos:,}", f"{pct_fe:.1f}%")
    
    with col3:
        fallas_mtto = (df['Causa Siniestro'] == 'FALLAS MTTO').sum()
        pct_fm = fallas_mtto / len(df) * 100
        st.metric("Fallas de Mantenimiento", f"{fallas_mtto:,}", f"{pct_fm:.1f}%")
    
    # Gráficos de causas
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución general
        fig = crear_grafico_pie(
            causas_count.index,
            causas_count.values,
            "Distribución de Causas de Siniestros"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Causas por año
        causas_año = df.groupby(['Año', 'Causa Siniestro']).size().reset_index(name='Cantidad')
        fig = px.bar(
            causas_año,
            x='Año',
            y='Cantidad',
            color='Causa Siniestro',
            title='Evolución de Causas por Año',
            barmode='stack',
            color_discrete_sequence=_COLORS
        )
        fig.update_layout(**_LAYOUT_BASE, height=500)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis detallado por causa
    st.markdown("---")
    st.markdown("### 🔎 Análisis Detallado por Causa")
    
    causa_analisis = st.selectbox(
        "Selecciona una Causa para Análisis Detallado",
        options=df['Causa Siniestro'].unique()
    )
    
    df_causa = df[df['Causa Siniestro'] == causa_analisis]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Casos", f"{len(df_causa):,}")
    
    with col2:
        tiempo_prom = df_causa['Tiempo Inoperativo ( HORAS )'].mean()
        st.metric("Tiempo Prom. (h)", f"{tiempo_prom:.1f}")
    
    with col3:
        pct_culpa = (df_causa['Culpabilidad Conductor'] == 'SI').sum() / len(df_causa) * 100
        st.metric("% Culpabilidad", f"{pct_culpa:.1f}%")
    
    with col4:
        pct_grua = (df_causa['Usó Grua'] == 'SI').sum() / len(df_causa) * 100
        st.metric("% Uso Grúa", f"{pct_grua:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tipos de accidente para esta causa
        tipo_acc = df_causa['Tipo accidente'].value_counts().head(8)
        fig = crear_grafico_barras(
            pd.DataFrame({'Tipo': tipo_acc.index, 'Cantidad': tipo_acc.values}),
            x='Cantidad',
            y='Tipo',
            title=f"Tipos de Accidente - {causa_analisis}",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Turno para esta causa
        turno_causa = df_causa['Turno'].value_counts()
        fig = crear_grafico_pie(
            turno_causa.index,
            turno_causa.values,
            f"Distribución por Turno - {causa_analisis}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS DE VEHÍCULOS
# ============================================================================
elif st.session_state.analisis_vehiculos:
    st.markdown("<h1>🚗 Análisis de Vehículos</h1>", unsafe_allow_html=True)
    st.markdown("### 📊 Resumen por Tipo de Vehículo")

    vehiculo_stats = df.groupby('Tipo de vehículo').agg({
        'CIUDAD': 'count',
        'Tiempo Inoperativo ( HORAS )': ['mean', 'median'],
        'Culpabilidad Conductor': lambda x: (x == 'SI').sum() / len(x) * 100,
        'Usó Grua': lambda x: (x == 'SI').sum() / len(x) * 100,
        'Reportado Aseguradora': lambda x: (x == 'SI').sum() / len(x) * 100
    }).round(2)
    vehiculo_stats.columns = ['Total', 'Tiempo_Prom', 'Tiempo_Mediana', 'Pct_Culpabilidad', 'Pct_Grua', 'Pct_Reportado']
    vehiculo_stats = vehiculo_stats.sort_values('Total', ascending=False)
    st.dataframe(vehiculo_stats.head(20), use_container_width=True)

    st.markdown("### 📈 Gráficos de Vehículos")
    top_8_vehiculos = vehiculo_stats.head(8)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Siniestros', 'Tiempo Promedio Inoperativo', '% Culpabilidad', '% Uso de Grúa'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )

    fig.add_trace(
        go.Bar(x=top_8_vehiculos.index, y=top_8_vehiculos['Total'],
               name='Total', marker_color='#00c2ff', marker_line_width=0),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=top_8_vehiculos.index, y=top_8_vehiculos['Tiempo_Prom'],
               name='Tiempo Prom', marker_color='#ff6b35', marker_line_width=0),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=top_8_vehiculos.index, y=top_8_vehiculos['Pct_Culpabilidad'],
               name='% Culpabilidad', marker_color='#00d296', marker_line_width=0),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=top_8_vehiculos.index, y=top_8_vehiculos['Pct_Grua'],
               name='% Grúa', marker_color='#a78bfa', marker_line_width=0),
        row=2, col=2
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=850,
        showlegend=False,
        title_text='Análisis Comparativo por Tipo de Vehículo (Top 8)'
    )
    fig.update_annotations(font_size=12, font_color='#8ba3bc',
                           font_family='Syne, sans-serif')
    fig.update_xaxes(gridcolor='rgba(0,194,255,0.07)',
                     linecolor='rgba(0,194,255,0.15)', tickfont=dict(size=10))
    fig.update_yaxes(gridcolor='rgba(0,194,255,0.07)',
                     linecolor='rgba(0,194,255,0.15)', tickfont=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS DE CONDUCTORES
# ============================================================================
elif st.session_state.analisis_conductores:
    st.markdown("<h1>👤 Análisis de Conductores</h1>", unsafe_allow_html=True)
    st.markdown("### 📊 Top Conductores por Siniestros")

    conductor_counts = df['Nombre Conductor'].value_counts().head(20)
    conductor_detail = df[df['Nombre Conductor'].isin(conductor_counts.index)].groupby('Nombre Conductor').agg({
        'Fecha Siniestro': 'count',
        'Culpabilidad Conductor': lambda x: (x == 'SI').sum(),
        'Tiempo Inoperativo ( HORAS )': 'sum',
        'Tipo de vehículo': lambda x: x.mode()[0] if not x.mode().empty else 'Varios'
    }).round(2)
    conductor_detail.columns = ['Total_Siniestros', 'Veces_Culpable', 'Tiempo_Total_Inop', 'Vehiculo_Usual']
    conductor_detail['Tasa_Culpabilidad_%'] = (conductor_detail['Veces_Culpable'] / conductor_detail['Total_Siniestros'] * 100).round(1)
    st.dataframe(conductor_detail.sort_values('Total_Siniestros', ascending=False), use_container_width=True)

    st.markdown("### 📈 Culpabilidad por Causa")
    culpa_causa = pd.crosstab(df['Causa Siniestro'], df['Culpabilidad Conductor'], normalize='index') * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=culpa_causa.index,
        y=culpa_causa.get('SI', pd.Series(0, index=culpa_causa.index)),
        name='SI',
        marker_color='#ff6b35',
        marker_line_width=0
    ))
    fig.update_layout(**_LAYOUT_BASE,
                      title='Porcentaje de Culpabilidad (SI) por Causa', height=450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECCIÓN: ANÁLISIS DE GRAVEDAD
# ============================================================================
elif st.session_state.analisis_gravedad:
    st.markdown("<h1>⚠️ Análisis de Gravedad</h1>", unsafe_allow_html=True)
    st.markdown("### 📊 Distribución por Gravedad")

    gravedad_counts = df['Gravedad'].value_counts().sort_index()
    gravedad_pct = (gravedad_counts / len(df)) * 100
    gravedad_df = pd.DataFrame({'Cantidad': gravedad_counts, 'Porcentaje': gravedad_pct.round(2)})
    st.dataframe(gravedad_df, use_container_width=True)

    fig = go.Figure(data=[go.Pie(
        labels=gravedad_counts.index,
        values=gravedad_counts.values,
        hole=0.42,
        marker=dict(colors=_COLORS, line=dict(color='rgba(13,27,42,0.8)', width=2)),
        textfont=dict(size=11, family="DM Sans, sans-serif")
    )])
    fig.update_layout(**_LAYOUT_BASE, title_text='Distribución por Nivel de Gravedad', height=450)
    st.plotly_chart(fig, use_container_width=True)

    casos_criticos = df[df['Gravedad'] == 'Crítico']
    st.markdown(f"### 🚨 Casos Críticos: {len(casos_criticos):,}")
    if len(casos_criticos) > 0:
        st.dataframe(casos_criticos[['Fecha Siniestro', 'DEPARTAMENTO', 'CIUDAD', 'Tipo de vehículo', 'Tipo accidente', 'Tiempo Inoperativo ( HORAS )', 'Causa Siniestro']].sort_values('Tiempo Inoperativo ( HORAS )', ascending=False).head(10), use_container_width=True)

# ============================================================================
# SECCIÓN: CONSULTA ESPECÍFICA
# ============================================================================
elif st.session_state.consulta_especifica:
    st.markdown("<h1>🔎 Consulta Específica</h1>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Filtrar datos con condiciones")

    dept_op = st.multiselect("Departamento", options=sorted(df['DEPARTAMENTO'].dropna().unique()), default=sorted(df['DEPARTAMENTO'].dropna().unique())[:5])
    ciudades_op = st.multiselect("Ciudad", options=sorted(df['CIUDAD'].dropna().unique()), default=sorted(df['CIUDAD'].dropna().unique())[:5])
    causas_op = st.multiselect("Causa", options=sorted(df['Causa Siniestro'].dropna().unique()), default=sorted(df['Causa Siniestro'].dropna().unique())[:3])
    turno_op = st.multiselect("Turno", options=sorted(df['Turno'].dropna().unique()), default=sorted(df['Turno'].dropna().unique()))

    df_consulta = df[
        (df['DEPARTAMENTO'].isin(dept_op)) &
        (df['CIUDAD'].isin(ciudades_op)) &
        (df['Causa Siniestro'].isin(causas_op)) &
        (df['Turno'].isin(turno_op))
    ]

    st.markdown(f"#### Resultados: {len(df_consulta):,} registros")
    st.dataframe(df_consulta.head(200), use_container_width=True)

# ============================================================================
# SECCIÓN: CONCLUSIONES
# ============================================================================
elif st.session_state.conclusiones:
    st.markdown("<h1>📝 Conclusiones y Recomendaciones</h1>", unsafe_allow_html=True)
 
    # ── Cálculos dinámicos para conclusiones ──
    dept_top       = df['DEPARTAMENTO'].value_counts().idxmax()
    dept_top_pct   = df['DEPARTAMENTO'].value_counts().iloc[0] / len(df) * 100
    causa_top      = df['Causa Siniestro'].value_counts().idxmax()
    causa_top_pct  = df['Causa Siniestro'].value_counts().iloc[0] / len(df) * 100
    causa_2nd      = df['Causa Siniestro'].value_counts().index[1] if len(df['Causa Siniestro'].value_counts()) > 1 else ''
    causa_2nd_pct  = df['Causa Siniestro'].value_counts().iloc[1] / len(df) * 100 if len(df['Causa Siniestro'].value_counts()) > 1 else 0
    turno_top      = df['Turno'].value_counts().idxmax()
    turno_pct      = df['Turno'].value_counts().iloc[0] / len(df) * 100
    pct_culp       = (df['Culpabilidad Conductor'] == 'SI').sum() / len(df) * 100
    tiempo_prom    = df['Tiempo Inoperativo ( HORAS )'].mean()
    tiempo_max     = df['Tiempo Inoperativo ( HORAS )'].max()
    pct_grua       = (df['Usó Grua'] == 'SI').sum() / len(df) * 100
    pct_reportado  = (df['Reportado Aseguradora'] == 'SI').sum() / len(df) * 100
    ciudad_top     = df['CIUDAD'].value_counts().idxmax()
    ciudad_top_n   = df['CIUDAD'].value_counts().iloc[0]
    vehiculo_top   = df['Tipo de vehículo'].value_counts().idxmax()
    vehiculo_pct   = df['Tipo de vehículo'].value_counts().iloc[0] / len(df) * 100
    criticos_n     = (df['Gravedad'] == 'Crítico').sum()
    criticos_pct   = criticos_n / len(df) * 100
    año_pico       = df.groupby('Año').size().idxmax()
    año_pico_n     = df.groupby('Año').size().max()
 
    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem;">
        <h3 style="margin-bottom:0.2rem;">ℹ️ Nota metodológica</h3>
        <p style="margin:0; font-size:0.88rem;">
            Las conclusiones y recomendaciones presentadas a continuación se derivan directamente
            del análisis estadístico del dataset de <strong style="color:#e8edf3;">{total:,} registros</strong>
            de siniestros viales de la operación de recolección de residuos.
            Los porcentajes y cifras son calculados en tiempo real sobre los datos cargados.
        </p>
    </div>
    """.format(total=len(df)), unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════
    # HALLAZGOS
    # ══════════════════════════════════════════════
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700;
                letter-spacing:2px; text-transform:uppercase; color:#00c2ff;
                background:rgba(0,194,255,0.08); border:1px solid rgba(0,194,255,0.2);
                border-radius:20px; padding:0.22rem 0.75rem; width:fit-content;
                margin-bottom:0.6rem;">
        🎯 Principales Hallazgos
    </div>
    """, unsafe_allow_html=True)
 
    hallazgos = [
        {
            "icono": "📍",
            "titulo": f"Concentración geográfica en {dept_top}",
            "texto": f"El departamento de <strong style='color:#e8edf3;'>{dept_top}</strong> concentra el "
                     f"<strong style='color:#00c2ff;'>{dept_top_pct:.1f}%</strong> del total de siniestros, "
                     f"siendo <strong style='color:#e8edf3;'>{ciudad_top}</strong> la ciudad con mayor número "
                     f"de incidentes registrados (<strong style='color:#00c2ff;'>{ciudad_top_n:,} casos</strong>). "
                     f"Esta concentración indica que las rutas operativas en esa zona presentan condiciones "
                     f"de riesgo superiores al promedio nacional."
        },
        {
            "icono": "⚠️",
            "titulo": "Causas dominantes: factores humanos y externos",
            "texto": f"La causa principal de siniestros es <strong style='color:#e8edf3;'>{causa_top}</strong> "
                     f"con el <strong style='color:#ff6b35;'>{causa_top_pct:.1f}%</strong> de los casos, "
                     f"seguida de <strong style='color:#e8edf3;'>{causa_2nd}</strong> con el "
                     f"<strong style='color:#ff6b35;'>{causa_2nd_pct:.1f}%</strong>. En conjunto, estas dos causas "
                     f"representan más del {causa_top_pct + causa_2nd_pct:.0f}% de toda la accidentalidad, "
                     f"lo que señala que la mayoría de los eventos son <em>prevenibles</em> con intervención directa."
        },
        {
            "icono": "🕐",
            "titulo": f"El turno {turno_top} concentra la mayor accidentalidad",
            "texto": f"El <strong style='color:#e8edf3;'>{turno_pct:.1f}%</strong> de los siniestros ocurre "
                     f"durante el turno <strong style='color:#00c2ff;'>{turno_top}</strong>. Esto puede estar "
                     f"relacionado con la mayor densidad de tráfico, la fatiga acumulada durante jornadas "
                     f"extensas o la mayor exposición operativa en ese horario. El patrón sugiere la necesidad "
                     f"de reforzar protocolos de seguridad específicos para ese periodo."
        },
        {
            "icono": "👤",
            "titulo": "Alta responsabilidad directa del conductor",
            "texto": f"En el <strong style='color:#e8edf3;'>{pct_culp:.1f}%</strong> de los siniestros se "
                     f"determinó culpabilidad del conductor. Este indicador, combinado con el hecho de que "
                     f"el <strong style='color:#ff6b35;'>{pct_grua:.1f}%</strong> de los incidentes requirió "
                     f"asistencia de grúa, sugiere que una proporción significativa de los eventos implican "
                     f"pérdida de control o maniobras incorrectas con consecuencias materiales relevantes."
        },
        {
            "icono": "⏱️",
            "titulo": "Impacto operativo: tiempo inoperativo elevado",
            "texto": f"El tiempo inoperativo promedio por siniestro es de "
                     f"<strong style='color:#a78bfa;'>{tiempo_prom:.1f} horas</strong>, con un máximo registrado "
                     f"de <strong style='color:#a78bfa;'>{tiempo_max:.0f} horas</strong>. Los "
                     f"<strong style='color:#e8edf3;'>{criticos_n:,} casos críticos</strong> "
                     f"(<strong style='color:#ff6b35;'>{criticos_pct:.1f}%</strong> del total) generan "
                     f"las mayores interrupciones al servicio de recolección, con un impacto directo "
                     f"en los niveles de servicio y en los costos de operación."
        },
        {
            "icono": "📋",
            "titulo": "Subregistro ante aseguradoras: riesgo financiero latente",
            "texto": f"Solo el <strong style='color:#e8edf3;'>{pct_reportado:.1f}%</strong> de los siniestros "
                     f"fue reportado a la aseguradora. El <strong style='color:#ff6b35;'>{100 - pct_reportado:.1f}%</strong> "
                     f"restante representa eventos que la empresa asumió sin respaldo asegurador, lo que implica "
                     f"un riesgo financiero considerable y posibles incumplimientos de las pólizas vigentes. "
                     f"Formalizar el proceso de reporte es una oportunidad inmediata de reducción de costos."
        },
        {
            "icono": "🚛",
            "titulo": f"El tipo de vehículo {vehiculo_top} lidera los incidentes",
            "texto": f"El vehículo tipo <strong style='color:#e8edf3;'>{vehiculo_top}</strong> protagoniza el "
                     f"<strong style='color:#00c2ff;'>{vehiculo_pct:.1f}%</strong> de los siniestros registrados. "
                     f"El año con mayor accidentalidad fue <strong style='color:#e8edf3;'>{año_pico}</strong> "
                     f"con <strong style='color:#ff6b35;'>{año_pico_n:,} eventos</strong>, lo que indica "
                     f"que existe variabilidad interanual significativa posiblemente ligada a la incorporación "
                     f"de nueva flota, cambios de ruta o variaciones en las condiciones operativas."
        },
    ]
 
    for h in hallazgos:
        st.markdown(f"""
        <div style="background:rgba(22,35,51,0.9); border:1px solid rgba(0,194,255,0.12);
                    border-left:3px solid rgba(0,194,255,0.5);
                    border-radius:12px; padding:1.1rem 1.4rem; margin-bottom:0.85rem;
                    box-shadow:0 4px 16px rgba(0,0,0,0.2);">
            <div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.35rem;">
                <span style="font-size:1.1rem;">{h['icono']}</span>
                <span style="font-family:'Syne',sans-serif; font-weight:700;
                             font-size:0.96rem; color:#e8edf3;">{h['titulo']}</span>
            </div>
            <p style="margin:0; color:#8ba3bc; font-size:0.88rem; line-height:1.75;">
                {h['texto']}
            </p>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════
    # RECOMENDACIONES
    # ══════════════════════════════════════════════
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700;
                letter-spacing:2px; text-transform:uppercase; color:#ff6b35;
                background:rgba(255,107,53,0.08); border:1px solid rgba(255,107,53,0.2);
                border-radius:20px; padding:0.22rem 0.75rem; width:fit-content;
                margin-bottom:0.6rem;">
        💡 Recomendaciones Estratégicas
    </div>
    """, unsafe_allow_html=True)
 
    recomendaciones = [
        ("🎓", "Capacitación focalizada en zonas y turnos críticos",
         f"Diseñar programas de formación en seguridad vial dirigidos específicamente a conductores "
         f"que operen en {dept_top} y en el turno {turno_top}, priorizando las causas "
         f"{causa_top} y {causa_2nd} como ejes temáticos centrales."),
        ("🔧", "Plan de mantenimiento preventivo diferenciado por tipo de vehículo",
         f"Intensificar los ciclos de mantenimiento para la flota tipo {vehiculo_top}, "
         f"dado su mayor participación en siniestros. Implementar alertas tempranas de falla mecánica "
         f"y registrar el historial de mantenimiento vinculado a cada evento de accidentalidad."),
        ("📡", "Monitoreo en tiempo real y alertas de comportamiento",
         f"Implementar soluciones de telemetría vehicular (GPS + sensores de conducción) que permitan "
         f"detectar eventos de riesgo como frenadas bruscas, exceso de velocidad o desvíos de ruta. "
         f"Esto reduciría directamente los incidentes por causa humana, actualmente el {pct_culp:.0f}% del total."),
        ("📄", "Estandarizar el reporte a aseguradoras",
         f"Establecer un protocolo obligatorio de reporte para el {100-pct_reportado:.0f}% de siniestros "
         f"que hoy no se registran ante la aseguradora. Crear un flujo digital de notificación inmediata "
         f"desde el conductor hacia el área de seguros, con seguimiento automático del estado del caso."),
        ("📊", "Dashboard de KPIs de seguridad vial en tiempo real",
         f"Centralizar los indicadores clave (tasa de siniestros por ruta, tiempo inoperativo acumulado, "
         f"índice de culpabilidad por conductor) en un panel directivo actualizado semanalmente, "
         f"para tomar decisiones correctivas antes de que los patrones escalen."),
        ("🗺️", "Rediseño de rutas en zonas de alta siniestralidad",
         f"Con base en el análisis geográfico, revisar los trazados de ruta en {ciudad_top} y otras "
         f"ciudades de alto riesgo. Evaluar cambios de horario, reducción de velocidad operativa "
         f"y coordinación con autoridades locales para mejorar las condiciones de vía en los "
         f"tramos con mayor concentración de incidentes."),
    ]
 
    cols = st.columns(2)
    for i, (icono, titulo, texto) in enumerate(recomendaciones):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:rgba(22,35,51,0.9); border:1px solid rgba(255,107,53,0.15);
                        border-left:3px solid rgba(255,107,53,0.5);
                        border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:0.85rem;
                        box-shadow:0 4px 16px rgba(0,0,0,0.2); height:100%;">
                <div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.35rem;">
                    <span style="font-size:1.05rem;">{icono}</span>
                    <span style="font-family:'Syne',sans-serif; font-weight:700;
                                 font-size:0.92rem; color:#ff6b35;">{titulo}</span>
                </div>
                <p style="margin:0; color:#8ba3bc; font-size:0.87rem; line-height:1.75;">
                    {texto}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <strong>🚛 SiniestrosApp — Análisis de Siniestros Nacionales</strong><br>
    <span>Desarrollado con Streamlit · Pandas · Plotly · NumPy &nbsp;|&nbsp; © 2025</span><br>
    <span style="font-size:0.78rem; opacity:0.6;">Datos actualizados hasta: {}</span>
</div>
""".format(df['Fecha Siniestro'].max().strftime('%d de %B de %Y')), unsafe_allow_html=True)

# ============================================================================
# BARRA LATERAL - NAVEGACIÓN
# ============================================================================

with st.sidebar:

    # ── Brand header ──
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2rem; line-height:1;">🚛</div>
        <div>
            <div class="sidebar-brand-title">SiniestrosApp</div>
            <div class="sidebar-brand-sub">Panel de Análisis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Menú principal ──
    st.markdown('<div class="nav-label">Principal</div>', unsafe_allow_html=True)

    if st.button("🏠  Inicio", use_container_width=True):
        cambiar_vista('inicio')
        st.rerun()

    if st.button("📊  Resumen General", use_container_width=True):
        cambiar_vista('resumen_general')
        st.rerun()

    # ── Análisis ──
    st.markdown('<div class="nav-label">Análisis</div>', unsafe_allow_html=True)

    if st.button("📅  Análisis Temporal", use_container_width=True):
        cambiar_vista('analisis_temporal')
        st.rerun()

    if st.button("🗺️  Análisis Geográfico", use_container_width=True):
        cambiar_vista('analisis_geografico')
        st.rerun()

    if st.button("🔍  Causas de Siniestros", use_container_width=True):
        cambiar_vista('analisis_causas')
        st.rerun()

    if st.button("🚗  Vehículos", use_container_width=True):
        cambiar_vista('analisis_vehiculos')
        st.rerun()

    if st.button("👤  Conductores", use_container_width=True):
        cambiar_vista('analisis_conductores')
        st.rerun()

    if st.button("⚠️  Gravedad", use_container_width=True):
        cambiar_vista('analisis_gravedad')
        st.rerun()

    # ── Herramientas ──
    st.markdown('<div class="nav-label">Herramientas</div>', unsafe_allow_html=True)

    if st.button("🔎  Consulta Específica", use_container_width=True):
        cambiar_vista('consulta_especifica')
        st.rerun()

    if st.button("📝  Conclusiones", use_container_width=True):
        cambiar_vista('conclusiones')
        st.rerun()

    # ── Stats card ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-card-row">
            <span class="stats-card-label">Total registros</span>
            <span class="stats-card-value">{len(df):,}</span>
        </div>
        <div class="stats-card-row">
            <span class="stats-card-label">Periodo</span>
            <span class="stats-card-value">{df['Fecha Siniestro'].min().strftime('%Y')} – {df['Fecha Siniestro'].max().strftime('%Y')}</span>
        </div>
        <div class="stats-card-row">
            <span class="stats-card-label">Departamentos</span>
            <span class="stats-card-value">{df['DEPARTAMENTO'].nunique()}</span>
        </div>
        <div class="stats-card-row">
            <span class="stats-card-label">Ciudades</span>
            <span class="stats-card-value">{df['CIUDAD'].nunique()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)