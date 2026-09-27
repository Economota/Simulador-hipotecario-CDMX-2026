# -*- coding: utf-8 -*-
"""
Modelo Dinámico de Costo de Oportunidad Inmobiliario — Rentar vs. Comprar
Calibrado para Ciudad de México, 2026.

Autor: Isaac Ortega Mota
Stack: Streamlit + Plotly + Pandas + NumPy (100% Python)

Cómo ejecutar:
    1. pip install -r requirements.txt
    2. streamlit run app_rentar_vs_comprar.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. CONFIGURACIÓN GENERAL Y ESTILOS
# ============================================================================

st.set_page_config(
    page_title="Rentar vs. Comprar | CDMX 2026",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Fuerza un tema claro consistente, sin importar el modo oscuro del
       dispositivo/navegador del visitante. Evita texto oscuro sobre fondo
       oscuro cuando el sistema operativo o el navegador están en "night mode". */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: var(--slate);
    }

    /* Paleta */
    :root {
        --navy: #0F172A;
        --slate: #475569;
        --accent-buy: #2563EB;
        --accent-rent: #059669;
        --bg-card: #F8FAFC;
        --border: #E2E8F0;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1 {
        font-weight: 800 !important;
        color: var(--navy) !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-weight: 700 !important;
        color: var(--navy) !important;
        letter-spacing: -0.01em;
    }
    p, li, label {
        color: var(--slate);
    }

    .subtitle {
        font-size: 1.05rem;
        color: var(--slate);
        margin-top: -0.6rem;
        margin-bottom: 1.8rem;
    }

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: var(--slate);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 1.0rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--slate) !important;
        margin-top: 1.4rem;
    }

    /* --- Sección de veredicto de alto impacto (hero) --- */
    .veredicto-hero {
        background: #0B0F19;
        border-radius: 20px;
        padding: 2.6rem 2rem 2.2rem 2rem;
        margin: 1.1rem 0 2rem 0;
        text-align: center;
    }
    .veredicto-label {
        color: #94A3B8;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .veredicto-ganador {
        font-size: clamp(2.8rem, 10vw, 4.6rem);
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.02em;
        margin: 0.3rem 0 1rem 0;
    }
    .veredicto-monto {
        color: #E2E8F0;
        font-size: 1.15rem;
        font-weight: 500;
        max-width: 30rem;
        margin: 0 auto;
    }
    .veredicto-monto b { color: #FFFFFF; }
    .veredicto-sub {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 1.3rem;
        padding-top: 1rem;
        border-top: 1px solid #1E293B;
    }

    .footnote {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 2rem;
    }

    hr { border-color: var(--border); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# 2. SUPUESTOS FIJOS DEL MODELO (Realidad CDMX 2026)
# ============================================================================
# Estos valores están hard-codeados a propósito: representan fricciones de
# mercado ampliamente documentadas para operaciones inmobiliarias en CDMX,
# y no deben ser controlados libremente por el usuario para mantener el
# rigor del ejercicio comparativo.

NOTARIAL_ISAI_AVALUO_PCT = 0.07      # % del valor de la propiedad (costo hundido, Mundo A)
COMISION_APERTURA_PCT    = 0.01      # % del monto del crédito (costo hundido, Mundo A)
MANTENIMIENTO_PREDIAL_SEGURO_PCT_MENSUAL = 0.0015  # % mensual del valor de la propiedad (Mundo A)
MESES_DEPOSITO_RENTA     = 1.5       # meses de renta: depósito + 50% póliza jurídica (Mundo B)


# ============================================================================
# 3. MOTOR DE CÁLCULO
# ============================================================================

@st.cache_data(show_spinner=False)
def calcular_modelo(
    valor_propiedad: float,
    enganche_pct: float,
    tasa_hipoteca_anual: float,
    renta_mensual_inicial: float,
    rendimiento_portafolio_anual: float,
    plusvalia_anual: float,
    horizonte_anios: int,
) -> pd.DataFrame:
    """
    Simula mes a mes ambos mundos paralelos (Comprar vs. Rentar) bajo
    ceteris paribus, y devuelve un DataFrame con la trayectoria completa.
    """

    V0 = valor_propiedad
    e = enganche_pct
    n_meses = horizonte_anios * 12

    # --- Montos iniciales, Mundo A (Comprar) ---
    enganche = V0 * e
    L0 = V0 - enganche
    costo_notarial_isai = V0 * NOTARIAL_ISAI_AVALUO_PCT
    comision_apertura = L0 * COMISION_APERTURA_PCT
    costo_inicial_A = enganche + costo_notarial_isai + comision_apertura

    # --- Tasa hipotecaria: convención nominal anual / 12 (estándar bancario MX) ---
    i = tasa_hipoteca_anual / 12
    if i > 0:
        pago_mensual = L0 * (i * (1 + i) ** n_meses) / ((1 + i) ** n_meses - 1)
    else:
        pago_mensual = L0 / n_meses

    # --- Tasas mensuales efectivas (plusvalía y rendimiento del portafolio) ---
    tasa_plusvalia_m = (1 + plusvalia_anual) ** (1 / 12) - 1
    tasa_rendimiento_m = (1 + rendimiento_portafolio_anual) ** (1 / 12) - 1

    # --- Montos iniciales, Mundo B (Rentar) ---
    gasto_inicial_renta = renta_mensual_inicial * MESES_DEPOSITO_RENTA
    capital_inicial_B = costo_inicial_A - gasto_inicial_renta

    # --- Estado inicial (mes 0) ---
    saldo = L0
    valor_inmueble = V0
    renta = renta_mensual_inicial
    portafolio = capital_inicial_B

    filas = [{
        "mes": 0,
        "anio": 0.0,
        "valor_inmueble": valor_inmueble,
        "saldo_insoluto": saldo,
        "pago_hipoteca": 0.0,
        "interes": 0.0,
        "amortizacion": 0.0,
        "gasto_mantenimiento": 0.0,
        "gasto_total_A": 0.0,
        "renta": renta,
        "diferencial_ahorro": 0.0,
        "patrimonio_comprador": valor_inmueble - saldo,
        "patrimonio_arrendatario": portafolio,
    }]

    # --- Simulación mes a mes ---
    for t in range(1, n_meses + 1):
        interes = saldo * i
        amortizacion = pago_mensual - interes
        saldo = max(saldo - amortizacion, 0.0)

        valor_inmueble = V0 * (1 + tasa_plusvalia_m) ** t
        gasto_mantenimiento = MANTENIMIENTO_PREDIAL_SEGURO_PCT_MENSUAL * valor_inmueble
        gasto_total_A = pago_mensual + gasto_mantenimiento

        renta = renta_mensual_inicial * (1 + tasa_plusvalia_m) ** t

        diferencial = gasto_total_A - renta
        portafolio = portafolio * (1 + tasa_rendimiento_m) + diferencial

        patrimonio_comprador = valor_inmueble - saldo
        patrimonio_arrendatario = portafolio

        filas.append({
            "mes": t,
            "anio": t / 12,
            "valor_inmueble": valor_inmueble,
            "saldo_insoluto": saldo,
            "pago_hipoteca": pago_mensual,
            "interes": interes,
            "amortizacion": amortizacion,
            "gasto_mantenimiento": gasto_mantenimiento,
            "gasto_total_A": gasto_total_A,
            "renta": renta,
            "diferencial_ahorro": diferencial,
            "patrimonio_comprador": patrimonio_comprador,
            "patrimonio_arrendatario": patrimonio_arrendatario,
        })

    df = pd.DataFrame(filas)
    df.attrs["enganche"] = enganche
    df.attrs["L0"] = L0
    df.attrs["costo_notarial_isai"] = costo_notarial_isai
    df.attrs["comision_apertura"] = comision_apertura
    df.attrs["costo_inicial_A"] = costo_inicial_A
    df.attrs["gasto_inicial_renta"] = gasto_inicial_renta
    df.attrs["capital_inicial_B"] = capital_inicial_B
    df.attrs["pago_mensual"] = pago_mensual
    return df


def encontrar_punto_equilibrio(df: pd.DataFrame):
    """
    Encuentra el mes (interpolado linealmente) donde el patrimonio neto
    del comprador y del arrendatario se igualan (cruce de curvas).
    Devuelve (anio_cruce, patrimonio_en_cruce) o (None, None) si no hay cruce.
    """
    diff = (df["patrimonio_comprador"] - df["patrimonio_arrendatario"]).to_numpy()
    signos = np.sign(diff)
    cambios = np.where(np.diff(signos) != 0)[0]

    if len(cambios) == 0:
        return None, None

    idx = cambios[0]
    d0, d1 = diff[idx], diff[idx + 1]
    if d1 == d0:
        frac = 0.0
    else:
        frac = d0 / (d0 - d1)

    mes_cruce = df["mes"].iloc[idx] + frac * (df["mes"].iloc[idx + 1] - df["mes"].iloc[idx])
    anio_cruce = mes_cruce / 12

    val0 = df["patrimonio_comprador"].iloc[idx]
    val1 = df["patrimonio_comprador"].iloc[idx + 1]
    valor_cruce = val0 + frac * (val1 - val0)

    return anio_cruce, valor_cruce


def fmt_mxn(valor: float) -> str:
    """Formatea un número como moneda MXN legible."""
    return f"${valor:,.0f}"


# ============================================================================
# 4. SIDEBAR — VARIABLES INTERACTIVAS
# ============================================================================

st.sidebar.markdown("## 🏙️ Parámetros del Modelo")
st.sidebar.caption("Ajusta las variables para simular tu escenario.")

st.sidebar.markdown("### Propiedad y Financiamiento")
valor_propiedad = st.sidebar.number_input(
    "Valor de la propiedad (MXN)",
    min_value=500_000, max_value=50_000_000,
    value=3_500_000, step=50_000, format="%d",
)
enganche_pct = st.sidebar.slider(
    "Enganche (%)", min_value=5, max_value=50, value=15, step=1,
) / 100
tasa_hipoteca_anual = st.sidebar.slider(
    "Tasa hipotecaria anual (%)", min_value=5.0, max_value=18.0, value=10.5, step=0.1,
) / 100

st.sidebar.markdown("### Mercado de Renta e Inversión")
renta_mensual_inicial = st.sidebar.number_input(
    "Renta mensual actual (MXN)",
    min_value=1_000, max_value=200_000, value=18_000, step=500,
)
rendimiento_portafolio_anual = st.sidebar.slider(
    "Rendimiento del portafolio / CETES (%)", min_value=3.0, max_value=15.0, value=6.5, step=0.1,
) / 100
plusvalia_anual = st.sidebar.slider(
    "Plusvalía anual inmobiliaria (%)", min_value=0.0, max_value=12.0, value=6.0, step=0.1,
) / 100

st.sidebar.markdown("### Horizonte")
horizonte_anios = st.sidebar.slider(
    "Horizonte de análisis (años)", min_value=5, max_value=30, value=20, step=1,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Supuestos fijos (no editables): notariales/ISAI/avalúo 7% · comisión de "
    "apertura 1% · seguros+predial+mantenimiento 0.15% mensual · gastos de "
    "arrendamiento 1.5 meses de renta. Ver pestaña Metodología."
)


# ============================================================================
# 5. EJECUCIÓN DEL MODELO
# ============================================================================

df = calcular_modelo(
    valor_propiedad, enganche_pct, tasa_hipoteca_anual,
    renta_mensual_inicial, rendimiento_portafolio_anual,
    plusvalia_anual, horizonte_anios,
)
anio_cruce, valor_cruce = encontrar_punto_equilibrio(df)

pat_final_comprador = df["patrimonio_comprador"].iloc[-1]
pat_final_arrendatario = df["patrimonio_arrendatario"].iloc[-1]
diferencia_final = pat_final_comprador - pat_final_arrendatario


# ============================================================================
# 6. ENCABEZADO
# ============================================================================

st.title("Rentar vs. Comprar: Costo de Oportunidad Inmobiliario")
st.markdown(
    '<p class="subtitle">Modelo dinámico calibrado para Ciudad de México · 2026</p>',
    unsafe_allow_html=True,
)

tab_dashboard, tab_metodologia = st.tabs(["📊 Dashboard", "📚 Metodología y Supuestos"])


# ============================================================================
# 7. TAB — DASHBOARD
# ============================================================================

with tab_dashboard:

    # --- Veredicto de alto impacto (hero) ---
    gana_comprar = diferencia_final > 0
    ganador_palabra = "COMPRAR" if gana_comprar else "RENTAR"
    accent = "#2563EB" if gana_comprar else "#059669"
    perdedor_palabra = "rentar e invertir" if gana_comprar else "comprar"

    if anio_cruce is not None and anio_cruce <= horizonte_anios:
        linea_secundaria = (
            f"El punto de cruce ocurre en el año {anio_cruce:.1f}: antes de eso, "
            f"convenía {perdedor_palabra}."
        )
    elif anio_cruce is not None:
        linea_secundaria = (
            f"{ganador_palabra.capitalize()} domina todo tu horizonte; el cruce con la otra "
            f"opción ocurriría hasta el año {anio_cruce:.1f}, fuera de tu plan."
        )
    else:
        linea_secundaria = (
            f"Con estos supuestos, {perdedor_palabra} no alcanza a {ganador_palabra.lower()} "
            f"dentro de los {horizonte_anios} años que planeas."
        )

    st.markdown(f"""
        <div class="veredicto-hero">
            <div class="veredicto-label">Si te quedas {horizonte_anios} años, te conviene:</div>
            <div class="veredicto-ganador" style="color:{accent};">{ganador_palabra}</div>
            <div class="veredicto-monto">
                Te deja <b>{fmt_mxn(abs(diferencia_final))}</b> más de patrimonio que la otra opción.
            </div>
            <div class="veredicto-sub">
                {linea_secundaria} Tu hipoteca sale en {fmt_mxn(df.attrs['pago_mensual'])}
                al mes (a {horizonte_anios} años).
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patrimonio neto final — Comprador", fmt_mxn(pat_final_comprador))
    c2.metric("Patrimonio neto final — Arrendatario", fmt_mxn(pat_final_arrendatario))
    c3.metric(
        "Diferencia final",
        fmt_mxn(abs(diferencia_final)),
        delta=("A favor de Comprar" if diferencia_final > 0 else "A favor de Rentar"),
    )
    c4.metric(
        "Punto de equilibrio",
        f"Año {anio_cruce:.1f}" if anio_cruce is not None else "Sin cruce",
    )

    st.markdown("### Trayectoria del Patrimonio Neto")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["anio"], y=df["patrimonio_comprador"],
        name="Comprador (equity inmueble)",
        line=dict(color="#2563EB", width=3.2),
        hovertemplate="Año %{x:.1f}<br>$%{y:,.0f} MXN<extra>Comprador</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["anio"], y=df["patrimonio_arrendatario"],
        name="Arrendatario (portafolio de inversión)",
        line=dict(color="#059669", width=3.2),
        hovertemplate="Año %{x:.1f}<br>$%{y:,.0f} MXN<extra>Arrendatario</extra>",
    ))

    if anio_cruce is not None:
        fig.add_vline(
            x=anio_cruce, line_width=1.5, line_dash="dash", line_color="#94A3B8",
        )
        fig.add_trace(go.Scatter(
            x=[anio_cruce], y=[valor_cruce],
            mode="markers+text",
            marker=dict(size=11, color="#0F172A", symbol="diamond"),
            text=[f"  Punto de equilibrio · Año {anio_cruce:.1f}"],
            textposition="top right",
            textfont=dict(color="#0F172A", size=12),
            name="Punto de equilibrio",
            showlegend=False,
            hovertemplate=f"Cruce en el año {anio_cruce:.1f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Años desde la decisión",
        yaxis_title="Patrimonio neto (MXN)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Gráfico secundario: flujo de efectivo mensual ---
    st.markdown("### Flujo de Efectivo Mensual Comparado")
    st.caption(
        "Gasto total mensual del comprador (hipoteca + mantenimiento/predial/seguros) "
        "vs. renta mensual. El área entre ambas líneas es exactamente lo que el "
        "arrendatario invierte (o desinvierte) cada mes."
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["anio"], y=df["gasto_total_A"],
        name="Gasto total mensual — Comprador",
        line=dict(color="#2563EB", width=2),
        hovertemplate="Año %{x:.1f}<br>$%{y:,.0f}<extra>Gasto Comprador</extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=df["anio"], y=df["renta"],
        name="Renta mensual — Arrendatario",
        line=dict(color="#059669", width=2),
        fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="Año %{x:.1f}<br>$%{y:,.0f}<extra>Renta</extra>",
    ))
    fig2.update_layout(
        template="plotly_white",
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Años desde la decisión",
        yaxis_title="Gasto mensual (MXN)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    fig2.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Desglose de costos iniciales ---
    with st.expander("Ver desglose de costos iniciales (Mundo A vs. Mundo B)"):
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Mundo A — Comprar (costos hundidos)**")
            st.write(f"Enganche: {fmt_mxn(df.attrs['enganche'])}")
            st.write(f"Notariales, avalúo, ISAI (7%): {fmt_mxn(df.attrs['costo_notarial_isai'])}")
            st.write(f"Comisión de apertura (1% crédito): {fmt_mxn(df.attrs['comision_apertura'])}")
            st.write(f"**Total desembolso inicial: {fmt_mxn(df.attrs['costo_inicial_A'])}**")
            st.write(f"Pago mensual hipotecario fijo: {fmt_mxn(df.attrs['pago_mensual'])}")
        with cB:
            st.markdown("**Mundo B — Rentar (capital a invertir)**")
            st.write(f"Capital equivalente disponible: {fmt_mxn(df.attrs['costo_inicial_A'])}")
            st.write(f"(–) Depósito + póliza jurídica (1.5 meses renta): {fmt_mxn(df.attrs['gasto_inicial_renta'])}")
            st.write(f"**Capital inicial invertido: {fmt_mxn(df.attrs['capital_inicial_B'])}**")

    st.markdown(
        '<p class="footnote">Modelo con fines educativos y de portafolio profesional. '
        'No constituye asesoría financiera. Los supuestos fijos y variables se detallan '
        'en la pestaña Metodología.</p>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 8. TAB — METODOLOGÍA
# ============================================================================

with tab_metodologia:
    st.markdown("## Marco Conceptual")

    st.markdown("""
Este modelo compara dos **mundos paralelos bajo el principio de *ceteris paribus***:
un individuo que compra una propiedad y otro, con la misma capacidad financiera
inicial, que renta un inmueble equivalente e invierte la diferencia. No se juzga
cuál decisión es moralmente superior — se calcula, mes a mes, cuál maximiza el
**patrimonio neto** del individuo bajo los supuestos definidos.
    """)

    st.markdown("### Costo de oportunidad")
    st.markdown("""
El enganche y los costos de cierre de la compra tienen un **costo de oportunidad**:
ese capital, si no se destina a la propiedad, podría invertirse en un portafolio
(CETES, fondos indexados, etc.) y generar rendimientos. El modelo captura esto
explícitamente: el arrendatario invierte el capital que el comprador hundió en
el enganche y los gastos notariales, menos el depósito de renta.
    """)

    st.markdown("### Costos hundidos (*sunk costs*)")
    st.markdown("""
Los gastos notariales, el ISAI, el avalúo y la comisión de apertura del crédito
son **costos hundidos**: se pagan una sola vez, al inicio, y no son recuperables
ni generan retorno directo. Se contabilizan como una salida de capital en el
Mundo A que reduce lo disponible para invertir en el Mundo B.
    """)

    st.markdown("### La disciplina del *Homo Economicus*")
    st.markdown("""
El modelo asume un arrendatario perfectamente racional y disciplinado: **invierte,
mes a mes, exactamente la diferencia** entre lo que le hubiera costado el gasto
total de ser comprador (hipoteca + mantenimiento + predial + seguros) y lo que
efectivamente paga de renta. No gasta ese excedente en consumo — lo capitaliza
íntegramente. Este es el supuesto más fuerte del modelo y vale la pena señalarlo
explícitamente: en la práctica, la disciplina de ahorro rara vez es perfecta.
    """)

    st.markdown("---")
    st.markdown("## Supuestos Fijos (Realidad CDMX 2026)")

    supuestos_df = pd.DataFrame({
        "Concepto": [
            "Notariales, avalúo e ISAI",
            "Comisión por apertura hipotecaria",
            "Seguros, predial y mantenimiento",
            "Gastos iniciales de arrendamiento",
        ],
        "Valor": ["7.0% del valor de la propiedad", "1.0% del monto del crédito",
                   "0.15% mensual sobre el valor de la propiedad",
                   "1.5 meses de renta (depósito + 50% póliza jurídica)"],
        "Aplica a": ["Mundo A (costo hundido inicial)", "Mundo A (costo hundido inicial)",
                      "Mundo A (costo hundido mensual)", "Mundo B (costo hundido inicial)"],
    })
    st.dataframe(supuestos_df, hide_index=True, use_container_width=True)

    st.markdown("### Supuestos adicionales del motor de cálculo")
    st.markdown("""
- El **plazo del crédito hipotecario coincide con el horizonte de análisis**
  seleccionado (p. ej., a 20 años de horizonte, el crédito también amortiza en 20 años).
- La **tasa hipotecaria** se aplica como nominal anual capitalizable mensualmente
  (`i = i_anual / 12`), la convención estándar de la banca mexicana.
- La **plusvalía inmobiliaria** y el **rendimiento del portafolio** se aplican como
  tasas efectivas anuales, convertidas a su equivalente mensual: `(1+i)^(1/12) − 1`.
- La **renta de mercado crece a la misma tasa que la plusvalía** del inmueble
  (supuesto de equilibrio de largo plazo entre precios y rentas).
- El diferencial de ahorro puede ser negativo en algunos escenarios (cuando la
  renta supera el gasto total del comprador); en ese caso, el modelo lo resta
  del portafolio del arrendatario, replicando una aportación negativa.
    """)

    st.markdown("---")
    st.markdown("## Fórmulas Clave")
    st.latex(r"L_0 = V_0 \times (1-e)")
    st.latex(r"M = L_0 \times \frac{i(1+i)^n}{(1+i)^n - 1}")
    st.latex(r"S(t) = S(t-1)(1+i) - M")
    st.latex(r"V(t) = V_0 (1+r_m)^t \qquad r_m = (1+g)^{1/12}-1")
    st.latex(r"\Delta(t) = \big[M + 0.0015\,V(t)\big] - \text{Renta}(t)")
    st.latex(r"P(t) = P(t-1)(1+k_m) + \Delta(t)")
    st.latex(r"PN_{\text{Comprador}}(t) = V(t) - S(t) \qquad PN_{\text{Arrendatario}}(t) = P(t)")

    st.markdown(
        '<p class="footnote">Modelo educativo desarrollado como proyecto de portafolio. '
        'Los resultados dependen enteramente de los supuestos ingresados y no deben '
        'usarse como única base para decisiones financieras reales.</p>',
        unsafe_allow_html=True,
    )
