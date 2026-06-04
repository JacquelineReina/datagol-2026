from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="DataGol 2026", page_icon="⚽", layout="wide")

KAGGLE_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/martj42/international-football-results-from-1872-to-2017"
GITHUB_FALLBACK_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MAX_GOLES = 8
FECHA_INICIO = "1998-01-01"
ANFITRIONES_2026 = {"Mexico", "United States", "Canada"}

# Calibración moderada para evitar que varios multiplicadores amplifiquen demasiado
# las diferencias entre selecciones.
PSEUDO_PARTIDOS = 24.0
PESO_FORMA_RECIENTE = 0.28
DIVISOR_ELO = 1800.0
BONO_ANFITRION = 1.03
MEZCLA_PROMEDIO_GENERAL = 0.24

NOMBRES_ES = {
    "Ivory Coast": "Costa de Marfil",
    "United States": "Estados Unidos",
    "South Africa": "Sudáfrica",
    "South Korea": "Corea del Sur",
    "North Korea": "Corea del Norte",
    "Netherlands": "Países Bajos",
    "England": "Inglaterra",
    "Wales": "Gales",
    "Scotland": "Escocia",
    "New Zealand": "Nueva Zelanda",
    "Saudi Arabia": "Arabia Saudita",
    "United Arab Emirates": "Emiratos Árabes Unidos",
    "Bosnia-Herzegovina": "Bosnia y Herzegovina",
    "Czech Republic": "República Checa",
    "Cape Verde": "Cabo Verde",
    "Trinidad and Tobago": "Trinidad y Tobago",
    "Republic of Ireland": "Irlanda",
    "Northern Ireland": "Irlanda del Norte",
    "DR Congo": "RD del Congo",
    "China PR": "China",
    "Iran": "Irán",
    "Japan": "Japón",
    "Spain": "España",
    "Germany": "Alemania",
    "France": "Francia",
    "Brazil": "Brasil",
    "Switzerland": "Suiza",
    "Belgium": "Bélgica",
    "Croatia": "Croacia",
    "Morocco": "Marruecos",
    "Tunisia": "Túnez",
    "Egypt": "Egipto",
    "Cameroon": "Camerún",
    "Mexico": "México",
    "Canada": "Canadá",
}

PESOS_TORNEO = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.92,
    "UEFA Euro": 0.84,
    "UEFA Euro qualification": 0.78,
    "Copa América": 0.84,
    "African Cup of Nations": 0.82,
    "African Cup of Nations qualification": 0.76,
    "Gold Cup": 0.78,
    "AFC Asian Cup": 0.80,
    "AFC Asian Cup qualification": 0.74,
    "UEFA Nations League": 0.68,
    "Friendly": 0.35,
}

def visible(equipo: str) -> str:
    return NOMBRES_ES.get(equipo, equipo)

def peso_torneo(nombre: str) -> float:
    if nombre in PESOS_TORNEO:
        return PESOS_TORNEO[nombre]
    nombre_l = str(nombre).lower()
    if "world cup" in nombre_l and "qualification" in nombre_l:
        return 0.92
    if "qualification" in nombre_l:
        return 0.72
    if "friendly" in nombre_l:
        return 0.35
    if "cup" in nombre_l or "league" in nombre_l:
        return 0.65
    return 0.55

@dataclass
class Contexto:
    fase: str = "Fase de grupos"
    anfitrion_a: bool = False
    anfitrion_b: bool = False
    descanso_a: int = 5
    descanso_b: int = 5
    ajuste_ataque_a: int = 0
    ajuste_ataque_b: int = 0
    ajuste_defensa_a: int = 0
    ajuste_defensa_b: int = 0

def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = {"date", "home_team", "away_team", "home_score", "away_score", "tournament"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    df = df.copy()
    if "neutral" not in df.columns:
        df["neutral"] = True

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].fillna(True).astype(bool)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
    df = df.loc[df["date"] >= FECHA_INICIO].copy()
    df["peso_torneo"] = df["tournament"].map(peso_torneo)

    claves = ["date", "home_team", "away_team", "home_score", "away_score", "tournament"]
    return df.drop_duplicates(subset=claves).sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=21600, show_spinner=False)
def cargar_base() -> tuple[pd.DataFrame, str]:
    try:
        respuesta = requests.get(KAGGLE_ZIP_URL, timeout=25)
        respuesta.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(respuesta.content)) as z:
            candidatos = [n for n in z.namelist() if n.lower().endswith("results.csv")]
            if not candidatos:
                candidatos = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not candidatos:
                raise ValueError("El ZIP no contiene CSV.")
            with z.open(candidatos[0]) as archivo:
                df = pd.read_csv(archivo)
        return normalizar(df), "Kaggle actualizado"
    except Exception:
        df = pd.read_csv(GITHUB_FALLBACK_URL)
        return normalizar(df), "GitHub de respaldo"

def combinar_suplemento(base: pd.DataFrame, archivo) -> tuple[pd.DataFrame, int]:
    if archivo is None:
        return base, 0
    extra = normalizar(pd.read_csv(archivo))
    antes = len(base)
    combinado = normalizar(pd.concat([base, extra], ignore_index=True))
    return combinado, max(0, len(combinado) - antes)

def formato_largo(df: pd.DataFrame) -> pd.DataFrame:
    local = df[["date", "home_team", "away_team", "home_score", "away_score", "peso_torneo"]].copy()
    local.columns = ["date", "equipo", "rival", "gf", "gc", "peso_torneo"]
    visita = df[["date", "away_team", "home_team", "away_score", "home_score", "peso_torneo"]].copy()
    visita.columns = ["date", "equipo", "rival", "gf", "gc", "peso_torneo"]
    return pd.concat([local, visita], ignore_index=True)

@st.cache_data(show_spinner=False)
def calcular_elos(df: pd.DataFrame) -> dict[str, float]:
    elo: dict[str, float] = {}
    for _, r in df.sort_values("date").iterrows():
        a, b = str(r["home_team"]), str(r["away_team"])
        ra, rb = elo.get(a, 1500.0), elo.get(b, 1500.0)
        ventaja = 0.0 if bool(r.get("neutral", True)) else 45.0
        ea = 1.0 / (1.0 + 10 ** ((rb - (ra + ventaja)) / 400.0))
        ga, gb = float(r["home_score"]), float(r["away_score"])
        if ga > gb:
            sa, sb = 1.0, 0.0
        elif ga < gb:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5
        dif = abs(ga - gb)
        mult = 1.0 if dif <= 1 else 1.0 + min(dif - 1, 3) * 0.12
        k = 28.0 * float(r["peso_torneo"]) * mult
        elo[a] = ra + k * (sa - ea)
        elo[b] = rb + k * (sb - (1.0 - ea))
    return elo

@st.cache_data(show_spinner=False)
def estadisticas(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    largo = formato_largo(df)
    fecha_ref = largo["date"].max()
    antig = (fecha_ref - largo["date"]).dt.days / 365.25
    largo["peso"] = np.exp(-np.log(2) * antig / 4.0) * largo["peso_torneo"]

    media = max(float(np.average(largo["gf"], weights=largo["peso"])), 0.05)
    filas = []

    for equipo, g in largo.groupby("equipo"):
        g = g.sort_values("date")
        pesos = g["peso"].to_numpy()
        peso_total = float(pesos.sum())

        gf_crudo = float(np.average(g["gf"], weights=pesos))
        gc_crudo = float(np.average(g["gc"], weights=pesos))

        # Suavizado empírico: evita extremos cuando el historial efectivo es limitado
        gf = (gf_crudo * peso_total + media * PSEUDO_PARTIDOS) / (peso_total + PSEUDO_PARTIDOS)
        gc = (gc_crudo * peso_total + media * PSEUDO_PARTIDOS) / (peso_total + PSEUDO_PARTIDOS)

        ult = g.tail(10)
        ult_gf = float(ult["gf"].mean())
        ult_gc = float(ult["gc"].mean())

        forma_atq_cruda = float(np.clip((ult_gf + 0.30) / (gf + 0.30), 0.84, 1.16))
        forma_def_cruda = float(np.clip((ult_gc + 0.30) / (gc + 0.30), 0.84, 1.16))

        # La forma reciente ajusta, pero no domina por completo el historial.
        forma_atq = 1.0 + PESO_FORMA_RECIENTE * (forma_atq_cruda - 1.0)
        forma_def = 1.0 + PESO_FORMA_RECIENTE * (forma_def_cruda - 1.0)

        filas.append({
            "equipo": equipo,
            "partidos": len(g),
            "peso_efectivo": peso_total,
            "gf": gf,
            "gc": gc,
            "ult_gf": ult_gf,
            "ult_gc": ult_gc,
            "victorias10": int((ult["gf"] > ult["gc"]).sum()),
            "ataque": gf / media,
            "def_debilidad": gc / media,
            "forma_atq": forma_atq,
            "forma_def": forma_def,
        })

    return media, pd.DataFrame(filas).set_index("equipo")

def pars(equipo: str, tabla: pd.DataFrame, elos: dict[str, float]) -> dict:
    if equipo in tabla.index:
        f = tabla.loc[equipo]
        return {
            "elo": float(elos.get(equipo, 1500)),
            "partidos": int(f["partidos"]),
            "gf": float(f["gf"]),
            "gc": float(f["gc"]),
            "ult_gf": float(f["ult_gf"]),
            "ult_gc": float(f["ult_gc"]),
            "victorias10": int(f["victorias10"]),
            "ataque": float(f["ataque"]),
            "def_debilidad": float(f["def_debilidad"]),
            "forma_atq": float(f["forma_atq"]),
            "forma_def": float(f["forma_def"]),
        }
    return {"elo": 1500.0, "partidos": 0, "gf": 1.25, "gc": 1.25, "ult_gf": 1.25, "ult_gc": 1.25,
            "victorias10": 0, "ataque": 1.0, "def_debilidad": 1.0, "forma_atq": 1.0, "forma_def": 1.0}

def ajustes_contexto(c: Contexto) -> tuple[float, float]:
    aa, ab = 1.0, 1.0
    if c.anfitrion_a: aa *= BONO_ANFITRION
    if c.anfitrion_b: ab *= BONO_ANFITRION
    dif = int(np.clip(c.descanso_a - c.descanso_b, -3, 3))
    aa *= 1 + 0.01 * dif
    ab *= 1 - 0.01 * dif
    if c.fase != "Fase de grupos":
        aa *= 0.98
        ab *= 0.98
    aa *= 1 + c.ajuste_ataque_a / 100
    ab *= 1 + c.ajuste_ataque_b / 100
    aa *= 1 + c.ajuste_defensa_b / 100
    ab *= 1 + c.ajuste_defensa_a / 100
    return max(0.75, aa), max(0.75, ab)

def predecir(equipo_a: str, equipo_b: str, df: pd.DataFrame, c: Contexto) -> dict:
    media, tabla = estadisticas(df)
    elos = calcular_elos(df)
    a, b = pars(equipo_a, tabla, elos), pars(equipo_b, tabla, elos)

    elo_a = float(np.clip(np.exp((a["elo"] - b["elo"]) / DIVISOR_ELO), 0.84, 1.18))
    elo_b = float(np.clip(np.exp((b["elo"] - a["elo"]) / DIVISOR_ELO), 0.84, 1.18))
    ctx_a, ctx_b = ajustes_contexto(c)

    la_cruda = media * a["ataque"] * b["def_debilidad"] * a["forma_atq"] * b["forma_def"] * elo_a * ctx_a
    lb_cruda = media * b["ataque"] * a["def_debilidad"] * b["forma_atq"] * a["forma_def"] * elo_b * ctx_b

    # Mezcla parcial con el promedio general: reduce sobreajuste y evita pronósticos extremos.
    la = (1 - MEZCLA_PROMEDIO_GENERAL) * la_cruda + MEZCLA_PROMEDIO_GENERAL * media
    lb = (1 - MEZCLA_PROMEDIO_GENERAL) * lb_cruda + MEZCLA_PROMEDIO_GENERAL * media

    la, lb = float(np.clip(la, 0.12, 4.2)), float(np.clip(lb, 0.12, 4.2))

    goles = np.arange(MAX_GOLES + 1)
    matriz = np.outer(poisson.pmf(goles, la), poisson.pmf(goles, lb))
    gana_a = float(np.tril(matriz, -1).sum())
    empate = float(np.trace(matriz))
    gana_b = float(np.triu(matriz, 1).sum())
    total = gana_a + empate + gana_b
    gana_a, empate, gana_b = [x / total * 100 for x in (gana_a, empate, gana_b)]

    escenarios = []
    for ga in range(MAX_GOLES + 1):
        for gb in range(MAX_GOLES + 1):
            escenarios.append((ga, gb, float(matriz[ga, gb]) * 100))
    escenarios = sorted(escenarios, key=lambda x: x[2], reverse=True)[:5]

    return {
        "a": a, "b": b, "media": media,
        "la_cruda": la_cruda, "lb_cruda": lb_cruda,
        "la": la, "lb": lb,
        "gana_a": gana_a, "empate": empate, "gana_b": gana_b,
        "escenarios": escenarios,
        "rango_a": (int(poisson.ppf(0.10, la)), int(poisson.ppf(0.90, la))),
        "rango_b": (int(poisson.ppf(0.10, lb)), int(poisson.ppf(0.90, lb))),
        "al_menos_un_gol_a": (1 - poisson.pmf(0, la)) * 100,
        "al_menos_un_gol_b": (1 - poisson.pmf(0, lb)) * 100,
        "mas_25": (1 - poisson.cdf(2, la + lb)) * 100,
    }

st.title("⚽ DataGol 2026")
st.subheader("Predicción calibrada con Poisson, Elo y forma reciente")
st.caption("Modelo académico probabilístico. No garantiza el marcador final.")

try:
    base, fuente = cargar_base()
except Exception as exc:
    st.error("No fue posible cargar la base histórica.")
    st.code(str(exc))
    st.stop()

with st.sidebar:
    st.header("Actualización de datos")
    suplemento = st.file_uploader("CSV opcional de partidos recientes", type=["csv"])

datos, agregados = combinar_suplemento(base, suplemento)
fecha_max = datos["date"].max().date()

with st.sidebar:
    st.success(f"Fuente: {fuente}")
    st.caption(f"Último partido disponible: {fecha_max}")
    st.caption(f"Partidos analizados: {len(datos):,}")
    if agregados:
        st.caption(f"Partidos añadidos desde CSV: {agregados}")

equipos = sorted(set(datos["home_team"]).union(datos["away_team"]))

with st.sidebar:
    st.header("Seleccione el partido")
    equipo_a = st.selectbox("Equipo A", equipos, index=equipos.index("Ecuador") if "Ecuador" in equipos else 0, format_func=visible)
    equipo_b = st.selectbox("Equipo B", equipos, index=equipos.index("Ivory Coast") if "Ivory Coast" in equipos else 1, format_func=visible)

    st.header("Contexto")
    fase = st.selectbox("Fase", ["Fase de grupos", "Dieciseisavos", "Octavos", "Cuartos", "Semifinal", "Final"])
    anfitrion_a = st.checkbox(f"{visible(equipo_a)} juega como anfitrión", value=equipo_a in ANFITRIONES_2026)
    anfitrion_b = st.checkbox(f"{visible(equipo_b)} juega como anfitrión", value=equipo_b in ANFITRIONES_2026)
    descanso_a = st.slider(f"Días de descanso: {visible(equipo_a)}", 2, 10, 5)
    descanso_b = st.slider(f"Días de descanso: {visible(equipo_b)}", 2, 10, 5)

    with st.expander("Ajustes manuales por lesiones o sanciones"):
        atq_a = st.slider(f"Ataque: {visible(equipo_a)}", -10, 10, 0, format="%d %%")
        def_a = st.slider(f"Debilidad defensiva adicional: {visible(equipo_a)}", -10, 10, 0, format="%d %%")
        atq_b = st.slider(f"Ataque: {visible(equipo_b)}", -10, 10, 0, format="%d %%")
        def_b = st.slider(f"Debilidad defensiva adicional: {visible(equipo_b)}", -10, 10, 0, format="%d %%")

if equipo_a == equipo_b:
    st.warning("Seleccione dos equipos diferentes.")
    st.stop()

contexto = Contexto(fase, anfitrion_a, anfitrion_b, descanso_a, descanso_b, atq_a, atq_b, def_a, def_b)
pred = predecir(equipo_a, equipo_b, datos, contexto)

a_es, b_es = visible(equipo_a), visible(equipo_b)
st.markdown(f"## {a_es} vs. {b_es}")
st.caption(f"Fase seleccionada: {fase}")

c1, c2, c3 = st.columns(3)
c1.metric(f"Victoria de {a_es}", f"{pred['gana_a']:.1f} %")
c2.metric("Empate", f"{pred['empate']:.1f} %")
c3.metric(f"Victoria de {b_es}", f"{pred['gana_b']:.1f} %")

c4, c5, c6 = st.columns(3)
c4.metric(f"Goles esperados: {a_es}", f"{pred['la']:.2f}")
c5.metric("Marcador modal", f"{pred['escenarios'][0][0]} - {pred['escenarios'][0][1]}")
c6.metric(f"Goles esperados: {b_es}", f"{pred['lb']:.2f}")

st.markdown("### Cinco marcadores exactos más probables")
top = pd.DataFrame([
    {"Posición": i + 1, "Marcador": f"{a_es} {ga} - {gb} {b_es}", "Probabilidad": f"{p:.1f} %"}
    for i, (ga, gb, p) in enumerate(pred["escenarios"])
])
st.dataframe(top, hide_index=True, use_container_width=True)
st.caption("El marcador modal es el escenario individual más probable; no es una certeza.")

st.markdown("### Indicadores adicionales")
ind = pd.DataFrame({
    "Indicador": [
        f"{a_es} marca al menos un gol",
        f"{b_es} marca al menos un gol",
        "Más de 2,5 goles en el partido",
        f"Rango probable de goles: {a_es}",
        f"Rango probable de goles: {b_es}",
    ],
    "Resultado": [
        f"{pred['al_menos_un_gol_a']:.1f} %",
        f"{pred['al_menos_un_gol_b']:.1f} %",
        f"{pred['mas_25']:.1f} %",
        f"{pred['rango_a'][0]} a {pred['rango_a'][1]}",
        f"{pred['rango_b'][0]} a {pred['rango_b'][1]}",
    ],
})
st.dataframe(ind, hide_index=True, use_container_width=True)

st.markdown("### Factores analizados")
factores = pd.DataFrame({
    "Indicador": ["Elo interno", "Partidos analizados", "Promedio GF suavizado", "Promedio GC suavizado", "GF últimos 10", "GC últimos 10", "Victorias últimos 10"],
    a_es: [f"{pred['a']['elo']:.0f}", pred["a"]["partidos"], f"{pred['a']['gf']:.2f}", f"{pred['a']['gc']:.2f}",
           f"{pred['a']['ult_gf']:.2f}", f"{pred['a']['ult_gc']:.2f}", pred["a"]["victorias10"]],
    b_es: [f"{pred['b']['elo']:.0f}", pred["b"]["partidos"], f"{pred['b']['gf']:.2f}", f"{pred['b']['gc']:.2f}",
           f"{pred['b']['ult_gf']:.2f}", f"{pred['b']['ult_gc']:.2f}", pred["b"]["victorias10"]],
})
st.dataframe(factores, hide_index=True, use_container_width=True)

with st.expander("¿Qué cambió en esta versión?"):
    st.markdown(
        """
        - Se suavizan ataque y defensa hacia el promedio general para reducir sobreajuste.
        - La forma reciente continúa influyendo, pero con un peso moderado.
        - El ajuste Elo se conserva, pero ya no amplifica excesivamente las diferencias.
        - La ventaja de anfitrión se mantiene con un efecto prudente.
        - Los goles esperados se mezclan parcialmente con el promedio general.

        El objetivo no es forzar coincidencia con otra página, sino producir estimaciones más estables.
        """
    )

st.divider()
st.caption(f"Datos disponibles hasta: {fecha_max}. Revise esta fecha antes de presentar una predicción.")
