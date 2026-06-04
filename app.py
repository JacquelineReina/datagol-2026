import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="DataGol 2026", page_icon="⚽", layout="wide")

URL_DATOS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MAX_GOLES = 8
FECHA_INICIO = "1998-01-01"

PESOS_TORNEO = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.90,
    "UEFA Euro": 0.80,
    "UEFA Euro qualification": 0.75,
    "Copa América": 0.80,
    "African Cup of Nations": 0.80,
    "African Cup of Nations qualification": 0.72,
    "CONCACAF Championship": 0.75,
    "Gold Cup": 0.75,
    "AFC Asian Cup": 0.78,
    "AFC Asian Cup qualification": 0.70,
    "Oceania Nations Cup": 0.70,
    "UEFA Nations League": 0.65,
    "Friendly": 0.35,
}


def peso_torneo(nombre: str) -> float:
    if nombre in PESOS_TORNEO:
        return PESOS_TORNEO[nombre]
    nombre_minusculas = str(nombre).lower()
    if "world cup" in nombre_minusculas and "qualification" in nombre_minusculas:
        return 0.90
    if "qualification" in nombre_minusculas:
        return 0.70
    if "friendly" in nombre_minusculas:
        return 0.35
    if "cup" in nombre_minusculas or "league" in nombre_minusculas:
        return 0.65
    return 0.55


@dataclass
class ContextoPartido:
    fase: str = "Fase de grupos"
    anfitrion_a: bool = False
    anfitrion_b: bool = False
    descanso_a: int = 5
    descanso_b: int = 5
    ajuste_ataque_a: int = 0
    ajuste_ataque_b: int = 0
    ajuste_defensa_a: int = 0
    ajuste_defensa_b: int = 0


@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(URL_DATOS)
    columnas = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    faltantes = columnas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = (
        df.loc[df["date"] >= FECHA_INICIO]
        .dropna(subset=["date", "home_team", "away_team", "home_score", "away_score", "tournament"])
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["peso_torneo"] = df["tournament"].map(peso_torneo)
    return df


def formato_largo(partidos: pd.DataFrame) -> pd.DataFrame:
    local = partidos[["date", "home_team", "away_team", "home_score", "away_score", "peso_torneo"]].copy()
    local.columns = ["date", "equipo", "rival", "goles_favor", "goles_contra", "peso_torneo"]
    visitante = partidos[["date", "away_team", "home_team", "away_score", "home_score", "peso_torneo"]].copy()
    visitante.columns = ["date", "equipo", "rival", "goles_favor", "goles_contra", "peso_torneo"]
    return pd.concat([local, visitante], ignore_index=True)


def calcular_elos(partidos: pd.DataFrame) -> dict:
    ratings: dict[str, float] = {}
    for _, fila in partidos.sort_values("date").iterrows():
        a, b = str(fila["home_team"]), str(fila["away_team"])
        goles_a, goles_b = float(fila["home_score"]), float(fila["away_score"])
        rating_a, rating_b = ratings.get(a, 1500.0), ratings.get(b, 1500.0)
        ventaja_local = 0.0 if bool(fila.get("neutral", False)) else 45.0
        esperado_a = 1.0 / (1.0 + 10 ** ((rating_b - (rating_a + ventaja_local)) / 400.0))
        esperado_b = 1.0 - esperado_a
        if goles_a > goles_b:
            real_a, real_b = 1.0, 0.0
        elif goles_a < goles_b:
            real_a, real_b = 0.0, 1.0
        else:
            real_a, real_b = 0.5, 0.5
        diferencia = abs(goles_a - goles_b)
        multiplicador = 1.0 if diferencia <= 1 else 1.0 + min(diferencia - 1, 3) * 0.12
        k = 28.0 * float(fila["peso_torneo"]) * multiplicador
        ratings[a] = rating_a + k * (real_a - esperado_a)
        ratings[b] = rating_b + k * (real_b - esperado_b)
    return ratings


def construir_estadisticas(partidos: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    largo = formato_largo(partidos)
    fecha_ref = largo["date"].max()
    antiguedad_anios = (fecha_ref - largo["date"]).dt.days / 365.25
    largo["peso_recencia"] = np.exp(-np.log(2) * antiguedad_anios / 4.0)
    largo["peso"] = largo["peso_recencia"] * largo["peso_torneo"]
    promedio_general = max(float(np.average(largo["goles_favor"], weights=largo["peso"])), 0.05)
    filas = []
    for equipo, grupo in largo.groupby("equipo"):
        grupo = grupo.sort_values("date")
        peso = grupo["peso"].to_numpy()
        promedio_gf = float(np.average(grupo["goles_favor"], weights=peso))
        promedio_gc = float(np.average(grupo["goles_contra"], weights=peso))
        recientes = grupo.tail(10)
        recientes_gf = float(recientes["goles_favor"].mean())
        recientes_gc = float(recientes["goles_contra"].mean())
        victorias = int((recientes["goles_favor"] > recientes["goles_contra"]).sum())
        forma_ataque = float(np.clip((recientes_gf + 0.25) / (promedio_gf + 0.25), 0.82, 1.18))
        forma_defensa = float(np.clip((recientes_gc + 0.25) / (promedio_gc + 0.25), 0.82, 1.18))
        filas.append({
            "equipo": equipo,
            "partidos": int(len(grupo)),
            "promedio_gf": promedio_gf,
            "promedio_gc": promedio_gc,
            "ataque": promedio_gf / promedio_general,
            "defensa_debilidad": promedio_gc / promedio_general,
            "ultimos_10_gf": recientes_gf,
            "ultimos_10_gc": recientes_gc,
            "victorias_ultimos_10": victorias,
            "forma_ataque": forma_ataque,
            "forma_defensa": forma_defensa,
        })
    return promedio_general, pd.DataFrame(filas).set_index("equipo")


def parametros_equipo(equipo: str, tabla: pd.DataFrame, elos: dict) -> dict:
    elo = float(elos.get(equipo, 1500.0))
    if equipo in tabla.index:
        fila = tabla.loc[equipo]
        return {
            "partidos": int(fila["partidos"]), "promedio_gf": float(fila["promedio_gf"]),
            "promedio_gc": float(fila["promedio_gc"]), "ataque": float(fila["ataque"]),
            "defensa_debilidad": float(fila["defensa_debilidad"]), "ultimos_10_gf": float(fila["ultimos_10_gf"]),
            "ultimos_10_gc": float(fila["ultimos_10_gc"]), "victorias_ultimos_10": int(fila["victorias_ultimos_10"]),
            "forma_ataque": float(fila["forma_ataque"]), "forma_defensa": float(fila["forma_defensa"]), "elo": elo,
        }
    return {"partidos": 0, "promedio_gf": 1.25, "promedio_gc": 1.25, "ataque": 1.0, "defensa_debilidad": 1.0,
            "ultimos_10_gf": 1.25, "ultimos_10_gc": 1.25, "victorias_ultimos_10": 0, "forma_ataque": 1.0,
            "forma_defensa": 1.0, "elo": elo}


def ajuste_contexto(ctx: ContextoPartido) -> tuple[float, float]:
    a, b = 1.0, 1.0
    if ctx.anfitrion_a:
        a *= 1.05
    if ctx.anfitrion_b:
        b *= 1.05
    diferencia_descanso = max(-3, min(3, ctx.descanso_a - ctx.descanso_b))
    a *= 1.0 + diferencia_descanso * 0.015
    b *= 1.0 - diferencia_descanso * 0.015
    if ctx.fase != "Fase de grupos":
        a *= 0.96
        b *= 0.96
    a *= 1.0 + ctx.ajuste_ataque_a / 100.0
    b *= 1.0 + ctx.ajuste_ataque_b / 100.0
    a *= 1.0 + ctx.ajuste_defensa_b / 100.0
    b *= 1.0 + ctx.ajuste_defensa_a / 100.0
    return max(0.70, a), max(0.70, b)


def predecir(equipo_a: str, equipo_b: str, partidos: pd.DataFrame, ctx: ContextoPartido | None = None) -> dict:
    ctx = ctx or ContextoPartido()
    promedio_general, tabla = construir_estadisticas(partidos)
    elos = calcular_elos(partidos)
    a, b = parametros_equipo(equipo_a, tabla, elos), parametros_equipo(equipo_b, tabla, elos)
    ajuste_elo_a = float(np.clip(np.exp((a["elo"] - b["elo"]) / 900.0), 0.72, 1.38))
    ajuste_elo_b = float(np.clip(np.exp((b["elo"] - a["elo"]) / 900.0), 0.72, 1.38))
    ajuste_ctx_a, ajuste_ctx_b = ajuste_contexto(ctx)
    lambda_a = promedio_general * a["ataque"] * b["defensa_debilidad"] * a["forma_ataque"] * b["forma_defensa"] * ajuste_elo_a * ajuste_ctx_a
    lambda_b = promedio_general * b["ataque"] * a["defensa_debilidad"] * b["forma_ataque"] * a["forma_defensa"] * ajuste_elo_b * ajuste_ctx_b
    lambda_a, lambda_b = float(np.clip(lambda_a, 0.08, 4.50)), float(np.clip(lambda_b, 0.08, 4.50))
    goles = np.arange(MAX_GOLES + 1)
    matriz = np.outer(poisson.pmf(goles, lambda_a), poisson.pmf(goles, lambda_b))
    gana_a, empate, gana_b = float(np.tril(matriz, -1).sum()), float(np.trace(matriz)), float(np.triu(matriz, 1).sum())
    total = gana_a + empate + gana_b
    gana_a, empate, gana_b = [x / total for x in (gana_a, empate, gana_b)]
    marcador = np.unravel_index(np.argmax(matriz), matriz.shape)
    return {
        "goles_a": lambda_a, "goles_b": lambda_b, "gana_a": gana_a * 100, "empate": empate * 100, "gana_b": gana_b * 100,
        "marcador": f"{marcador[0]} - {marcador[1]}",
        "rango_a": (int(poisson.ppf(0.10, lambda_a)), int(poisson.ppf(0.90, lambda_a))),
        "rango_b": (int(poisson.ppf(0.10, lambda_b)), int(poisson.ppf(0.90, lambda_b))), "a": a, "b": b,
    }


def resultado_1x2(goles_a: float, goles_b: float) -> str:
    return "A" if goles_a > goles_b else "B" if goles_a < goles_b else "E"


@st.cache_data(show_spinner=False)
def validar_modelo(partidos: pd.DataFrame) -> dict:
    pruebas = partidos.tail(60).copy()
    errores, aciertos = [], []
    for _, fila in pruebas.iterrows():
        entrenamiento = partidos.loc[partidos["date"] < fila["date"]].copy()
        if len(entrenamiento) < 300:
            continue
        pred = predecir(str(fila["home_team"]), str(fila["away_team"]), entrenamiento, ContextoPartido())
        errores.extend([abs(pred["goles_a"] - float(fila["home_score"])), abs(pred["goles_b"] - float(fila["away_score"]))])
        probabilidades = {"A": pred["gana_a"], "E": pred["empate"], "B": pred["gana_b"]}
        aciertos.append(max(probabilidades, key=probabilidades.get) == resultado_1x2(fila["home_score"], fila["away_score"]))
    return {"mae": float(np.mean(errores)), "precision": float(np.mean(aciertos) * 100), "partidos": int(len(aciertos))}


st.title("⚽ DataGol 2026")
st.subheader("Predictor mundialista con Poisson, Elo y forma reciente")
st.caption("Prototipo académico: calcula escenarios probables a partir de resultados internacionales desde 1998. Los porcentajes son estimaciones, no garantías.")

try:
    datos = cargar_datos()
except Exception as exc:
    st.error("No fue posible cargar los datos históricos.")
    st.code(str(exc))
    st.stop()

with st.spinner("Calculando métricas históricas..."):
    metricas = validar_modelo(datos)

equipos = sorted(set(datos["home_team"]).union(datos["away_team"]))

with st.sidebar:
    st.header("1. Seleccione el partido")
    equipo_a = st.selectbox("Equipo A", equipos, index=equipos.index("Ecuador") if "Ecuador" in equipos else 0)
    equipo_b = st.selectbox("Equipo B", equipos, index=equipos.index("Argentina") if "Argentina" in equipos else 1)
    st.header("2. Contexto del partido")
    fase = st.selectbox("Fase", ["Fase de grupos", "Dieciseisavos", "Octavos", "Cuartos", "Semifinal", "Final"])
    anfitrion_a = st.checkbox(f"{equipo_a} juega como anfitrión")
    anfitrion_b = st.checkbox(f"{equipo_b} juega como anfitrión")
    descanso_a = st.slider(f"Días de descanso: {equipo_a}", 2, 10, 5)
    descanso_b = st.slider(f"Días de descanso: {equipo_b}", 2, 10, 5)
    with st.expander("Ajustes manuales por lesiones o sanciones"):
        st.caption("Use valores negativos si una selección llega debilitada.")
        ajuste_ataque_a = st.slider(f"Ajuste de ataque: {equipo_a}", -10, 10, 0, format="%d %%")
        ajuste_defensa_a = st.slider(f"Debilidad defensiva adicional: {equipo_a}", -10, 10, 0, format="%d %%")
        ajuste_ataque_b = st.slider(f"Ajuste de ataque: {equipo_b}", -10, 10, 0, format="%d %%")
        ajuste_defensa_b = st.slider(f"Debilidad defensiva adicional: {equipo_b}", -10, 10, 0, format="%d %%")
    calcular = st.button("Calcular predicción", type="primary", use_container_width=True)

if equipo_a == equipo_b:
    st.warning("Seleccione dos equipos diferentes.")
    st.stop()

ctx = ContextoPartido(fase=fase, anfitrion_a=anfitrion_a, anfitrion_b=anfitrion_b, descanso_a=descanso_a, descanso_b=descanso_b,
                      ajuste_ataque_a=ajuste_ataque_a, ajuste_ataque_b=ajuste_ataque_b, ajuste_defensa_a=ajuste_defensa_a, ajuste_defensa_b=ajuste_defensa_b)
pred = predecir(equipo_a, equipo_b, datos, ctx)

st.markdown(f"## {equipo_a} vs. {equipo_b}")
st.caption(f"Fase seleccionada: {fase}")
c1, c2, c3 = st.columns(3)
c1.metric(f"Victoria de {equipo_a}", f"{pred['gana_a']:.1f} %")
c2.metric("Empate", f"{pred['empate']:.1f} %")
c3.metric(f"Victoria de {equipo_b}", f"{pred['gana_b']:.1f} %")
c4, c5, c6 = st.columns(3)
c4.metric(f"Goles esperados: {equipo_a}", f"{pred['goles_a']:.2f}")
c5.metric("Marcador más probable", pred["marcador"])
c6.metric(f"Goles esperados: {equipo_b}", f"{pred['goles_b']:.2f}")

st.markdown("### Probabilidades del partido")
grafico = pd.DataFrame({"Resultado": [f"Gana {equipo_a}", "Empate", f"Gana {equipo_b}"], "Probabilidad (%)": [pred["gana_a"], pred["empate"], pred["gana_b"]]}).set_index("Resultado")
st.bar_chart(grafico)

st.markdown("### Factores utilizados por el modelo")
factores = pd.DataFrame({
    "Indicador": ["Elo calculado internamente", "Partidos internacionales analizados", "Promedio de goles anotados", "Promedio de goles recibidos",
                  "Goles anotados en los últimos 10 partidos", "Goles recibidos en los últimos 10 partidos", "Victorias en los últimos 10 partidos",
                  "Rango probable de goles", "Días de descanso", "Condición de anfitrión"],
    equipo_a: [f"{pred['a']['elo']:.0f}", pred['a']['partidos'], f"{pred['a']['promedio_gf']:.2f}", f"{pred['a']['promedio_gc']:.2f}",
               f"{pred['a']['ultimos_10_gf']:.2f}", f"{pred['a']['ultimos_10_gc']:.2f}", pred['a']['victorias_ultimos_10'],
               f"{pred['rango_a'][0]} a {pred['rango_a'][1]}", descanso_a, "Sí" if anfitrion_a else "No"],
    equipo_b: [f"{pred['b']['elo']:.0f}", pred['b']['partidos'], f"{pred['b']['promedio_gf']:.2f}", f"{pred['b']['promedio_gc']:.2f}",
               f"{pred['b']['ultimos_10_gf']:.2f}", f"{pred['b']['ultimos_10_gc']:.2f}", pred['b']['victorias_ultimos_10'],
               f"{pred['rango_b'][0]} a {pred['rango_b'][1]}", descanso_b, "Sí" if anfitrion_b else "No"],
})
st.dataframe(factores, hide_index=True, use_container_width=True)

st.markdown("### Validación histórica")
v1, v2, v3 = st.columns(3)
v1.metric("Margen de error aproximado", f"± {metricas['mae']:.2f} goles")
v2.metric("Precisión histórica 1X2", f"{metricas['precision']:.1f} %")
v3.metric("Partidos usados en backtesting", metricas["partidos"])

with st.expander("¿Cómo funciona la predicción?"):
    st.markdown("""
    1. Se analizan partidos internacionales desde 1998.
    2. Los torneos oficiales pesan más que los amistosos.
    3. Los encuentros recientes tienen más influencia que los antiguos.
    4. Se calcula un Elo propio para representar la fortaleza relativa de cada selección.
    5. Se incorporan los últimos 10 partidos y el contexto seleccionado.
    6. La distribución de Poisson estima la probabilidad de cada marcador.
    7. La suma de escenarios determina victoria, empate o derrota.
    """)

st.divider()
st.caption("DataGol 2026 — prototipo demostrativo de Ciencia de Datos. Las lesiones y sanciones se registran mediante ajustes manuales.")
