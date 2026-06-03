import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(
    page_title="DataGol 2026",
    page_icon="⚽",
    layout="wide",
)

URL_DATOS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
MAX_GOLES = 7

@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pd.read_csv(URL_DATOS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.loc[
            (df["tournament"] == "FIFA World Cup")
            & (df["date"] >= "1998-01-01")
        ]
        .dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
        .copy()
    )
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df.dropna(subset=["home_score", "away_score"])

def formato_largo(partidos):
    local = partidos[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    local.columns = ["date", "equipo", "rival", "goles_favor", "goles_contra"]

    visitante = partidos[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    visitante.columns = ["date", "equipo", "rival", "goles_favor", "goles_contra"]

    return pd.concat([local, visitante], ignore_index=True)

def construir_estadisticas(partidos):
    largo = formato_largo(partidos)
    fecha_ref = largo["date"].max()
    antiguedad_anios = (fecha_ref - largo["date"]).dt.days / 365.25

    # Los partidos recientes reciben más peso.
    largo["peso"] = np.exp(-np.log(2) * antiguedad_anios / 8.0)

    promedio_general = float(np.average(largo["goles_favor"], weights=largo["peso"]))
    promedio_general = max(promedio_general, 0.05)

    filas = []
    for equipo, grupo in largo.groupby("equipo"):
        gf = float(np.average(grupo["goles_favor"], weights=grupo["peso"]))
        gc = float(np.average(grupo["goles_contra"], weights=grupo["peso"]))
        filas.append(
            {
                "equipo": equipo,
                "partidos": int(len(grupo)),
                "ataque": gf / promedio_general,
                "defensa_debilidad": gc / promedio_general,
            }
        )

    return promedio_general, pd.DataFrame(filas).set_index("equipo")

def parametros_equipo(equipo, tabla):
    if equipo in tabla.index:
        fila = tabla.loc[equipo]
        return {
            "partidos": int(fila["partidos"]),
            "ataque": float(fila["ataque"]),
            "defensa_debilidad": float(fila["defensa_debilidad"]),
        }
    return {"partidos": 0, "ataque": 1.0, "defensa_debilidad": 1.0}

def predecir(equipo_a, equipo_b, partidos):
    promedio_general, tabla = construir_estadisticas(partidos)
    a = parametros_equipo(equipo_a, tabla)
    b = parametros_equipo(equipo_b, tabla)

    lambda_a = max(0.05, promedio_general * a["ataque"] * b["defensa_debilidad"])
    lambda_b = max(0.05, promedio_general * b["ataque"] * a["defensa_debilidad"])

    goles = np.arange(MAX_GOLES + 1)
    matriz = np.outer(poisson.pmf(goles, lambda_a), poisson.pmf(goles, lambda_b))

    gana_a = float(np.tril(matriz, -1).sum())
    empate = float(np.trace(matriz))
    gana_b = float(np.triu(matriz, 1).sum())
    total = gana_a + empate + gana_b
    gana_a, empate, gana_b = [x / total for x in (gana_a, empate, gana_b)]

    marcador = np.unravel_index(np.argmax(matriz), matriz.shape)
    rango_a = (int(poisson.ppf(0.10, lambda_a)), int(poisson.ppf(0.90, lambda_a)))
    rango_b = (int(poisson.ppf(0.10, lambda_b)), int(poisson.ppf(0.90, lambda_b)))

    return {
        "goles_a": lambda_a,
        "goles_b": lambda_b,
        "gana_a": gana_a * 100,
        "empate": empate * 100,
        "gana_b": gana_b * 100,
        "marcador": f"{marcador[0]} - {marcador[1]}",
        "rango_a": rango_a,
        "rango_b": rango_b,
        "partidos_a": a["partidos"],
        "partidos_b": b["partidos"],
    }

def resultado(goles_a, goles_b):
    if goles_a > goles_b:
        return "A"
    if goles_a < goles_b:
        return "B"
    return "E"

@st.cache_data(show_spinner=False)
def validar_modelo(partidos):
    pruebas = partidos.loc[partidos["date"] >= "2014-01-01"].sort_values("date")
    errores = []
    aciertos = []

    for _, fila in pruebas.iterrows():
        entrenamiento = partidos.loc[partidos["date"] < fila["date"]]
        if len(entrenamiento) < 100:
            continue

        pred = predecir(fila["home_team"], fila["away_team"], entrenamiento)
        errores.extend(
            [
                abs(pred["goles_a"] - float(fila["home_score"])),
                abs(pred["goles_b"] - float(fila["away_score"])),
            ]
        )
        probabilidades = {"A": pred["gana_a"], "E": pred["empate"], "B": pred["gana_b"]}
        pronostico = max(probabilidades, key=probabilidades.get)
        aciertos.append(pronostico == resultado(fila["home_score"], fila["away_score"]))

    return {
        "mae": float(np.mean(errores)),
        "precision": float(np.mean(aciertos) * 100),
        "partidos": len(aciertos),
    }

st.title("⚽ DataGol 2026")
st.subheader("Predictor mundialista basado en datos históricos")
st.caption(
    "Prototipo académico de Ciencia de Datos: estima goles y probabilidades mediante "
    "una distribución de Poisson aplicada al histórico mundialista desde 1998."
)

try:
    datos = cargar_datos()
except Exception:
    st.error("No fue posible descargar los datos históricos. Intente recargar la aplicación.")
    st.stop()

metricas = validar_modelo(datos)
equipos = sorted(set(datos["home_team"]).union(datos["away_team"]))

with st.sidebar:
    st.header("Seleccione el partido")
    equipo_a = st.selectbox("Equipo A", equipos, index=equipos.index("Ecuador") if "Ecuador" in equipos else 0)
    equipo_b = st.selectbox("Equipo B", equipos, index=equipos.index("Argentina") if "Argentina" in equipos else 1)
    calcular = st.button("Calcular predicción", type="primary", use_container_width=True)

if equipo_a == equipo_b:
    st.warning("Seleccione dos equipos diferentes.")
    st.stop()

if calcular or True:
    pred = predecir(equipo_a, equipo_b, datos)

    st.markdown(f"## {equipo_a} vs. {equipo_b}")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Probabilidad de victoria: {equipo_a}", f"{pred['gana_a']:.1f} %")
    c2.metric("Probabilidad de empate", f"{pred['empate']:.1f} %")
    c3.metric(f"Probabilidad de victoria: {equipo_b}", f"{pred['gana_b']:.1f} %")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"Goles esperados: {equipo_a}", f"{pred['goles_a']:.2f}")
    c5.metric("Marcador más probable", pred["marcador"])
    c6.metric(f"Goles esperados: {equipo_b}", f"{pred['goles_b']:.2f}")

    st.markdown("### Probabilidades del partido")
    grafico = pd.DataFrame(
        {
            "Resultado": [f"Gana {equipo_a}", "Empate", f"Gana {equipo_b}"],
            "Probabilidad (%)": [pred["gana_a"], pred["empate"], pred["gana_b"]],
        }
    ).set_index("Resultado")
    st.bar_chart(grafico)

    st.markdown("### Incertidumbre y validación histórica")
    resumen = pd.DataFrame(
        {
            "Indicador": [
                f"Rango probable de goles de {equipo_a}",
                f"Rango probable de goles de {equipo_b}",
                "Margen de error histórico aproximado",
                "Precisión histórica del resultado 1X2",
                "Partidos utilizados para validar",
            ],
            "Resultado": [
                f"{pred['rango_a'][0]} a {pred['rango_a'][1]} goles",
                f"{pred['rango_b'][0]} a {pred['rango_b'][1]} goles",
                f"± {metricas['mae']:.2f} goles",
                f"{metricas['precision']:.1f} %",
                metricas["partidos"],
            ],
        }
    )
    st.dataframe(resumen, hide_index=True, use_container_width=True)

    with st.expander("¿Cómo interpretar estos resultados?"):
        st.write(
            "El sistema no adivina el marcador. Calcula escenarios probables a partir del "
            "rendimiento histórico de las selecciones en mundiales. El margen de error se obtiene "
            "comparando predicciones retrospectivas con partidos que ya ocurrieron."
        )

st.divider()
st.caption(
    "Proyecto demostrativo de Ciencia de Datos. Para una segunda versión se pueden añadir "
    "ranking Elo, últimos partidos internacionales y registro de participantes."
)
