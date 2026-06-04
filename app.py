from __future__ import annotations

import io
import math
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DataGol 2026", page_icon="⚽", layout="wide")

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
KAGGLE_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/martj42/international-football-results-from-1872-to-2017"
GITHUB_FALLBACK_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

FECHA_INICIO = "2006-01-01"
MAX_GOLES = 8
VENTANA_RECIENTE = 10
MIN_PARTIDOS_EQUIPO = 5
VALIDACION_N = 500
ANFITRIONES_2026 = {"Mexico", "United States", "Canada"}

# Parámetros prudentes: evitan amplificación excesiva
PSEUDO_PARTIDOS = 18.0
DIVISOR_ELO_GOLES = 2000.0
BONO_ANFITRION_GOLES = 1.025
MEZCLA_PROMEDIO_GENERAL = 0.20

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


# ============================================================
# UTILIDADES
# ============================================================
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
    requeridas = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    }
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    df = df.copy()
    if "neutral" not in df.columns:
        df["neutral"] = True

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].fillna(True).astype(bool)

    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
        ]
    )
    df = df.loc[df["date"] >= FECHA_INICIO].copy()
    df["peso_torneo"] = df["tournament"].map(peso_torneo)

    claves = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    ]
    return (
        df.drop_duplicates(subset=claves)
        .sort_values("date")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=21600, show_spinner=False)
def cargar_base() -> tuple[pd.DataFrame, str]:
    try:
        respuesta = requests.get(KAGGLE_ZIP_URL, timeout=25)
        respuesta.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(respuesta.content)) as z:
            candidatos = [
                n for n in z.namelist() if n.lower().endswith("results.csv")
            ]
            if not candidatos:
                candidatos = [
                    n for n in z.namelist() if n.lower().endswith(".csv")
                ]
            if not candidatos:
                raise ValueError("El ZIP no contiene archivos CSV.")

            with z.open(candidatos[0]) as archivo:
                df = pd.read_csv(archivo)

        return normalizar(df), "Kaggle actualizado"
    except Exception:
        df = pd.read_csv(GITHUB_FALLBACK_URL)
        return normalizar(df), "GitHub de respaldo"


def combinar_suplemento(
    base: pd.DataFrame, archivo
) -> tuple[pd.DataFrame, int]:
    if archivo is None:
        return base, 0

    extra = normalizar(pd.read_csv(archivo))
    antes = len(base)
    combinado = normalizar(pd.concat([base, extra], ignore_index=True))
    return combinado, max(0, len(combinado) - antes)


def resultado_1x2(goles_a: float, goles_b: float) -> str:
    if goles_a > goles_b:
        return "A"
    if goles_a < goles_b:
        return "B"
    return "E"


def brier_multiclase(
    probs: np.ndarray, reales: list[str]
) -> float:
    clases = ["A", "E", "B"]
    indices = {c: i for i, c in enumerate(clases)}
    y = np.zeros_like(probs)

    for i, real in enumerate(reales):
        y[i, indices[real]] = 1.0

    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


# ============================================================
# ESTADO PREPARTIDO: ELO + FORMA RECIENTE
# ============================================================
@dataclass
class EquipoEstado:
    elo: float = 1500.0
    historial: deque | None = None

    def __post_init__(self):
        if self.historial is None:
            self.historial = deque(maxlen=VENTANA_RECIENTE)


def resumen_historial(
    historial: deque, media_global: float
) -> dict[str, float]:
    if not historial:
        return {
            "n": 0,
            "gf": media_global,
            "gc": media_global,
            "winrate": 0.33,
        }

    gf = np.array([x[0] for x in historial], dtype=float)
    gc = np.array([x[1] for x in historial], dtype=float)

    return {
        "n": len(historial),
        "gf": float(gf.mean()),
        "gc": float(gc.mean()),
        "winrate": float(np.mean(gf > gc)),
    }


def actualizar_elo(
    estado_a: EquipoEstado,
    estado_b: EquipoEstado,
    goles_a: float,
    goles_b: float,
    neutral: bool,
    peso: float,
) -> None:
    ventaja = 0.0 if neutral else 45.0
    esperado_a = 1.0 / (
        1.0 + 10 ** ((estado_b.elo - (estado_a.elo + ventaja)) / 400.0)
    )

    if goles_a > goles_b:
        real_a, real_b = 1.0, 0.0
    elif goles_a < goles_b:
        real_a, real_b = 0.0, 1.0
    else:
        real_a, real_b = 0.5, 0.5

    diferencia = abs(goles_a - goles_b)
    multiplicador = (
        1.0
        if diferencia <= 1
        else 1.0 + min(diferencia - 1, 3) * 0.12
    )

    k = 28.0 * peso * multiplicador
    estado_a.elo = estado_a.elo + k * (real_a - esperado_a)
    estado_b.elo = estado_b.elo + k * (real_b - (1.0 - esperado_a))


def caracteristicas_prepartido(
    estado_a: EquipoEstado,
    estado_b: EquipoEstado,
    media_global: float,
    anfitrion_a: bool,
    anfitrion_b: bool,
) -> dict[str, float]:
    a = resumen_historial(estado_a.historial, media_global)
    b = resumen_historial(estado_b.historial, media_global)

    return {
        "elo_diff": (estado_a.elo - estado_b.elo) / 400.0,
        "gf_diff": a["gf"] - b["gf"],
        "gc_diff": a["gc"] - b["gc"],
        "win_diff": a["winrate"] - b["winrate"],
        "host_diff": float(anfitrion_a) - float(anfitrion_b),
        "n_a": a["n"],
        "n_b": b["n"],
        "gf_a": a["gf"],
        "gf_b": b["gf"],
        "gc_a": a["gc"],
        "gc_b": b["gc"],
        "win_a": a["winrate"],
        "win_b": b["winrate"],
        "elo_a": estado_a.elo,
        "elo_b": estado_b.elo,
    }


def goles_esperados_desde_features(
    f: dict[str, float],
    media_global: float,
    contexto: Contexto,
) -> tuple[float, float]:
    # Suavizado de los promedios recientes hacia la media general.
    peso_a = f["n_a"] / (f["n_a"] + PSEUDO_PARTIDOS)
    peso_b = f["n_b"] / (f["n_b"] + PSEUDO_PARTIDOS)

    gf_a = peso_a * f["gf_a"] + (1.0 - peso_a) * media_global
    gf_b = peso_b * f["gf_b"] + (1.0 - peso_b) * media_global
    gc_a = peso_a * f["gc_a"] + (1.0 - peso_a) * media_global
    gc_b = peso_b * f["gc_b"] + (1.0 - peso_b) * media_global

    ataque_a = gf_a / media_global
    ataque_b = gf_b / media_global
    debilidad_def_a = gc_a / media_global
    debilidad_def_b = gc_b / media_global

    ajuste_elo_a = float(
        np.clip(
            np.exp((f["elo_a"] - f["elo_b"]) / DIVISOR_ELO_GOLES),
            0.86,
            1.16,
        )
    )
    ajuste_elo_b = float(
        np.clip(
            np.exp((f["elo_b"] - f["elo_a"]) / DIVISOR_ELO_GOLES),
            0.86,
            1.16,
        )
    )

    ajuste_a = 1.0
    ajuste_b = 1.0

    if contexto.anfitrion_a:
        ajuste_a *= BONO_ANFITRION_GOLES
    if contexto.anfitrion_b:
        ajuste_b *= BONO_ANFITRION_GOLES

    diferencia_descanso = int(
        np.clip(contexto.descanso_a - contexto.descanso_b, -3, 3)
    )
    ajuste_a *= 1.0 + diferencia_descanso * 0.01
    ajuste_b *= 1.0 - diferencia_descanso * 0.01

    if contexto.fase != "Fase de grupos":
        ajuste_a *= 0.98
        ajuste_b *= 0.98

    ajuste_a *= 1.0 + contexto.ajuste_ataque_a / 100.0
    ajuste_b *= 1.0 + contexto.ajuste_ataque_b / 100.0
    ajuste_a *= 1.0 + contexto.ajuste_defensa_b / 100.0
    ajuste_b *= 1.0 + contexto.ajuste_defensa_a / 100.0

    lambda_a_cruda = (
        media_global
        * ataque_a
        * debilidad_def_b
        * ajuste_elo_a
        * ajuste_a
    )
    lambda_b_cruda = (
        media_global
        * ataque_b
        * debilidad_def_a
        * ajuste_elo_b
        * ajuste_b
    )

    lambda_a = (
        (1.0 - MEZCLA_PROMEDIO_GENERAL) * lambda_a_cruda
        + MEZCLA_PROMEDIO_GENERAL * media_global
    )
    lambda_b = (
        (1.0 - MEZCLA_PROMEDIO_GENERAL) * lambda_b_cruda
        + MEZCLA_PROMEDIO_GENERAL * media_global
    )

    return (
        float(np.clip(lambda_a, 0.12, 4.2)),
        float(np.clip(lambda_b, 0.12, 4.2)),
    )


# ============================================================
# MODELO 1: POISSON INDEPENDIENTE
# ============================================================
def matriz_poisson(lambda_a: float, lambda_b: float) -> np.ndarray:
    goles = np.arange(MAX_GOLES + 1)
    matriz = np.outer(
        poisson.pmf(goles, lambda_a),
        poisson.pmf(goles, lambda_b),
    )
    return matriz / matriz.sum()


def probs_1x2_desde_matriz(matriz: np.ndarray) -> np.ndarray:
    gana_a = float(np.tril(matriz, -1).sum())
    empate = float(np.trace(matriz))
    gana_b = float(np.triu(matriz, 1).sum())
    total = gana_a + empate + gana_b
    return np.array([gana_a, empate, gana_b], dtype=float) / total


# ============================================================
# MODELO 2: DIXON-COLES
# Ajusta específicamente 0-0, 1-0, 0-1 y 1-1.
# ============================================================
def tau_dixon_coles(
    goles_a: int,
    goles_b: int,
    lambda_a: float,
    lambda_b: float,
    rho: float,
) -> float:
    if goles_a == 0 and goles_b == 0:
        return 1.0 - lambda_a * lambda_b * rho
    if goles_a == 0 and goles_b == 1:
        return 1.0 + lambda_a * rho
    if goles_a == 1 and goles_b == 0:
        return 1.0 + lambda_b * rho
    if goles_a == 1 and goles_b == 1:
        return 1.0 - rho
    return 1.0


def matriz_dixon_coles(
    lambda_a: float, lambda_b: float, rho: float
) -> np.ndarray:
    matriz = matriz_poisson(lambda_a, lambda_b).copy()

    for goles_a in range(2):
        for goles_b in range(2):
            matriz[goles_a, goles_b] *= tau_dixon_coles(
                goles_a,
                goles_b,
                lambda_a,
                lambda_b,
                rho,
            )

    matriz = np.clip(matriz, 1e-12, None)
    return matriz / matriz.sum()


def estimar_rho(
    validacion: pd.DataFrame,
) -> float:
    if validacion.empty:
        return -0.05

    mejor_rho = -0.05
    mejor_loglik = -np.inf

    for rho in np.linspace(-0.20, 0.20, 81):
        loglik = 0.0

        for _, fila in validacion.iterrows():
            tau = tau_dixon_coles(
                int(fila["goles_a"]),
                int(fila["goles_b"]),
                float(fila["lambda_a"]),
                float(fila["lambda_b"]),
                float(rho),
            )

            if tau <= 0:
                loglik = -np.inf
                break

            loglik += math.log(tau)

        if loglik > mejor_loglik:
            mejor_loglik = loglik
            mejor_rho = float(rho)

    return mejor_rho


# ============================================================
# MODELO 3: REGRESIÓN LOGÍSTICA MULTINOMIAL
# Predice A / E / B con variables prepartido.
# ============================================================
FEATURES_LOGIT = [
    "elo_diff",
    "gf_diff",
    "gc_diff",
    "win_diff",
    "host_diff",
]


def crear_modelo_logit() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=0.75,
                ),
            ),
        ]
    )


# ============================================================
# PREPARACIÓN TEMPORAL SIN FUGA DE INFORMACIÓN
# ============================================================
@st.cache_data(show_spinner=False)
def construir_dataset_temporal(
    partidos: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, EquipoEstado], float]:
    partidos = partidos.sort_values("date").reset_index(drop=True).copy()
    media_global = float(
        (
            partidos["home_score"].sum()
            + partidos["away_score"].sum()
        )
        / (2 * len(partidos))
    )

    estados: dict[str, EquipoEstado] = defaultdict(EquipoEstado)
    filas = []

    for _, r in partidos.iterrows():
        equipo_a = str(r["home_team"])
        equipo_b = str(r["away_team"])
        goles_a = float(r["home_score"])
        goles_b = float(r["away_score"])
        neutral = bool(r["neutral"])

        # En el histórico, el equipo local se considera con ventaja solo si no es neutral.
        anfitrion_a = not neutral
        anfitrion_b = False

        estado_a = estados[equipo_a]
        estado_b = estados[equipo_b]

        f = caracteristicas_prepartido(
            estado_a,
            estado_b,
            media_global,
            anfitrion_a,
            anfitrion_b,
        )

        if f["n_a"] >= MIN_PARTIDOS_EQUIPO and f["n_b"] >= MIN_PARTIDOS_EQUIPO:
            contexto_historico = Contexto(
                fase="Fase de grupos",
                anfitrion_a=anfitrion_a,
                anfitrion_b=anfitrion_b,
            )
            lambda_a, lambda_b = goles_esperados_desde_features(
                f,
                media_global,
                contexto_historico,
            )

            filas.append(
                {
                    "date": r["date"],
                    "tournament": str(r["tournament"]),
                    "peso": float(r["peso_torneo"]),
                    "resultado": resultado_1x2(goles_a, goles_b),
                    "goles_a": goles_a,
                    "goles_b": goles_b,
                    "lambda_a": lambda_a,
                    "lambda_b": lambda_b,
                    **{k: f[k] for k in FEATURES_LOGIT},
                }
            )

        actualizar_elo(
            estado_a,
            estado_b,
            goles_a,
            goles_b,
            neutral,
            float(r["peso_torneo"]),
        )

        estado_a.historial.append((goles_a, goles_b))
        estado_b.historial.append((goles_b, goles_a))

    return pd.DataFrame(filas), dict(estados), media_global


def obtener_prob_logit(
    modelo: Pipeline,
    fila: pd.DataFrame,
) -> np.ndarray:
    probas = modelo.predict_proba(fila[FEATURES_LOGIT])[0]
    clases = list(modelo.named_steps["model"].classes_)

    mapa = {clase: proba for clase, proba in zip(clases, probas)}
    return np.array(
        [mapa.get("A", 0.0), mapa.get("E", 0.0), mapa.get("B", 0.0)],
        dtype=float,
    )


def evaluar_modelos(
    temporal: pd.DataFrame,
) -> tuple[dict, Pipeline, float, np.ndarray]:
    """
    Validación temporal estricta en tres bloques:

    1. Entrenamiento: ajusta la regresión logística.
    2. Calibración: estima rho de Dixon-Coles y pesos del ensamble.
    3. Prueba final: reporta métricas sin volver a ajustar parámetros.

    De esta forma, las métricas visibles no se calculan sobre los mismos
    partidos usados para elegir los pesos del ensamble.
    """
    if len(temporal) < 1800:
        raise ValueError(
            "No existen suficientes partidos para una validación temporal estricta."
        )

    test_n = min(350, max(220, len(temporal) // 12))
    calibracion_n = min(350, max(220, len(temporal) // 12))

    entrenamiento = temporal.iloc[: -(calibracion_n + test_n)].copy()
    calibracion = temporal.iloc[-(calibracion_n + test_n) : -test_n].copy()
    prueba = temporal.iloc[-test_n:].copy()

    logit = crear_modelo_logit()
    logit.fit(
        entrenamiento[FEATURES_LOGIT],
        entrenamiento["resultado"],
        model__sample_weight=entrenamiento["peso"],
    )

    rho = estimar_rho(calibracion)

    def probabilidades_bloque(
        bloque: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[float]]:
        probs_poisson = []
        probs_dc = []
        probs_logit = []
        reales = []
        errores_goles = []

        for _, r in bloque.iterrows():
            matriz_p = matriz_poisson(r["lambda_a"], r["lambda_b"])
            matriz_dc = matriz_dixon_coles(r["lambda_a"], r["lambda_b"], rho)

            probs_poisson.append(probs_1x2_desde_matriz(matriz_p))
            probs_dc.append(probs_1x2_desde_matriz(matriz_dc))

            fila_logit = pd.DataFrame([{k: r[k] for k in FEATURES_LOGIT}])
            probs_logit.append(obtener_prob_logit(logit, fila_logit))

            reales.append(r["resultado"])
            errores_goles.extend(
                [
                    abs(float(r["goles_a"]) - float(r["lambda_a"])),
                    abs(float(r["goles_b"]) - float(r["lambda_b"])),
                ]
            )

        return (
            np.array(probs_poisson),
            np.array(probs_dc),
            np.array(probs_logit),
            reales,
            errores_goles,
        )

    # Calibración: los pesos se estiman aquí.
    cal_p, cal_dc, cal_l, cal_reales, _ = probabilidades_bloque(calibracion)

    brier_cal_p = brier_multiclase(cal_p, cal_reales)
    brier_cal_dc = brier_multiclase(cal_dc, cal_reales)
    brier_cal_l = brier_multiclase(cal_l, cal_reales)

    inversos = np.array(
        [
            1.0 / (brier_cal_p + 1e-9),
            1.0 / (brier_cal_dc + 1e-9),
            1.0 / (brier_cal_l + 1e-9),
        ]
    )
    pesos = inversos / inversos.sum()

    # Prueba final: no se reajusta rho ni pesos.
    tst_p, tst_dc, tst_l, tst_reales, errores_goles = probabilidades_bloque(prueba)
    tst_ensemble = pesos[0] * tst_p + pesos[1] * tst_dc + pesos[2] * tst_l

    clases = np.array(["A", "E", "B"])
    reales_arr = np.array(tst_reales)

    metricas = {
        "rho": rho,
        "calibracion_n": len(calibracion),
        "prueba_n": len(prueba),
        "mae_goles": float(np.mean(errores_goles)),
        "q80_error_goles": float(np.quantile(errores_goles, 0.80)),
        "q90_error_goles": float(np.quantile(errores_goles, 0.90)),
        "brier_poisson": brier_multiclase(tst_p, tst_reales),
        "brier_dc": brier_multiclase(tst_dc, tst_reales),
        "brier_logit": brier_multiclase(tst_l, tst_reales),
        "brier_ensemble": brier_multiclase(tst_ensemble, tst_reales),
        "acc_poisson": float(
            np.mean(clases[np.argmax(tst_p, axis=1)] == reales_arr)
        ),
        "acc_dc": float(
            np.mean(clases[np.argmax(tst_dc, axis=1)] == reales_arr)
        ),
        "acc_logit": float(
            np.mean(clases[np.argmax(tst_l, axis=1)] == reales_arr)
        ),
        "acc_ensemble": float(
            np.mean(clases[np.argmax(tst_ensemble, axis=1)] == reales_arr)
        ),
    }

    # Para la predicción pública se aprovecha todo el historial disponible.
    logit_final = crear_modelo_logit()
    logit_final.fit(
        temporal[FEATURES_LOGIT],
        temporal["resultado"],
        model__sample_weight=temporal["peso"],
    )

    return metricas, logit_final, rho, pesos


# ============================================================
# PREDICCIÓN FINAL
# ============================================================
def predecir_partido(
    equipo_a: str,
    equipo_b: str,
    estados: dict[str, EquipoEstado],
    media_global: float,
    logit: Pipeline,
    rho: float,
    pesos: np.ndarray,
    contexto: Contexto,
) -> dict:
    estado_a = estados.get(equipo_a, EquipoEstado())
    estado_b = estados.get(equipo_b, EquipoEstado())

    f = caracteristicas_prepartido(
        estado_a,
        estado_b,
        media_global,
        contexto.anfitrion_a,
        contexto.anfitrion_b,
    )

    lambda_a, lambda_b = goles_esperados_desde_features(
        f,
        media_global,
        contexto,
    )

    matriz_p = matriz_poisson(lambda_a, lambda_b)
    matriz_dc = matriz_dixon_coles(lambda_a, lambda_b, rho)

    probs_p = probs_1x2_desde_matriz(matriz_p)
    probs_dc = probs_1x2_desde_matriz(matriz_dc)

    fila_logit = pd.DataFrame([{k: f[k] for k in FEATURES_LOGIT}])
    probs_l = obtener_prob_logit(logit, fila_logit)

    probs_ensemble = pesos[0] * probs_p + pesos[1] * probs_dc + pesos[2] * probs_l
    probs_ensemble = probs_ensemble / probs_ensemble.sum()

    # Para marcador exacto se combinan solo modelos que producen matrices de score.
    pesos_score = pesos[:2] / pesos[:2].sum()
    matriz_score = pesos_score[0] * matriz_p + pesos_score[1] * matriz_dc
    matriz_score = matriz_score / matriz_score.sum()

    escenarios = []
    for goles_a in range(MAX_GOLES + 1):
        for goles_b in range(MAX_GOLES + 1):
            escenarios.append(
                (
                    goles_a,
                    goles_b,
                    float(matriz_score[goles_a, goles_b]) * 100,
                )
            )

    escenarios = sorted(
        escenarios,
        key=lambda x: x[2],
        reverse=True,
    )[:5]

    return {
        "features": f,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "matriz_score": matriz_score,
        "probs_poisson": probs_p,
        "probs_dc": probs_dc,
        "probs_logit": probs_l,
        "probs_ensemble": probs_ensemble,
        "escenarios": escenarios,
        "al_menos_un_gol_a": (1.0 - poisson.pmf(0, lambda_a)) * 100,
        "al_menos_un_gol_b": (1.0 - poisson.pmf(0, lambda_b)) * 100,
        "mas_25": (1.0 - poisson.cdf(2, lambda_a + lambda_b)) * 100,
        "rango_a": (
            int(poisson.ppf(0.10, lambda_a)),
            int(poisson.ppf(0.90, lambda_a)),
        ),
        "rango_b": (
            int(poisson.ppf(0.10, lambda_b)),
            int(poisson.ppf(0.90, lambda_b)),
        ),
    }


# ============================================================
# INTERFAZ
# ============================================================
st.title("⚽ DataGol 2026")
st.subheader("Predicción final con tres modelos matemáticos")
st.caption(
    "Modelo académico probabilístico. No garantiza el marcador final. "
    "El ensamble combina Poisson, Dixon-Coles y regresión logística multinomial."
)

try:
    base, fuente = cargar_base()
except Exception as exc:
    st.error("No fue posible cargar la base histórica.")
    st.code(str(exc))
    st.stop()

with st.sidebar:
    st.header("Actualización de datos")
    suplemento = st.file_uploader(
        "CSV opcional de partidos recientes",
        type=["csv"],
        help="Cargue únicamente resultados verificados.",
    )

datos, agregados = combinar_suplemento(base, suplemento)
fecha_max = datos["date"].max().date()

with st.spinner("Entrenando y validando los modelos..."):
    temporal, estados, media_global = construir_dataset_temporal(datos)
    metricas, logit_final, rho_final, pesos_finales = evaluar_modelos(temporal)

with st.sidebar:
    st.success(f"Fuente: {fuente}")
    st.caption(f"Último partido disponible: {fecha_max}")
    st.caption(f"Partidos analizados: {len(datos):,}")
    if agregados:
        st.caption(f"Partidos añadidos desde CSV: {agregados}")

equipos = sorted(set(datos["home_team"]).union(datos["away_team"]))

with st.sidebar:
    st.header("Seleccione el partido")
    equipo_a = st.selectbox(
        "Equipo A",
        equipos,
        index=equipos.index("Ecuador") if "Ecuador" in equipos else 0,
        format_func=visible,
    )
    equipo_b = st.selectbox(
        "Equipo B",
        equipos,
        index=equipos.index("Ivory Coast") if "Ivory Coast" in equipos else 1,
        format_func=visible,
    )

    st.header("Contexto")
    fase = st.selectbox(
        "Fase",
        [
            "Fase de grupos",
            "Dieciseisavos",
            "Octavos",
            "Cuartos",
            "Semifinal",
            "Final",
        ],
    )
    anfitrion_a = st.checkbox(
        f"{visible(equipo_a)} juega como anfitrión",
        value=equipo_a in ANFITRIONES_2026,
    )
    anfitrion_b = st.checkbox(
        f"{visible(equipo_b)} juega como anfitrión",
        value=equipo_b in ANFITRIONES_2026,
    )
    descanso_a = st.slider(
        f"Días de descanso: {visible(equipo_a)}",
        2,
        10,
        5,
    )
    descanso_b = st.slider(
        f"Días de descanso: {visible(equipo_b)}",
        2,
        10,
        5,
    )

    with st.expander("Ajustes manuales por lesiones o sanciones"):
        st.caption(
            "Mantenga 0 % si no existe información verificada. "
            "Evite ajustes superiores a ±5 % salvo casos extraordinarios."
        )
        ajuste_ataque_a = st.slider(
            f"Ataque: {visible(equipo_a)}",
            -10,
            10,
            0,
            format="%d %%",
        )
        ajuste_defensa_a = st.slider(
            f"Debilidad defensiva adicional: {visible(equipo_a)}",
            -10,
            10,
            0,
            format="%d %%",
        )
        ajuste_ataque_b = st.slider(
            f"Ataque: {visible(equipo_b)}",
            -10,
            10,
            0,
            format="%d %%",
        )
        ajuste_defensa_b = st.slider(
            f"Debilidad defensiva adicional: {visible(equipo_b)}",
            -10,
            10,
            0,
            format="%d %%",
        )

if equipo_a == equipo_b:
    st.warning("Seleccione dos equipos diferentes.")
    st.stop()

contexto = Contexto(
    fase=fase,
    anfitrion_a=anfitrion_a,
    anfitrion_b=anfitrion_b,
    descanso_a=descanso_a,
    descanso_b=descanso_b,
    ajuste_ataque_a=ajuste_ataque_a,
    ajuste_ataque_b=ajuste_ataque_b,
    ajuste_defensa_a=ajuste_defensa_a,
    ajuste_defensa_b=ajuste_defensa_b,
)

pred = predecir_partido(
    equipo_a,
    equipo_b,
    estados,
    media_global,
    logit_final,
    rho_final,
    pesos_finales,
    contexto,
)

a_es = visible(equipo_a)
b_es = visible(equipo_b)
ens = pred["probs_ensemble"]

st.markdown(f"## {a_es} vs. {b_es}")
st.caption(f"Fase seleccionada: {fase}")

c1, c2, c3 = st.columns(3)
c1.metric(f"Victoria de {a_es}", f"{ens[0] * 100:.1f} %")
c2.metric("Empate", f"{ens[1] * 100:.1f} %")
c3.metric(f"Victoria de {b_es}", f"{ens[2] * 100:.1f} %")

c4, c5, c6 = st.columns(3)
c4.metric(f"Goles esperados: {a_es}", f"{pred['lambda_a']:.2f}")
c5.metric(
    "Marcador modal",
    f"{pred['escenarios'][0][0]} - {pred['escenarios'][0][1]}",
)
c6.metric(f"Goles esperados: {b_es}", f"{pred['lambda_b']:.2f}")

st.markdown("### Cinco marcadores exactos más probables")
top = pd.DataFrame(
    [
        {
            "Posición": i + 1,
            "Marcador": f"{a_es} {ga} - {gb} {b_es}",
            "Probabilidad": f"{p:.1f} %",
        }
        for i, (ga, gb, p) in enumerate(pred["escenarios"])
    ]
)
st.dataframe(top, hide_index=True, use_container_width=True)
st.caption(
    "El marcador modal es el escenario individual más probable; "
    "no es una certeza."
)

st.markdown("### Comparación de los tres modelos")
comparacion = pd.DataFrame(
    {
        "Modelo": [
            "1. Poisson independiente",
            "2. Dixon-Coles",
            "3. Regresión logística multinomial",
            "Ensamble final",
        ],
        f"Victoria {a_es}": [
            f"{pred['probs_poisson'][0] * 100:.1f} %",
            f"{pred['probs_dc'][0] * 100:.1f} %",
            f"{pred['probs_logit'][0] * 100:.1f} %",
            f"{pred['probs_ensemble'][0] * 100:.1f} %",
        ],
        "Empate": [
            f"{pred['probs_poisson'][1] * 100:.1f} %",
            f"{pred['probs_dc'][1] * 100:.1f} %",
            f"{pred['probs_logit'][1] * 100:.1f} %",
            f"{pred['probs_ensemble'][1] * 100:.1f} %",
        ],
        f"Victoria {b_es}": [
            f"{pred['probs_poisson'][2] * 100:.1f} %",
            f"{pred['probs_dc'][2] * 100:.1f} %",
            f"{pred['probs_logit'][2] * 100:.1f} %",
            f"{pred['probs_ensemble'][2] * 100:.1f} %",
        ],
    }
)
st.dataframe(comparacion, hide_index=True, use_container_width=True)

st.markdown("### Incertidumbre y validación temporal")
v1, v2, v3 = st.columns(3)
v1.metric("Error medio absoluto en goles", f"± {metricas['mae_goles']:.2f}")
v2.metric(
    "80 % de errores por debajo de",
    f"± {metricas['q80_error_goles']:.2f}",
)
v3.metric(
    "Partidos reservados para prueba final",
    f"{metricas['prueba_n']}",
)

metricas_tabla = pd.DataFrame(
    {
        "Modelo": [
            "Poisson independiente",
            "Dixon-Coles",
            "Regresión logística multinomial",
            "Ensamble final",
        ],
        "Brier Score": [
            f"{metricas['brier_poisson']:.3f}",
            f"{metricas['brier_dc']:.3f}",
            f"{metricas['brier_logit']:.3f}",
            f"{metricas['brier_ensemble']:.3f}",
        ],
        "Exactitud 1X2": [
            f"{metricas['acc_poisson'] * 100:.1f} %",
            f"{metricas['acc_dc'] * 100:.1f} %",
            f"{metricas['acc_logit'] * 100:.1f} %",
            f"{metricas['acc_ensemble'] * 100:.1f} %",
        ],
        "Peso en el ensamble": [
            f"{pesos_finales[0] * 100:.1f} %",
            f"{pesos_finales[1] * 100:.1f} %",
            f"{pesos_finales[2] * 100:.1f} %",
            "100 %",
        ],
    }
)
st.dataframe(metricas_tabla, hide_index=True, use_container_width=True)

st.caption(
    "Un Brier Score menor indica mejor calidad probabilística. "
    "La exactitud 1X2 no debe interpretarse como certeza individual."
)

st.markdown("### Indicadores adicionales")
indicadores = pd.DataFrame(
    {
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
    }
)
st.dataframe(indicadores, hide_index=True, use_container_width=True)

st.markdown("### Factores prepartido")
f = pred["features"]
factores = pd.DataFrame(
    {
        "Indicador": [
            "Elo interno",
            "Partidos recientes disponibles",
            "Promedio GF últimos 10",
            "Promedio GC últimos 10",
            "Tasa de victorias últimos 10",
            "Condición de anfitrión",
            "Días de descanso",
        ],
        a_es: [
            f"{f['elo_a']:.0f}",
            f"{int(f['n_a'])}",
            f"{f['gf_a']:.2f}",
            f"{f['gc_a']:.2f}",
            f"{f['win_a'] * 100:.1f} %",
            "Sí" if anfitrion_a else "No",
            descanso_a,
        ],
        b_es: [
            f"{f['elo_b']:.0f}",
            f"{int(f['n_b'])}",
            f"{f['gf_b']:.2f}",
            f"{f['gc_b']:.2f}",
            f"{f['win_b'] * 100:.1f} %",
            "Sí" if anfitrion_b else "No",
            descanso_b,
        ],
    }
)
st.dataframe(factores, hide_index=True, use_container_width=True)

with st.expander("Metodología científica resumida"):
    st.markdown(
        f"""
        **Modelo 1 — Poisson independiente.** Estima la probabilidad de cada
        marcador a partir de los goles esperados de cada selección.

        **Modelo 2 — Dixon-Coles.** Corrige las probabilidades de resultados
        bajos, especialmente 0-0, 1-0, 0-1 y 1-1. El parámetro estimado es
        `rho = {rho_final:.3f}`.

        **Modelo 3 — Regresión logística multinomial.** Predice directamente
        victoria, empate o derrota mediante Elo interno, forma ofensiva,
        forma defensiva, tasa reciente de victorias y anfitrión.

        **Ensamble.** Las tres probabilidades se combinan con pesos calculados
        a partir del desempeño en un bloque temporal de calibración de
        `{metricas['calibracion_n']}` partidos. Las métricas visibles se reportan
        sobre un bloque posterior e independiente de `{metricas['prueba_n']}`
        partidos. Esto evita seleccionar
        manualmente el modelo que más convenga para un partido específico.
        """
    )

st.divider()
st.caption(
    f"Datos disponibles hasta: {fecha_max}. "
    "Revise esta fecha y las bajas confirmadas antes de presentar una predicción."
)
