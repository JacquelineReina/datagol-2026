import pandas as pd
import streamlit as st

from predictor import FrozenPredictor, fixture_context, load_fixtures

st.set_page_config(page_title="DataGol 2026", page_icon="⚽", layout="wide")

NOMBRES = {
    "Ivory Coast": "Costa de Marfil",
    "United States": "Estados Unidos",
    "South Africa": "Sudáfrica",
    "South Korea": "Corea del Sur",
    "Czech Republic": "República Checa",
    "Netherlands": "Países Bajos",
    "Cape Verde": "Cabo Verde",
    "Saudi Arabia": "Arabia Saudita",
    "DR Congo": "RD del Congo",
    "Curaçao": "Curazao",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
}

def es(name):
    return NOMBRES.get(name, name)

@st.cache_resource
def predictor():
    return FrozenPredictor()

@st.cache_data
def fixtures():
    return load_fixtures()

p = predictor()
fx = fixtures()

st.title("⚽ DataGol 2026")
st.subheader("Predictor probabilístico congelado y validado temporalmente")
st.caption(
    f"Versión {p.meta['version']} · Corte de datos: {p.meta['cutoff']} · "
    "La aplicación solo realiza inferencia; no reentrena al abrirse."
)

modo = st.sidebar.radio("Modo", ["Calendario del Mundial 2026", "Comparación manual"])

if modo == "Calendario del Mundial 2026":
    labels = [
        f"{row.date.date()} · Grupo {row.group} · {es(row.team_a)} vs. {es(row.team_b)}"
        for _, row in fx.iterrows()
    ]
    selected = st.sidebar.selectbox(
        "Partido",
        range(len(labels)),
        format_func=lambda index: labels[index],
    )
    context = fixture_context(fx, selected)
    team_a, team_b = context["team_a"], context["team_b"]
    host_a, host_b = context["host_a"], context["host_b"]
    rest_a, rest_b = context["rest_a"], context["rest_b"]

    st.sidebar.caption(f"Sede: {context['row'].ground}")
    st.sidebar.caption(
        f"Descanso estimado: {es(team_a)} {rest_a} días · "
        f"{es(team_b)} {rest_b} días"
    )
else:
    teams = sorted(set(fx.team_a).union(fx.team_b))
    team_a = st.sidebar.selectbox("Selección A", teams, format_func=es)
    team_b = st.sidebar.selectbox("Selección B", teams, index=1, format_func=es)
    host_a = st.sidebar.checkbox(f"{es(team_a)} juega como anfitrión")
    host_b = st.sidebar.checkbox(f"{es(team_b)} juega como anfitrión")
    rest_a = st.sidebar.slider(f"Días de descanso: {es(team_a)}", 2, 14, 7)
    rest_b = st.sidebar.slider(f"Días de descanso: {es(team_b)}", 2, 14, 7)

if team_a == team_b:
    st.warning("Seleccione dos equipos diferentes.")
    st.stop()

pred = p.predict(team_a, team_b, host_a, host_b, rest_a, rest_b)
ensemble = pred["probabilities"]["ensemble"]
lambda_a, lambda_b = pred["expected_goals"]

st.markdown(f"## {es(team_a)} vs. {es(team_b)}")

c1, c2, c3 = st.columns(3)
c1.metric(f"Victoria de {es(team_a)}", f"{ensemble[0] * 100:.1f} %")
c2.metric("Empate", f"{ensemble[1] * 100:.1f} %")
c3.metric(f"Victoria de {es(team_b)}", f"{ensemble[2] * 100:.1f} %")

c4, c5, c6 = st.columns(3)
c4.metric(f"Goles esperados: {es(team_a)}", f"{lambda_a:.2f}")
c5.metric("Marcador modal", pred["top5_scores"][0]["score"])
c6.metric(f"Goles esperados: {es(team_b)}", f"{lambda_b:.2f}")

st.markdown("### Cinco marcadores exactos más probables")
top = pd.DataFrame(
    [
        {
            "Posición": index + 1,
            "Marcador": f"{es(team_a)} {item['score']} {es(team_b)}",
            "Probabilidad": f"{item['prob'] * 100:.1f} %",
        }
        for index, item in enumerate(pred["top5_scores"])
    ]
)
st.dataframe(top, hide_index=True, use_container_width=True)
st.caption(
    "El marcador exacto proviene del modelo Dixon-Coles. "
    "Es el escenario modal, no una certeza."
)

st.markdown("### Comparación de modelos")
comparison = []

for label, key in [
    ("Dixon-Coles completo", "dixon_coles"),
    ("Elo + logística multinomial", "elo_logit"),
    ("Ensamble publicado", "ensemble"),
]:
    probs = pred["probabilities"][key]
    comparison.append(
        {
            "Modelo": label,
            f"Victoria {es(team_a)}": f"{probs[0] * 100:.1f} %",
            "Empate": f"{probs[1] * 100:.1f} %",
            f"Victoria {es(team_b)}": f"{probs[2] * 100:.1f} %",
        }
    )

comparison.append(
    {
        "Modelo": "Gradient Boosting calibrado",
        f"Victoria {es(team_a)}": "Excluido",
        "Empate": "Peso 0 %",
        f"Victoria {es(team_b)}": "No se carga en producción",
    }
)

st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)

st.markdown("### Validación fuera de muestra")
metrics = p.meta["test_metrics"]["ensemble"]
score_metrics = p.meta["score_metrics"]

v1, v2, v3, v4 = st.columns(4)
v1.metric("Exactitud 1X2", f"{metrics['accuracy_1x2'] * 100:.1f} %")
v2.metric("Log Loss", f"{metrics['log_loss']:.3f}")
v3.metric("Brier Score", f"{metrics['brier']:.3f}")
v4.metric("ECE", f"{metrics['ece']:.3f}")

st.caption(
    f"Marcador exacto: top 1 {score_metrics['score_top1_coverage'] * 100:.1f} % · "
    f"top 3 {score_metrics['score_top3_coverage'] * 100:.1f} % · "
    f"top 5 {score_metrics['score_top5_coverage'] * 100:.1f} %. "
    f"El resultado modal más frecuente en prueba fue "
    f"{score_metrics['most_common_modal_score']} con "
    f"{score_metrics['most_common_modal_share'] * 100:.1f} %."
)

with st.expander("Metodología y alcance"):
    st.markdown(
        f"""
- **Probabilidad 1X2 publicada:** ensamble validado de Dixon-Coles
  ({p.meta['ensemble_weights']['dixon_coles'] * 100:.1f} %) y Elo-logística
  ({p.meta['ensemble_weights']['elo_logit'] * 100:.1f} %).
- **Gradient Boosting:** fue entrenado, calibrado y evaluado, pero recibió
  peso {p.meta['ensemble_weights']['gradient_boosting'] * 100:.1f} %.
  Por esa razón no se carga durante la inferencia.
- **Marcadores exactos:** Dixon-Coles completo penalizado, con decaimiento temporal.
- **Base pública conservadora:** partidos amistosos y clasificatorios con
  objetivo compatible con 90 minutos.
- **Corte de datos:** {p.meta['cutoff']}.

El sistema estima probabilidades. No garantiza resultados y no debe utilizarse
como asesoría de apuestas.
"""
    )
