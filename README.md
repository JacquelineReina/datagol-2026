# DataGol 2026 — Producción congelada v1

Aplicación Streamlit de inferencia para el predictor probabilístico validado.

## Principio de producción

La aplicación **no entrena ni calibra modelos al abrirse**. Carga artefactos congelados:

- `models/dixon_coles.json`
- `models/elo_logit.joblib`
- `models/gradient_boosting.joblib`
- `models/inference_state.json`
- `models/production_metadata.json`

## Modelo publicado

Las probabilidades 1X2 utilizan el ensamble validado:

- Dixon-Coles completo: 79,21 %
- Elo + logística multinomial: 20,79 %
- Gradient Boosting calibrado: 0 %

Gradient Boosting permanece visible para auditoría, pero el optimizador lo excluyó del ensamble porque no mejoró la calibración.

Los marcadores exactos provienen de Dixon-Coles.

## Publicación en Streamlit

Suba todos los archivos y carpetas de este paquete a un repositorio GitHub y seleccione `app.py` como archivo principal.

## Fuente del calendario

`data/worldcup_2026_group_stage.csv` es una captura pública estructurada del calendario de fase de grupos. Antes de una publicación institucional debe contrastarse con la programación oficial FIFA.
