# DataGol 2026 — Versión 1.1

Actualización del prototipo:

- intenta descargar primero el dataset público actualizado de Kaggle;
- conserva GitHub como respaldo;
- muestra la fecha real del último partido disponible;
- permite cargar un CSV suplementario de partidos recientes;
- traduce nombres frecuentes al español;
- activa por defecto la condición de anfitrión para México, Estados Unidos y Canadá;
- muestra los cinco marcadores exactos más probables;
- muestra probabilidad de al menos un gol y más de 2,5 goles;
- mantiene Poisson + Elo interno + forma reciente + contexto.

## Actualización en GitHub

1. Descomprima este ZIP.
2. En el repositorio `datagol-2026`, reemplace `app.py` y `requirements.txt`.
3. Cargue también `partidos_recientes_plantilla.csv`.
4. Confirme con **Commit changes**.
5. Streamlit se actualizará automáticamente.

## Uso del CSV opcional

La plantilla permite agregar resultados recientes cuando la fuente automática todavía no los incluya.
