# DataGol 2026 — Producción portable v1.1

Corrección de despliegue para Streamlit Cloud.

## Motivo

El modelo Gradient Boosting fue entrenado y evaluado, pero recibió peso 0 % en el
ensamble final. La versión anterior intentaba cargar innecesariamente su archivo
`joblib`, lo que podía provocar incompatibilidad de módulos entre el entorno de
entrenamiento y Streamlit Cloud.

## Solución

- Se elimina la carga de Gradient Boosting durante inferencia.
- La regresión logística seleccionada se exporta a JSON portable.
- La aplicación ya no depende de `joblib` ni de `scikit-learn` en producción.
- Las métricas de validación se conservan sin cambios.
