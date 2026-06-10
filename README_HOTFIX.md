# DataGol 2026 v2.0 — Hotfix de despliegue

## Corrección aplicada

La aplicación ahora importa el módulo con un nombre único:

`datagol_predictor_v2.py`

Esto evita que GitHub o Streamlit reutilicen accidentalmente un archivo antiguo
llamado `predictor.py` perteneciente a una versión previa.

## Despliegue

1. Elimine del repositorio el archivo antiguo `predictor.py`.
2. Suba todos los archivos y carpetas internos de este paquete.
3. Verifique que `app.py` y `datagol_predictor_v2.py` estén en la raíz.
4. Confirme con `Commit changes`.
5. En Streamlit Cloud seleccione `Manage app` → `Reboot app`.
