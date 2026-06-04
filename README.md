# DataGol 2026 — Versión final con tres modelos matemáticos

## Modelos

1. Poisson independiente para goles y marcadores exactos.
2. Dixon-Coles para corregir resultados bajos.
3. Regresión logística multinomial para victoria, empate y derrota.

## Ensamble

Los tres modelos se combinan con pesos obtenidos mediante validación temporal y Brier Score.

## Qué muestra la aplicación

- goles esperados;
- victoria, empate y derrota;
- marcador modal;
- cinco marcadores más probables;
- comparación de modelos;
- margen de error;
- Brier Score;
- exactitud 1X2;
- factores prepartido;
- fuente y fecha del último partido disponible.

## Actualización de GitHub

Reemplace estos archivos:
- `app.py`
- `requirements.txt`

Cargue `partidos_recientes_plantilla.csv` únicamente cuando disponga de resultados verificados.

## Validación temporal estricta

La versión final separa cronológicamente los datos en:
- entrenamiento;
- calibración;
- prueba final independiente.

El parámetro Dixon-Coles y los pesos del ensamble se seleccionan con el bloque de calibración.
Las métricas visibles se calculan únicamente con el bloque posterior de prueba.
