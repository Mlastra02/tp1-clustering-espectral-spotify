# Clustering espectral de países según lo que escuchan en Spotify

Trabajo Práctico 1 · Álgebra Lineal y Optimización para Data Science (ING-560), Universidad Adolfo Ibáñez.
Integrantes: Max Lastra y Benjamín Nuñez.

Se construye una red de 72 países, conectados por la similitud de su Top 50 diario de Spotify
(18-10-2023 a 11-06-2025), y se aplica clustering espectral para ver cómo se agrupan y si los grupos
se explican mejor por idioma, geografía o clima.

## Contenido

| Archivo / carpeta | Qué es |
|---|---|
| `Informe_TP1_Clustering_Espectral.pdf` | Informe final |
| `Informe_TP1_Clustering_Espectral.tex` | Fuente LaTeX del informe |
| `Informe_TP1_Clustering_Espectral.docx` | Versión Word del informe |
| `clustering_espectral_spotify.ipynb` | Notebook con todo el análisis (con resultados ya ejecutados) |
| `clustering_espectral_spotify.py` | El mismo notebook como script (formato jupytext) |
| `figuras/` | Las 9 figuras del informe, generadas por el notebook |
| `resultados/` | Tablas en CSV: comunidades por país, evaluación y los tres experimentos |

## Cómo obtener los datos

No hace falta descargar nada a mano ni tener cuenta de Kaggle. **La primera ejecución del notebook descarga
automáticamente todos los datos (≈ 330 MB) en la carpeta `data/`**, desde estas fuentes públicas:

| Datos | Fuente |
|---|---|
| Top 50 diario por país | [Top Spotify Songs in 73 Countries (Kaggle, ODC-By)](https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated) |
| Clima Köppen-Geiger 1991–2020 | [Figshare (CC0)](https://figshare.com/articles/dataset/21937571) |
| Ciudades y población | [GeoNames cities15000 (CC BY 4.0)](https://download.geonames.org/export/dump/) |
| Temperatura media anual | [WorldClim 2.1](https://www.worldclim.org) |
| Límites de países | [Natural Earth 1:50m](https://www.naturalearthdata.com) |

Los datos no están en este repositorio porque el CSV de Spotify pesa unos 500 MB, más que el límite de GitHub.
Si alguna descarga falla, se puede bajar el archivo desde el enlace de la tabla y dejarlo en `data/`
(el CSV de Spotify como `data/universal_top_spotify_songs.csv`); el notebook usa los archivos que ya existen.
El análisis filtra siempre el período 18-10-2023 a 11-06-2025, así que da los mismos resultados aunque el
dataset de Kaggle tenga fechas nuevas.

## Cómo ejecutarlo

```bash
pip install numpy pandas matplotlib scipy scikit-learn networkx tifffile imagecodecs jupyter
jupyter notebook clustering_espectral_spotify.ipynb
```

Ejecutar todas las celdas en orden. El notebook regenera las figuras de `figuras/` y las tablas de `resultados/`.
