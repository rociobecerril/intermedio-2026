# 🗞️ Tarea 1 – Noticias peruanas: del titular al clasificador

## 1. Pregunta del proyecto
**¿Pueden convertir titulares peruanos en un clasificador de noticias?**
Sí: un DistilBERT multilingüe afinado con 280 titulares de La República clasifica en `politica`, `economia`, `deportes` y `ciencia_tecnologia` con **F1 macro = 0.93** en un test reservado, frente a **0.65** de un zero-shot sin entrenar.

## 2. Medio y fechas de extracción
- **Medio:** La República (`larepublica.pe`) – Opción A. **400 noticias**, 100 por categoría.
- **Fecha de descarga:** 2026-10-18. Fechas de publicación: 2024-02-29 a 2026-10-18.
- **Fuentes:** RSS y sitemaps oficiales (≈50 notas de las 4 categorías, no alcanzaban) y, para completar, URLs históricas listadas con el índice CDX de Internet Archive y descargadas desde `larepublica.pe`. Se respetó su `robots.txt` (no se usó la paginación, que está bloqueada), con `User-Agent` descriptivo, pausa de 1–2 s y un reintento como máximo. Detalle en `01_recoleccion.ipynb`.

| Modelo (sobre el test de 60 noticias) | Accuracy | F1 macro |
| :-- | :-: | :-: |
| Zero-shot (`Recognai/bert-base-spanish-wwm-cased-xnli`) | 0.667 | 0.630 (0.648 en las 40 noticias del enunciado) |
| LLM débil (`gemini-3.1-flash-lite`) | 0.950 | 0.949 |
| **DistilBERT fine-tuned** | **0.933** | **0.932** |

Acuerdo del LLM con la auditoría humana: **19/20**. Split 70/15/15 estratificado (`random_state=42`): 280 / 60 / 60.

## 3. Dos hallazgos analíticos
1. **El fine-tuning aporta ~30 puntos de F1 sobre el zero-shot** (0.93 vs 0.63) con solo 280 ejemplos y 30 s de entrenamiento, y queda a ~1 acierto del LLM (0.95) pero con ~9 ms de latencia local y sin depender de una API.
2. **La política es la categoría más difícil (F1 0.86) y la ciencia/tecnología la más fácil (1.00).** Ciencia y tecnología tiene vocabulario propio; política es heterogénea y sus errores son notas cuyo titular habla de El Niño, anécdotas o institutos, sin vocabulario político típico.

## 4. Dos limitaciones técnicas
1. **Datos pequeños y de un solo medio:** test de 60 noticias (15 por clase), así que un error mueve el F1 de una clase ~3 puntos; además, 350 de las 400 notas vienen de un muestreo histórico, no de un corte temporal continuo. El modelo duda con titulares fuera de distribución (p. ej. un titular de iPhone con IA salió como `economia` con solo 46 % de probabilidad).
2. **Etiqueta de referencia ruidosa:** la verdad de terreno es la sección del diario, que en notas de frontera (trámites del Estado, regulación) no coincide con el criterio de un lector. Los pesos del modelo (`modelo/model.safetensors`, ~540 MB) **no están en el repositorio** (superan lo razonable para GitHub); se regeneran ejecutando `02_clasificacion.ipynb`.

## 5. Cómo ejecutar en local
```bash
cd tarea_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) (opcional) volver a generar datos y modelo
python -m ipykernel install --user --name tarea1 --display-name "tarea_1 (.venv)"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=tarea1 01_recoleccion.ipynb
echo "GEMINI_API_KEY=tu_clave" > .env          # solo si se borra datos/etiquetas_llm.csv
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=tarea1 02_clasificacion.ipynb

# 2) dashboard (requiere modelo/model.safetensors, generado por el paso anterior)
streamlit run app.py
```
Estructura: `01_recoleccion.ipynb`, `02_clasificacion.ipynb`, `app.py`, `requirements.txt`, `datos/` (`noticias.csv`, `etiquetas_llm.csv`, `auditoria_20.csv`, `raw/`), `modelo/` (configuración, tokenizer y `metricas.json`).
