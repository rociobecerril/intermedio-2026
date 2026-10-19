# 2026_Tarea_1

Fuente: https://github.com/d2cml-ai/Data-Science-Python/issues/188

# 🗞️ Tarea 1 – Noticias peruanas: del titular al clasificador

## Pregunta guía del trabajo

> **¿Pueden convertir titulares peruanos en un clasificador de noticias?**

En este proyecto van a extraer noticias reales de **La República o El Comercio**, etiquetarlas de forma semi-supervisada con ayuda de un LLM, entrenar un modelo Transformer ligero (DistilBERT / BETO) y desplegar un dashboard interactivo en Streamlit donde cualquier persona pueda ingresar un titular y ver la categoría predicha junto con su nivel de confianza.

---

## 📌 Instrucciones generales

El trabajo es **individual** y corresponde a la **Tarea 1** del curso.

### 1. Datos a recolectar

Cada estudiante debe elegir **una** de las siguientes opciones:

| Opción | Medio | Volumen mínimo | Descripción |
| :--- | :--- | :--- | :--- |
| **A** | **La República** | 350 – 400 noticias | Titulares de `larepublica.pe` (~90–100 por categoría). |
| **B** | **El Comercio** | 350 – 400 noticias | Titulares de `elcomercio.pe` (~90–100 por categoría). |
| **C** | **Comparada** | 200 + 200 (400 total) | 200 de cada medio. Permite analizar diferencias de redacción y sesgo. |

> [!IMPORTANT]
> **Categorías fijas y normalizadas (las mismas para todos):**
> `politica`, `economia`, `deportes`, `ciencia_tecnologia`
> 
> Si una noticia no encaja en ninguna de estas 4 categorías, se descarta y se reporta en el notebook. **No inventen una quinta categoría.**

**Columnas obligatorias del CSV (`datos/noticias.csv`):**

| Columna | Significado | Ejemplo ilustrativo |
| :--- | :--- | :--- |
| `titular` | Título principal de la noticia. | *"Congreso aprueba nuevo retiro de fondos de AFP"* |
| `bajada` | Subtítulo, copete o texto breve de 1–2 líneas que va debajo del titular. | *"La medida beneficiará a más de 4 millones de afiliados y pasa al Ejecutivo."* |
| `medio` | Nombre del diario fuente (`La Republica` o `El Comercio`). | `La Republica` |
| `seccion_original` | Nombre de la sección tal como figura en la web/RSS del diario. | `economia-mercados` o `futbol-peruano` |
| `url` | Enlace directo funcional a la noticia. | `https://larepublica.pe/economia/...` |
| `fecha` | Fecha de publicación (`AAAA-MM-DD` o formato estándar del RSS). | `2026-10-11` |
| `etiqueta` | **Categoría estandarizada** (una de las 4 oficiales: `politica`, `economia`, `deportes`, `ciencia_tecnologia`). | `economia` |

---

### 2. Uso de inteligencia artificial asistida

**Pueden usar Claude, OpenCode (Muse), ChatGPT o Copilot.** Para esta tarea el uso de IA es bienvenido, ya que tareas de web scraping y plantillas de UI son repetitivas. Lo que se evalúa estrictamente es la validez técnica de su resultado:

- El CSV debe tener las **7 columnas** y **350+ filas reales** con URLs funcionales.
- La matriz de confusión y el reporte F1 deben calcularse sobre el **conjunto de test reservado**, nunca sobre train.
- El dashboard de Streamlit debe ser funcional y permitir clasificar titulares nuevos en vivo.

> [!WARNING]
> **Fallas comunes de IA generativa sin supervisión:**
> - Scraping que solo corre una vez y se cae por bloqueos o cambios de HTML.
> - Archivos de pesos pesados (`.safetensors`, `.bin`) que superan los 100 MB y rompen el `git push` a GitHub.
> - Archivo `requirements.txt` incompleto con versiones incompatibles.
> - Métricas infladas por evaluar sobre los datos de entrenamiento.

---

### 3. Buenas prácticas de scraping y respeto a los medios

- Revisen y respeten el archivo `robots.txt` de cada medio.
- Configuren un `User-Agent` descriptivo, pausas de **1 a 2 segundos entre solicitudes** y máximo 1 reintento.
- Si un medio bloquea el acceso HTML directo, utilicen sus canales RSS oficiales (`/rss`, `/feed`) y documéntenlo en el notebook. Un bloqueo documentado y resuelto con RSS no baja puntaje; un scraping agresivo sí.

---

### 4. Flujo de trabajo en GitHub y estructura del repositorio

1. Creen una carpeta llamada `tarea_1/` dentro de su repositorio de trabajo del curso.
2. Trabajen en **VS Code con Git** (o GitHub Desktop).
3. Creen **3 Issues en su repositorio** para organizar el avance (usándolos como bitácora de desarrollo):
   - `T1 – Parte 1: Recolección y Limpieza`
   - `T1 – Parte 2: Modelado y Clasificación`
   - `T1 – Parte 3: Dashboard y Despliegue`
4. Usen commits descriptivos y frecuentes a lo largo de los días (ej. `scrape: 400 titulares LR`, `train: distilbert 3 epochs F1 0.81`).
5. Al finalizar cada etapa, **cierren su Issue correspondiente**.

El repositorio debe quedar organizado de la siguiente manera:

```text
repo-del-estudiante/
└── tarea_1/
    ├── 01_recoleccion.ipynb       # Scraping, limpieza y EDA
    ├── 02_clasificacion.ipynb     # Modelado, LLM y evaluación
    ├── app.py                     # Aplicación interactiva Streamlit
    ├── requirements.txt           # Dependencias exactas
    ├── README.md                  # Reporte y guía de ejecución
    ├── .gitignore                 # Exclusión de archivos pesados
    ├── datos/
    │   ├── noticias.csv           # Dataset recolectado (350+ filas)
    │   └── etiquetas_llm.csv      # Clasificaciones generadas por el LLM
    └── modelo/
        └── (configuraciones o artefactos ligeros del modelo)
```

> [!TIP]
> **Manejo de archivos pesados en Git:**
> GitHub rechaza archivos de más de 100 MB. Agreguen a su `.gitignore`:
> ```text
> modelo/*.safetensors
> modelo/*.bin
> modelo/checkpoint-*/
> ```
> Para la calificación, **los 2 notebooks deben entregarse completamente ejecutados** (con salidas, métricas y gráficos visibles directamente en GitHub).

---

## 📊 Criterios de calificación (Sobre 20 puntos)

| Componente | Puntos |
| :--- | :---: |
| **Flujo de trabajo en GitHub** | 2 pts |
| **Parte 1 – Recolección (`01_recoleccion.ipynb`)** | 6 pts |
| **Parte 2 – Clasificación (`02_clasificacion.ipynb`)** | 8 pts |
| **Parte 3 – Dashboard (`app.py` + `README.md`)** | 4 pts |
| **Total** | **20 pts** |

### Desglose del Flujo en GitHub [2 pts]
- Carpeta `tarea_1/` con estructura limpia y ordenada: **0.5 pt**
- 3 Issues creados, documentados como bitácora y cerrados: **0.5 pt**
- Historial de commits distribuidos en el tiempo (no todo el último día): **1.0 pt**

---

# 📥 Parte 1 – Recolección de noticias [6 pts]

### Notebook: `01_recoleccion.ipynb`

**Pasos requeridos:**
1. **Configuración inicial:** Registrar la fecha de descarga, el medio elegido (A, B o C) y las URLs base en la primera celda Markdown.
2. **Extracción:** Extraer mínimo **350 noticias** con las columnas: `titular`, `bajada`, `medio`, `seccion_original`, `url`, `fecha`. Usar RSS o HTML (`requests` + `BeautifulSoup`), implementando pausas entre peticiones y guardando respaldo crudo en `datos/raw/`.
3. **Limpieza y normalización:**
   - Eliminación de duplicados por URL y titulares vacíos.
   - Mapeo explícito de `seccion_original` a las 4 categorías finales (`politica`, `economia`, `deportes`, `ciencia_tecnologia`), mostrando la tabla de equivalencias.
   - Reporte del número de noticias descartadas y motivos.
4. **Análisis Exploratorio (EDA):** Generar **3 gráficos con Plotly Express**:
   - Distribución de noticias por categoría.
   - Distribución de noticias por medio o rango de fechas.
   - Histograma de longitud de caracteres de los titulares.
   - *Cada gráfico debe tener un título que exprese la conclusión analítica, no solo la variable.*
5. **Validación:** Mostrar las 5 noticias con titulares más largos y más cortos, verificando coherencia con el histograma.
6. **Exportación:** Guardar `datos/noticias.csv` (7 columnas) y mostrar `df.shape` junto con las primeras 5 filas.

### Rúbrica Parte 1 [6 pts]

| Criterio | Pts |
| :--- | :---: |
| CSV con 350+ filas reales, 7 columnas, URLs funcionales y reporte de descartes | 2.0 |
| Tabla de mapeo sección → categoría visible + limpieza de duplicados y vacíos | 2.0 |
| 3 gráficos interactivos en Plotly con títulos analíticos + notebook ejecutado | 2.0 |

---

# 🤖 Parte 2 – Del LLM al clasificador [8 pts]

### Notebook: `02_clasificacion.ipynb`

**Pasos requeridos:**
1. **Línea base Zero-Shot (sin fine-tuning):**
   - Evaluar 40 titulares usando `pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")` o `Recognai/bert-base-spanish-wwm-cased-xnli`.
   - Reportar accuracy y F1 macro. Este puntaje representa el piso mínimo a superar.
2. **Supervisión débil con LLM:**
   - Usar un LLM (Ollama, Qwen local, Gemini API o similar) con un prompt estructurado para etiquetar el dataset completo en las 4 categorías.
   - Guardar `datos/etiquetas_llm.csv` con: `titular`, `etiqueta_llm`, `confianza`, `modelo_llm`, `fecha`.
   - **Auditoría humana:** Revisar manualmente **20 predicciones al azar** y reportar el porcentaje de acuerdo (ej. 18/20 = 90%).
3. **Partición de datos (Split 70 / 15 / 15):**
   - Realizar split estratificado con `random_state=42` (Train ~70%, Val ~15%, Test ~15%).
   - El conjunto de **Test** debe permanecer estrictamente aislado hasta la evaluación final.
4. **Fine-Tuning de Transformer:**
   - Modelo recomendado: `distilbert-base-multilingual-cased` o `dccuchile/bert-base-spanish-wwm-cased` (BETO).
   - Configuración mínima con `HuggingFace Trainer`: 3 épocas, batch size de 8 a 16, evaluación por época y métrica de optimización **F1 macro** (`sklearn.metrics.f1_score(average="macro")`).
   - Guardar el modelo en `modelo/`.
   > [!TIP]
   > DistilBERT corre sin problemas en CPU en VS Code. Si su laptop presenta lentitud severa durante el entrenamiento, pueden ejecutar este notebook en **Google Colab (con GPU T4 gratuita)** y luego descargar los resultados.
5. **Evaluación final en Test:**
   - Calcular sobre el test reservado: Accuracy, **F1 Macro** y la **Matriz de Confusión** (con `seaborn`).
   - Incluir una breve reflexión por categoría: identificar cuál tuvo mejor/peor desempeño y explicar la causa lingüística o temática.
6. **Comparativa y decisión:**
   - Tabla comparativa: Zero-shot vs. LLM débil vs. Transformer Fine-tuned.
   - Justificar cuál modelo elegirían para un entorno real considerando precisión, costo computacional y latencia.

### Rúbrica Parte 2 [8 pts]

| Criterio | Pts |
| :--- | :---: |
| Baseline Zero-shot + etiquetas LLM guardadas + auditoría de 20 casos documentada | 2.5 |
| Split 70/15/15 estratificado con semilla fija + Fine-tuning funcional | 2.5 |
| Métricas en test (F1 macro, matriz de confusión) + tabla comparativa y justificación | 3.0 |

---

# 🚀 Parte 3 – Dashboard interactivo [4 pts]

### Archivos: `app.py`, `requirements.txt`, `README.md`

**Pasos requeridos:**
1. **Encabezado y contexto:** Título principal centrado en la pregunta del trabajo y resumen metodológico.
2. **Filtros interactivos en barra lateral:** Filtros por `medio` y `categoría` que actualicen dinámicamente las visualizaciones y la tabla de datos.
3. **Indicadores clave (`st.metric`):**
   - Total de noticias recolectadas.
   - F1 macro obtenido por el modelo en test.
   - Porcentaje de concordancia LLM vs. auditoría humana.
4. **Visualización y exploración de datos:**
   - Gráfico Plotly interactivo con la distribución de categorías filtradas.
   - `st.dataframe` interactivo con opción de descarga en CSV.
5. **Probador interactivo en vivo (Core de la demo):**
   - Campo de texto donde el usuario escribe o pega un titular nuevo.
   - Al presionar un botón, el clasificador predice la categoría y muestra la probabilidad asociada.
6. **Despliegue y documentación:**
   - `requirements.txt` con todas las dependencias necesarias.
   - `README.md` que detalle: pregunta del proyecto, medios y fechas de extracción, 2 hallazgos analíticos, 2 limitaciones técnicas y comandos exactos para ejecutar en local (`streamlit run app.py`).
   - Despliegue opcional pero recomendado en Streamlit Community Cloud (`share.streamlit.io`).

### Rúbrica Parte 3 [4 pts]

| Criterio | Pts |
| :--- | :---: |
| App funcional en local/incógnito (filtros, métricas, gráfico, tabla y probador en vivo) | 2.5 |
| `README.md` completo con las 5 secciones requeridas + `requirements.txt` reproducible | 1.5 |

---

## 💡 Ideas de innovación (Puntos de bonificación opcionales)

- **Análisis de sesgo comparativo (Opción C):** Evaluar cómo titulan ambos medios la misma noticia y si el clasificador altera su predicción.
- **NER peruano:** Extracción de entidades nombradas (políticos, instituciones, empresas) más frecuentes por categoría usando `mrm8488/bert-spanish-cased-finetuned-ner`.
- **Buscador semántico:** Búsqueda por similitud coseno de embeddings para recuperar noticias similares a un titular consultado.
- **Video demostrativo:** Grabación de 2 a 3 minutos explicando los hallazgos y mostrando la aplicación en funcionamiento.

---

## 📦 Lista de verificación antes de entregar

- [ ] La carpeta `tarea_1/` contiene: `01_recoleccion.ipynb`, `02_clasificacion.ipynb`, `app.py`, `requirements.txt`, `README.md`, `datos/noticias.csv` y `datos/etiquetas_llm.csv`.
- [ ] Los 2 notebooks están **completamente ejecutados** y sus salidas son visibles en GitHub.
- [ ] Los 3 Issues creados fueron documentados y están cerrados.
- [ ] Los commits reflejan trabajo progresivo a lo largo de los días.
- [ ] No hay archivos pesados (mayores a 100 MB) ni credenciales/claves API subidas al repositorio.
- [ ] La app de Streamlit se ejecuta sin errores con `streamlit run app.py`.

---

## 📬 Formato y canal de entrega

Publicar un comentario en este Issue con el siguiente formato:

```text
Tarea 1 – Individual
Medio: [La República / El Comercio / Comparada]
Estudiante: Nombre Completo (@usuario_github)
Dashboard en vivo: https://...streamlit.app  (o indicar ejecución local comprobada)
Carpeta del trabajo: https://github.com/.../tarea_1
Resultados Test: F1 Macro: 0.xx | Zero-shot: 0.xx | Acuerdo LLM-Humano: xx/20
```

> [!NOTE]
> **¿De dónde obtienen estos resultados?**
> Se copian directamente de las salidas impresas en su notebook `02_clasificacion.ipynb`:
> - `F1 Macro`: Métrica obtenida al evaluar el modelo entrenado sobre el conjunto de test (Paso 5).
> - `Zero-shot`: Métrica F1 del modelo base sin entrenar (Paso 1).
> - `Acuerdo LLM-Humano`: Número de aciertos en la auditoría manual de 20 casos (Paso 2).

### ⏰ Fecha límite de entrega

**Domingo 18 de octubre a las 23:59 (hora de Perú / UTC-5)**.
