# Challenge: Telecom X - Predicción de Abandono con Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-Completado-brightgreen" alt="Estado del Proyecto">
  <img src="https://img.shields.io/badge/Tecnología-Python-blue" alt="Tecnología Python">
  <img src="https://img.shields.io/badge/Entorno-Google%20Colab-yellow" alt="Entorno Google Colab">
  <img src="https://img.shields.io/badge/Bibliotecas-Pandas%20%7C%20XGBoost%20%7C%20Sklearn-red" alt="Bibliotecas">
</p>

---

### Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Preparación de Datos](#preparación-de-datos)
3. [Funciones de Evaluación y Feature Importance](#funciones-de-evaluación-y-feature-importance)
4. [Análisis Comparativo de Modelos](#análisis-comparativo-de-modelos)
5. [Modelo Champion: XGBoost + RFECV](#modelo-champion-xgboost--rfecv)
6. [Estrategias de Retención Propuestas](#estrategias-de-retención-propuestas)
7. [Estructura del Proyecto](#estructura-del-proyecto)
8. [Tecnologías y Dependencias](#tecnologías-y-dependencias)
9. [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)

---

### Descripción del Proyecto
Este proyecto es parte del **Programa ONE - Oracle Next Education** en colaboración con **Alura**, y corresponde al **Challenge: Telecom X Parte 2**, enfocado en profundizar en el análisis de datos mediante la implementación de modelos de **Machine Learning** utilizando Python.

El objetivo principal es desarrollar y comparar diversos algoritmos de clasificación para predecir el abandono de clientes (**Churn**) en la empresa **Telecom X**. A través del modelado predictivo, se busca proporcionar una herramienta que identifique patrones de fuga y permita a la gerencia anticiparse a la cancelación de servicios, optimizando las estrategias de retención y fortaleciendo la sostenibilidad del negocio mediante decisiones basadas en datos.

---

### Preparación de Datos

Se realizó una refinación técnica del set de datos para mejorar el entrenamiento:
* **Eliminación de Irrelevantes:** Se descartó la columna `id_cliente` por ser un identificador único que no aporta información a los patrones de comportamiento.
* **Reducción por Correlación:** Tras analizar la colinealidad, se eliminaron las variables `genero`, `servicio_telefono`, `cobro_diario` y `cobro_total` para reducir el ruido y evitar el sobreajuste.
* **Codificación:** Implementación de `OneHotEncoder` para transformar variables categóricas en formatos procesables por los modelos y `LabelEncoder` para la variable objetivo.

---

### Funciones de Evaluación y Feature Importance

Para garantizar la eficiencia y transparencia del modelado, se crearon las siguientes herramientas:

* **`pipeline_balanceo`**: Función que integra el balanceador (SMOTE o NearMiss) con el modelo, asegurando que el remuestreo se ejecute correctamente dentro de la validación cruzada.
* **`evaluar_modelo_cv`**: Centraliza la evaluación mediante **Validación Cruzada Estratificada**, generando reportes de métricas y matrices de confusión automáticas.
* **`importancia_variables`**: Función personalizada para extraer y visualizar el peso predictivo de cada característica en los modelos finales.

---
### Análisis Comparativo de Modelos

Se evaluó el rendimiento de los algoritmos bajo tres condiciones: **sin sampleo**, con **SMOTE** (Oversampling) y con **NearMiss v3** (Undersampling). A continuación se detallan los resultados obtenidos mediante validación cruzada:

* **Dummy Classifier (Baseline):** Utilizado para establecer los valores mínimos de rendimiento. Obtuvo un **recall de 0%** para la clase de interés (Churn), confirmando que el desbalance de datos imposibilita la predicción sin un modelo predictivo real.
* **Decision Tree Classifier:** Se evaluaron tres versiones. La versión original sin sampleo alcanzó un accuracy del 80% pero un recall limitado (57%). Al aplicar **SMOTE** y **NearMiss**, el recall subió a 65%-66%, pero a costa de una caída significativa en la precisión (54% y 51% respectivamente).
* **Random Forest Classifier:** Fue el algoritmo más balanceado en sus métricas generales. La variante con **SMOTE** destacó con un **F1-score de 0.62** y un recall de 65%. A pesar de estos buenos números, se buscaba un modelo con mayor capacidad de detección de fugas.
* **Selección Final (XGBoost):** Este modelo fue seleccionado debido a su superioridad técnica en la métrica más crítica para el negocio: el **Recall**. En validación cruzada, alcanzó un **80% de Recall** para la clase Churn. Esto significa que es capaz de identificar correctamente a 8 de cada 10 clientes que abandonarán la empresa, superando ampliamente a los modelos de bosque aleatorio y árboles de decisión.

---

### Modelo Champion: XGBoost + RFECV

Al ser seleccionado como el mejor modelo, se procedió a su optimización mediante **RFECV** (Eliminación Recursiva de Variables con Validación Cruzada), reduciendo el set de datos de 23 a **20 variables clave**.

La comparación final sobre el **set de test (datos nunca vistos por el modelo)** confirmó la robustez de la optimización:

![Gráfico: Comparación modelos XGBoost y XGBoost+RFECV mediante matriz de confusión](https://raw.githubusercontent.com/iesvs-campi/oracle-one-challenge-telecom-x-2/refs/heads/main/plot_final/comparacion_matriz_confusion_modelos_finales.png)

| Métrica (Clase Churn) | XGBoost Original (23 Var) | XGBoost + RFECV (20 Var) |
| :--- | :---: | :---: |
| **Recall (Detección de fugas)** | 78% | **79%** |
| **Precisión** | 51% | **52%** |
| **F1-Score** | 0.62 | **0.62** |
| **Accuracy General** | 75% | 75% |

**Conclusión del Modelado:** La aplicación de RFECV no solo simplificó el modelo eliminando 3 variables de ruido, sino que incrementó la capacidad de detección (Recall) y la precisión sobre datos reales, consolidándolo como una herramienta predictiva confiable para Telecom X.

#### Variables más influyentes (Feature Importance):
El modelo final reveló que los principales disparadores del abandono son factores contractuales y tecnológicos, mientras que la **antigüedad** resultó ser menos determinante de lo previsto inicialmente (6° lugar):

1. **Tipo de Contrato (Mes a mes):** 65.07% de importancia.
2. **Servicio de Internet (Fibra óptica):** 12.31% de importancia.
3. **Tipo de Contrato (Dos años):** 2.63% de importancia.

---

### Estrategias de Retención Propuestas

Basándose en los hallazgos finales del modelo, se proponen:

* **Plan de Migración Contractual:** Diseñar incentivos agresivos para trasladar a los clientes del contrato "Mes a mes" hacia modalidades de permanencia anual. Atacar este punto impactaría directamente sobre el factor de riesgo del 65%.
* **Control de Calidad en Fibra Óptica:** Investigar la causa raíz de la insatisfacción en este segmento tecnológico para equilibrar la relación costo-beneficio percibida por el cliente.
* **Fomento de la Automatización:** Implementar bonificaciones por la migración de pagos manuales (Cheque electrónico) hacia débito automático para reducir la fricción en el proceso de cobro.
* **Fortalecimiento de Servicios de Valor Agregado:** Incentivar el uso de *Soporte Técnico* y *Seguridad Online*, ya que el modelo identifica que los clientes con estos servicios activos presentan una lealtad superior.

---
### Estructura del Proyecto

* **`Challenge: Telecom X Parte 2.ipynb`**: Notebook principal con el flujo completo de Machine Learning, desde el balanceo de clases hasta la selección del modelo final.
* **`datos_tratados.csv`**: Dataset procesado utilizado para el entrenamiento y prueba de los modelos predictivos.
* **`plot_final/`**: Carpeta que contiene la comparativa visual de las matrices de confusión de XGBoost (Original vs. RFECV) sobre los datos de test.
* **`README.md`**: Documentación técnica detallada y resumen ejecutivo de los modelos.
* **`LICENSE`**: Licencia de uso MIT.

### Tecnologías y Dependencias

* **Python**
* **Pandas / Numpy**: Manipulación de datos y manejo de arreglos numéricos.
* **XGBoost**: Algoritmo Champion de Gradient Boosting para clasificación.
* **Scikit-Learn**: Procesamiento de datos, métricas y validación cruzada.
* **Imbalanced-learn**: Implementación de técnicas de balanceo (SMOTE y NearMiss).
* **Seaborn / Matplotlib**: Visualización de rendimiento y matrices de confusión.
* **Google Colab**: Entorno de desarrollo en la nube.

### Cómo Ejecutar el Proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/iesvs-campi/oracle-one-challenge-telecom-x-2.git
   cd oracle-one-challenge-telecom-x-2
   ```

2. **Ejecución:**
* Abre el archivo `Challenge: Telecom X Parte 2.ipynb` en **Google Colab** o **Jupyter Notebook**.
* Asegúrate de que el archivo `datos_tratados.csv` esté en la misma ruta que el notebook.
* Ejecuta las celdas secuencialmente. El notebook está diseñado para realizar la comparativa automatizada de modelos y finalizar con la optimización mediante **RFECV**.
