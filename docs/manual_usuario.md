# Manual de Usuario - Sistema SomnoScope

## Introducción
**SomnoScope** es una plataforma clínica interactiva de apoyo al diagnóstico asistido por inteligencia artificial, diseñada para la clasificación y estadificación automática de registros polisomnográficos (PSG) nocturnos a partir de señales de electroencefalografía (EEG) de un solo canal.

El sistema procesa registros conforme al estándar clínico de la Academia Americana de Medicina del Sueño (**AASM**), clasificando cada época de 30 segundos en cinco estadios fisiológicos:
* **W (Wake):** Vigilia activa o relajada.
* **N1 (Non-REM Stage 1):** Transición de sueño ligero o somnolencia.
* **N2 (Non-REM Stage 2):** Sueño intermedio caracterizado por husos de sueño y complejos K.
* **N3 (Non-REM Stage 3 / SWS):** Sueño profundo de ondas lentas delta.
* **REM (Rapid Eye Movement):** Sueño paradójico con atonía muscular y ensoñaciones vívidas.

---

## Acceso al Sistema
Para acceder a la plataforma web de SomnoScope:
* **Entorno local:** Abra su navegador web (Google Chrome, Mozilla Firefox o Microsoft Edge) y navegue a la dirección: `http://localhost:8050`.
* **Entorno en la nube (AWS EC2):** Acceda mediante la dirección IP pública y el puerto configurados en el despliegue del servidor: `http://18.212.239.119:8050`.

Al cargar la aplicación, se desplegará el encabezado clínico de SomnoScope con selector de tema de iluminación, resumen del modelo activo y el panel lateral de ingesta.

---

## Preparación de los Datos
Para garantizar un procesamiento e inferencia correctos, el estudio debe cumplir con las siguientes especificaciones técnicas:

1. **Formato de archivo:** Archivos en formato estándar biomédico European Data Format (`.edf` o `.rec`), compatibles con el estándar internacional Sleep-EDFx.
2. **Canales de señal requeridos:** Registros que contengan al menos una derivación electroencefalográfica estándar, preferiblemente `EEG Fpz-Cz` o `EEG Pz-Oz`.
3. **Frecuencia de muestreo:** Señales adquiridas a 100 Hz (frecuencia nativa de Sleep-EDF) o preprocesadas con filtro antialiasing.
4. **Segmentación:** El sistema segmenta internamente la señal en épocas discretas de 30 segundos (3,000 muestras a 100 Hz).
5. **Anotaciones de referencia (Opcional):** Si dispone del archivo de anotaciones médicas del hipnograma real (`-Hypnogram.edf`), cárguelo conjuntamente para habilitar la comparación en tiempo real y la matriz de confusión.

---

## Carga y Análisis del Estudio

1. **Selección de la fuente de datos (Barra lateral izquierda):**
   * **Modo Estudio Pregrabado:** Seleccione un paciente del menú desplegable de muestras de validación integradas en el sistema (por ejemplo, `SC4001E0 - Sujeto Control Sano`).
   * **Modo Carga Externa:** Haga clic en el botón de carga (*Browse files*) para subir un nuevo archivo `.edf` de PSG desde su ordenador local.

2. **Selección del canal de análisis:**
   Elija la derivación de trabajo (`EEG Fpz-Cz` por defecto).

3. **Ejecución de la inferencia:**
   El tablero envía la solicitud automáticamente a la **API REST desacoplada (FastAPI)** mediante una llamada HTTP POST segura al endpoint `/predict/recording`. Si la API remota se encontrara en mantenimiento, SomnoScope conmuta de forma transparente al motor de inferencia local empaquetado para asegurar alta disponibilidad.

---

## Interpretación de Resultados Clínicos

Una vez procesado el estudio, la pantalla principal desplegará los componentes analíticos organizados en secciones:

### 1. Tarjetas de Indicadores Clave de Desempeño (KPIs Clínicos)
En la parte superior se presentan cinco métricas cuantitativas estándar de la arquitectura nocturna del paciente:
* **TIB (Time in Bed):** Tiempo total que el paciente permaneció en cama durante el estudio (expresado en horas y minutos).
* **TST (Total Sleep Time):** Tiempo total real de sueño efectivo acumulado (suma de N1, N2, N3 y REM).
* **Eficiencia de Sueño (%):** Relación porcentual entre TST y TIB. Un valor $\ge 85\%$ se considera dentro de los límites clínicos normales.
* **SOL (Sleep Onset Latency):** Tiempo transcurrido desde el inicio de la grabación hasta la primera época de sueño consolidado.
* **WASO (Wake After Sleep Onset):** Minutos totales de vigilia experimentados tras el inicio del sueño, indicador clave de fragmentación o insomnio.

### 2. Hipnograma Dual Nocturno Interactivo
Gráfico temporal continuo que ilustra la evolución cronológica del sueño durante toda la noche:
* **Franja Superior (Predicho por IA):** Estadificación automática generada por el modelo supervisado optimizado (LightGBM).
* **Franja Inferior (Anotación Médica / Ground Truth):** Clasificación de referencia realizada por el especialista clínico.
* **Código de colores AASM:**
  * 🔴 **Vigilia (W):** Rojo / Coral.
  * 🟣 **N1:** Lavanda claro.
  * 🔵 **N2:** Azul índigo.
  * 🔷 **N3:** Azul marino profundo.
  * 🟢 **REM:** Verde menta esmeralda.

### 3. Inspector Detallado de Época (Ventana de 30 Segundos)
Permite realizar una auditoría microscópica de cualquier época del estudio mediante un control deslizante (*slider*):
* **Trazado de la señal cruda EEG:** Gráfico de amplitud en microvoltios ($\mu V$) en función del tiempo (0 a 30 segundos), permitiendo identificar husos de sueño, complejos K o ritmos alfa de vigilia.
* **Distribución de probabilidades:** Gráfico de barras horizontales que desglosa el vector de certidumbre del modelo para cada una de las 5 clases (W, N1, N2, N3, REM), brindando total interpretabilidad al especialista.

### 4. Evaluación Diagnóstica y Matriz de Confusión
Si se cargaron anotaciones de referencia, el sistema despliega:
* **Matriz de confusión normalizada:** Detalla los aciertos diagnósticos y los patrones de confusión entre estadios (por ejemplo, transiciones entre N1 y vigilia o N2 y sueño profundo).
* **Tabla de métricas por clase:** Precisión, Sensibilidad (*Recall*) y F1-Score desglosados por estadio clínico.

---

## Acciones Adicionales y Personalización

* **Alternador de Tema (Modo Claro / Modo Oscuro):**
  En el panel lateral encontrará un selector para alternar entre el **Modo Claro** (recomendado para lectura clínica diurna y reportes impresos) y el **Modo Oscuro** (optimizado para salas de monitoreo nocturno o lectura prolongada con fatiga visual reducida).
* **Exportación de Datos:**
  El especialista puede exportar la serie temporal de estadios predichos en formato tabular para su integración con sistemas de historia clínica electrónica (EHR).

---

## Consideraciones Clínicas Importantes
* **Herramienta de asistencia diagnóstica:** SomnoScope está concebido como una plataforma de triaje y soporte analítico para médicos neurofisiólogos, somnólogos y personal de salud calificado. No sustituye el juicio médico profesional ni constituye por sí sola una orden terapéutica.
* **Calidad de la señal EEG:** Artefactos de movimiento excesivo, desconexión de electrodos o impedancias elevadas pueden degradar la precisión diagnóstica del modelo. Se recomienda inspeccionar visualmente la señal en el visualizador de épocas antes de validar el diagnóstico.
* **Monitoreo del estadio N1:** Debido a la naturaleza transitoria y de bajo voltaje del estadio N1, la sensibilidad algorítmica es naturalmente menor que en N2 o N3; se recomienda atención especial en dichas épocas.

---

## Solución de Problemas Frecuentes

| Problema | Causa Probable | Solución Recomendada |
| :--- | :--- | :--- |
| **Error al cargar archivo `.edf`** | Formato de archivo corrupto o canal EEG con nomenclatura no estándar. | Asegúrese de que el archivo cumpla con la especificación European Data Format y contenga la derivación Fpz-Cz o Pz-Oz. |
| **Aviso de advertencia de API REST** | La API en el puerto 8000 está iniciando o inalcanzable. | El tablero activará automáticamente el motor local. Si desea reconectar la API, verifique `docker compose ps` en el servidor. |
| **Lentitud al renderizar el hipnograma** | El estudio completo contiene más de 2,500 épocas (más de 20 horas de grabación continua). | Filtre el rango temporal deseado o verifique que el navegador tenga habilitada la aceleración gráfica por hardware. |
