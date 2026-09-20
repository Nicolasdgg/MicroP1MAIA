# Manual de Instalación - Sistema SomnoScope

## Requisitos Previos
El sistema SomnoScope requiere un entorno compatible con Linux, macOS o Windows (WSL2 recomendado). Para el funcionamiento óptimo de los servicios, se establecen los siguientes requisitos:

* **Software base:** Python 3.10 o superior, Git (gestor de control de versiones) y Docker Engine con Docker Compose v2.
* **Capacidad de cómputo recomendada:** Mínimo 4 GB de memoria RAM (o 2 GB de RAM física con 2 GB adicionales de espacio Swap configurado) y al menos 5 GB de espacio libre en disco para imágenes de contenedores, pesos de modelos y registros de polisomnografía.
* **Puertos de red requeridos:**
  * Puerto `8000`: API REST de Inferencia (FastAPI).
  * Puerto `8050`: Tablero Clínico Interactivo (Streamlit).
  * Puerto `5000`: Servidor de Seguimiento de Experimentos (MLflow).

---

## Instalación y Clonación

1. **Clonar el repositorio oficial del proyecto:**
   Clone el repositorio desde GitHub navegando al directorio de su preferencia en la terminal y verificando la rama principal (`main`):
   ```bash
   git clone https://github.com/Nicolasdgg/MicroP1MAIA.git
   cd MicroP1MAIA
   git checkout main
   ```

2. **Configuración de credenciales de AWS (para almacenamiento remoto DVC):**
   Para sincronizar conjuntos de datos crudos y modelos almacenados en Amazon S3, configure las credenciales de AWS ejecutando:
   ```bash
   aws configure
   ```
   Ingrese su `AWS Access Key ID`, `AWS Secret Access Key` y la región por defecto (`us-east-1`).

3. **Descarga de datos y modelos con DVC:**
   Una vez configuradas las credenciales, descargue los registros de sueño y artefactos serializados:
   ```bash
   dvc pull
   ```

---

## Configuración del Entorno
SomnoScope utiliza variables de entorno preconfiguradas para entornos de desarrollo y producción. Puede personalizar el comportamiento creando un archivo `.env` en la raíz del proyecto:

```env
# URL de comunicación entre el tablero y la API REST
API_URL=http://api:8000

# Parámetros del servidor API
API_HOST=0.0.0.0
API_PORT=8000

# Parámetros del tablero Streamlit
STREAMLIT_SERVER_PORT=8050
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Parámetros de MLflow
MLFLOW_TRACKING_URI=http://0.0.0.0:5000
```

---

## Inicio del Sistema con Contenedores Docker (Recomendado)
El proyecto incluye un `Dockerfile` optimizado (que instala PyTorch compilado exclusivamente para CPU, evitando descargas masivas de librerías CUDA) y un archivo `docker-compose.yml` que orquesta los tres microservicios en una red aislada (`microp1maia_default`).

1. **Construir las imágenes de los contenedores:**
   ```bash
   docker compose build
   ```

2. **Iniciar los servicios en segundo plano (Modo Daemon):**
   ```bash
   docker compose up -d
   ```

3. **Verificar el estado de los contenedores:**
   ```bash
   docker compose ps
   ```
   Deberá observar los tres contenedores en estado `Up`:
   * `somnoscope-api` (FastAPI en puerto 8000, con healthcheck activo).
   * `somnoscope-dashboard` (Streamlit en puerto 8050).
   * `somnoscope-mlflow` (MLflow Tracking en puerto 5000).

---

## Inicio Local sin Contenedores (Alternativa)
Si desea ejecutar la solución directamente sobre un entorno virtual de Python:

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# .\venv\Scripts\activate   # En Windows PowerShell

# Instalar dependencias base
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Terminal 1: Iniciar API REST
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Iniciar Tablero Clínico
streamlit run app/dashboard.py --server.port 8050

# Terminal 3: Iniciar MLflow UI
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

---

## Verificación de la Instalación
Para comprobar que los servicios están operando correctamente, verifique los siguientes puntos de acceso desde su navegador o consola:

1. **Estado de salud de la API REST:**
   Acceda a `http://localhost:8000/health` o ejecute:
   ```bash
   curl -s http://localhost:8000/health
   ```
   Respuesta esperada en formato JSON:
   ```json
   {"status":"healthy","model":"LightGBM (Optimizado)","classes":["W","N1","N2","N3","REM"],"n_features_expected":25}
   ```
   La documentación interactiva OpenAPI (Swagger UI) está disponible en `http://localhost:8000/docs`.

2. **Acceso al Tablero SomnoScope:**
   Abra en el navegador `http://localhost:8050`. Verá la interfaz con el visualizador de hipnogramas, panel de ingesta y métricas de arquitectura de sueño.

3. **Acceso a la interfaz de MLflow:**
   Abra `http://localhost:5000`. Verá el experimento `Microproyecto_Sleep_Staging` con el registro de corridas y parámetros de los modelos supervisados.

---

## Despliegue en Amazon Web Services (AWS EC2)

1. **Instancia virtual:**
   Se recomienda una instancia `t3.small` o superior con sistema operativo Ubuntu 24.04 LTS y volumen EBS de 24 GB.

2. **Configuración del Grupo de Seguridad (Security Group):**
   Asegúrese de habilitar las siguientes reglas de entrada (*Inbound Rules*):
   * `TCP 22`: Acceso SSH para administración remota.
   * `TCP 8050`: Acceso público al Tablero Streamlit.
   * `TCP 5000`: Acceso público al servidor de seguimiento MLflow.
   * `TCP 8000`: Acceso a la API REST (o consumo interno mediante la red Docker).

3. **Instalación de Docker en la máquina virtual EC2:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-v2
   sudo usermod -aG docker ubuntu
   ```

4. **Configuración de memoria Swap (Recomendada para t3.small):**
   ```bash
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

5. **Despliegue de contenedores en EC2:**
   ```bash
   cd MicroP1MAIA
   docker compose up -d
   ```

---

## Solución de Problemas Frecuentes

* **Conflicto de puertos (`address already in use`):**
  Si el puerto 8000, 8050 o 5000 está ocupado por un proceso previo, identifíquelo y deténgalo:
  ```bash
  sudo lsof -i :8050
  sudo kill -9 <PID>
  ```
* **Memoria insuficiente durante la compilación:**
  Asegúrese de que el archivo swap esté activo (`swapon --show`).
* **Inspección de registros de error (Logs):**
  Para consultar los logs en tiempo real de cualquiera de los microservicios:
  ```bash
  docker compose logs -f api
  docker compose logs -f dashboard
  docker compose logs -f mlflow
  ```
