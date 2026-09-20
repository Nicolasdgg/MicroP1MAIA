import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def style_heading(p, text):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = 'Montserrat SemiBold'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x5b, 0x57, 0xd1)
    return p

def style_subheading(p, text):
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = 'Montserrat SemiBold'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    return p

def style_body(p, bold_prefix, text):
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = 'Montserrat SemiBold'
        r_bold.font.size = Pt(9.5)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    r_text = p.add_run(text)
    r_text.font.name = 'Montserrat Light'
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def style_bullet(p, bold_prefix, text):
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.line_spacing = 1.15
    r_bullet = p.add_run("▪  ")
    r_bullet.font.name = 'Montserrat SemiBold'
    r_bullet.font.size = Pt(9)
    r_bullet.font.color.rgb = RGBColor(0x5b, 0x57, 0xd1)
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = 'Montserrat SemiBold'
        r_bold.font.size = Pt(9.5)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    r_text = p.add_run(text)
    r_text.font.name = 'Montserrat Light'
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def style_code(p, code_text):
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.25)
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
    return p

def add_sections_to_report(doc_path):
    doc = docx.Document(doc_path)
    
    # Locate index of Colaboradores
    target_idx = None
    for i, p in enumerate(doc.paragraphs):
        if 'Colaboradores' in p.text:
            target_idx = i
            break
            
    if target_idx is None:
        raise ValueError("Could not find 'Colaboradores' heading in document")
        
    target_p = doc.paragraphs[target_idx]
    
    # Helper to insert paragraph before target
    def insert_p():
        return target_p.insert_paragraph_before()
        
    # --- SECCIÓN 1: ARQUITECTURA DE MICROSERVICIOS, API REST Y DOCKER ---
    style_heading(insert_p(), "Despliegue y Arquitectura de Microservicios: API REST, Docker y DVC")
    
    style_body(insert_p(), "Modelos empaquetados y servicio desacoplado vía API REST: ",
               "Con el objetivo de garantizar un desacoplamiento estricto entre el motor algorítmico y la interfaz de usuario, se construyó una API REST de nivel de producción utilizando el framework FastAPI. El modelo supervisado óptimo (LightGBM optimizado con 150 estimadores y balanceo de pesos) y el estandarizador espectral fueron empaquetados y serializados mediante joblib en la ruta 'models/best_sleep_model.pkl'. La API implementa esquemas de validación tipados mediante Pydantic y expone cuatro endpoints especializados:")
               
    style_bullet(insert_p(), "GET /health: ", "Endpoint de monitoreo y healthcheck continuo que verifica el estado operativo del servicio, las 5 clases objetivo AASM (W, N1, N2, N3, REM) y las 25 características espectrales esperadas.")
    style_bullet(insert_p(), "POST /predict/features: ", "Recibe un vector normalizado de 25 características espectrales calculadas para una época y retorna la clase diagnóstica predicha junto con la distribución probabilística calibrada.")
    style_bullet(insert_p(), "POST /predict/epoch: ", "Recibe una ventana cruda de 30 segundos de señal electroencefalográfica (3,000 muestras a 100 Hz), ejecuta la extracción espectral en tiempo real y clasifica la época.")
    style_bullet(insert_p(), "POST /predict/recording: ", "Permite la ingesta directa de un archivo polisomnográfico completo en formato European Data Format (.edf), procesando la noche completa para generar el hipnograma continuo y los KPIs clínicos de sueño.")

    style_body(insert_p(), "Contenedorización integral con Docker Compose: ",
               "Para asegurar la reproducibilidad completa del entorno y eliminar problemas de dependencias entre entornos de desarrollo y producción, se construyó un Dockerfile multietapa optimizado sobre 'python:3.12-slim'. Para evitar el sobrecosto de más de 3 GB de librerías CUDA innecesarias para inferencia, se instaló la versión de PyTorch compilada exclusivamente para arquitectura CPU (180 MB). A través del archivo 'docker-compose.yml', se orquestan tres contenedores interconectados en la red aislada 'microp1maia_default': 'somnoscope-api' (FastAPI en puerto 8000), 'somnoscope-dashboard' (Streamlit en puerto 8050) y 'somnoscope-mlflow' (MLflow en puerto 5000 con base SQLite persistente en volumen Docker).")

    style_body(insert_p(), "Versionamiento de datos y modelos con DVC y Amazon S3: ",
               "Siguiendo los lineamientos de MLOps del proyecto, los conjuntos de datos masivos de polisomnografía (Sleep-EDFx, 729 MB distribuidos en 39 archivos) se encuentran versionados mediante DVC (Data Version Control) en el archivo de control 'data.dvc', respaldados en un bucket seguro de Amazon S3 ('s3://microproyecto-maia'). Esta configuración mantiene el repositorio Git ágil y libre de binarios masivos, permitiendo la sincronización inmediata del dataset crudo en cualquier instancia mediante 'dvc pull'.")

    # --- SECCIÓN 2: MANUAL DE INSTALACIÓN ---
    style_heading(insert_p(), "Manual de Instalación - Tablero SomnoScope")
    
    style_subheading(insert_p(), "Requisitos Previos")
    style_body(insert_p(), "", "El sistema SomnoScope requiere un entorno compatible con Linux, macOS o Windows (WSL2 recomendado). Especificaciones requeridas:")
    style_bullet(insert_p(), "Hardware: ", "Mínimo 4 GB de memoria RAM disponible (o 2 GB de RAM física con 2 GB adicionales de memoria Swap activa) y al menos 5 GB de espacio libre en disco.")
    style_bullet(insert_p(), "Software base: ", "Python 3.10 o superior, Git, Docker Engine y Docker Compose v2 instalados y operativos.")
    style_bullet(insert_p(), "Puertos de red: ", "Disponibilidad de los puertos TCP 8000 (API REST), 8050 (Tablero Streamlit) y 5000 (MLflow).")

    style_subheading(insert_p(), "Instalación y Clonación del Repositorio")
    style_body(insert_p(), "1. Clonación oficial: ", "Clone el repositorio oficial del proyecto alojado en GitHub y sitúese en la rama principal 'main':")
    style_code(insert_p(), "git clone https://github.com/Nicolasdgg/MicroP1MAIA.git\ncd MicroP1MAIA\ngit checkout main")
    style_body(insert_p(), "2. Descarga de datos y modelos (DVC): ", "Configure las credenciales de AWS ejecutando 'aws configure' y descargue los registros de sueño:")
    style_code(insert_p(), "dvc pull")

    style_subheading(insert_p(), "Inicio del Sistema con Docker Compose (Recomendado)")
    style_body(insert_p(), "", "Para inicializar de manera desatendida los tres microservicios (API, Dashboard y MLflow), ejecute:")
    style_code(insert_p(), "docker compose build\ndocker compose up -d")
    style_body(insert_p(), "Verificación de contenedores: ", "Compruebe que los servicios estén activos ejecutando 'docker compose ps'. Los contenedores 'somnoscope-api', 'somnoscope-dashboard' y 'somnoscope-mlflow' deben mostrar estado 'Up'.")

    style_subheading(insert_p(), "Verificación del Funcionamiento")
    style_body(insert_p(), "API REST (Puerto 8000): ", "Compruebe la salud de la API ejecutando 'curl -s http://localhost:8000/health'. La API responderá con estado 'healthy' y la lista de estadios. La interfaz interactiva Swagger UI está disponible en 'http://localhost:8000/docs'.")
    style_body(insert_p(), "Tablero SomnoScope (Puerto 8050): ", "Acceda desde el navegador a 'http://localhost:8050'.")
    style_body(insert_p(), "Servidor MLflow (Puerto 5000): ", "Acceda desde el navegador a 'http://localhost:5000'.")

    style_subheading(insert_p(), "Despliegue en Amazon Web Services (AWS EC2)")
    style_body(insert_p(), "1. Configuración de Red: ", "En la consola de AWS EC2, configure el Security Group de la instancia habilitando reglas de entrada (Inbound) para los puertos TCP 22 (SSH), 8050 (Dashboard), 5000 (MLflow) y 8000 (API).")
    style_body(insert_p(), "2. Preparación de la Máquina Virtual: ", "Conéctese por SSH e instale Docker y active la memoria Swap:")
    style_code(insert_p(), "sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2\nsudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
    style_body(insert_p(), "3. Despliegue en la Nube: ", "Clone el repositorio en la máquina virtual y levante los servicios:")
    style_code(insert_p(), "git clone https://github.com/Nicolasdgg/MicroP1MAIA.git && cd MicroP1MAIA && git checkout main\ndocker compose up -d")

    style_subheading(insert_p(), "Solución de Problemas")
    style_bullet(insert_p(), "Conflicto de puertos: ", "Si el puerto 8050 o 5000 está ocupado, detenga el proceso previo con 'sudo lsof -i :8050' y 'sudo kill -9 <PID>'.")
    style_bullet(insert_p(), "Inspección de registros: ", "Consulte los logs en tiempo real con 'docker compose logs -f dashboard' o 'docker compose logs -f api'.")

    # --- SECCIÓN 3: MANUAL DE USUARIO ---
    style_heading(insert_p(), "Manual de Usuario - Sistema SomnoScope")
    
    style_subheading(insert_p(), "Introducción y Acceso al Sistema")
    style_body(insert_p(), "", "SomnoScope es una estación de trabajo analítica diseñada para apoyar a neurólogos y somnólogos en la estadificación automática de polisomnografías conforme al estándar AASM (W, N1, N2, N3, REM).")
    style_bullet(insert_p(), "Acceso Local: ", "Abra su navegador web y diríjase a 'http://localhost:8050'.")
    style_bullet(insert_p(), "Acceso en la Nube (AWS EC2): ", "Navegue a 'http://18.212.239.119:8050' (IP pública de la instancia de cómputo en AWS).")

    style_subheading(insert_p(), "Preparación y Carga de Datos")
    style_body(insert_p(), "Requisitos del estudio: ", "El sistema procesa archivos biomédicos en formato European Data Format (.edf) correspondientes a estudios PSG nocturnos con canal electroencefalográfico EEG Fpz-Cz o Pz-Oz a 100 Hz.")
    style_body(insert_p(), "Procedimiento de carga: ", "En el panel lateral izquierdo, seleccione un paciente precargado de validación clínica (ej. 'SC4001E0 - Sujeto Control') o utilice el control de carga para subir un archivo .edf externo. Seleccione la derivación EEG de interés y el sistema enviará la solicitud de inferencia automáticamente a la API REST.")

    style_subheading(insert_p(), "Interpretación de Resultados Clínicos")
    style_body(insert_p(), "1. Tarjetas de KPIs de Arquitectura de Sueño: ", "La franja superior presenta métricas cuantitativas clave: Tiempo Total en Cama (TIB), Tiempo Total de Sueño (TST), Eficiencia de Sueño (% de tiempo dormido, normal ≥ 85%), Latencia de Inicio del Sueño (SOL) y Vigilia Después del Inicio del Sueño (WASO).")
    style_body(insert_p(), "2. Hipnograma Dual Nocturno Interactivo: ", "Despliega la serie temporal continua de la noche completa mediante dos ribbons sincronizados: la franja superior refleja la predicción del modelo por IA y la franja inferior la clasificación médica de referencia (Ground Truth), codificadas con la paleta AASM: Rojo (W), Lavanda (N1), Azul Índigo (N2), Azul Marino (N3) y Verde Menta (REM).")
    style_body(insert_p(), "3. Inspector Detallado de Época (30 Segundos): ", "Permite auditar microscópicamente cualquier época del estudio mediante un control deslizante continuo. Se visualiza la señal cruda EEG (amplitud en microvoltios vs tiempo) y un gráfico de barras con la distribución probabilística de certidumbre calculada por el modelo para cada uno de los 5 estadios.")
    style_body(insert_p(), "4. Matriz de Confusión y Métricas de Rendimiento: ", "En la sección inferior se despliega la matriz de confusión normalizada comparando predicción vs ground truth, junto con la tabla de sensibilidad (recall) y F1-Score por estadio.")

    style_subheading(insert_p(), "Acciones Adicionales y Buenas Prácticas Clínicas")
    style_bullet(insert_p(), "Alternador de Iluminación: ", "El panel lateral incluye un selector para alternar entre 'Modo Claro' (ideal para reportes médicos diurnos e impresión) y 'Modo Oscuro' (optimizado para salas de monitoreo nocturno).")
    style_bullet(insert_p(), "Consideración diagnóstica: ", "SomnoScope es una herramienta de asistencia y triaje clínico que no sustituye el juicio médico profesional. Se recomienda prestar especial atención diagnóstica a las épocas de estadio N1 debido a su alta variabilidad electroencefalográfica transitoria.")

    doc.save(doc_path)
    print(f"Report successfully updated and saved to: {doc_path}")

if __name__ == '__main__':
    add_sections_to_report('Reporte Entrega 3 - Microproyecto.docx')
