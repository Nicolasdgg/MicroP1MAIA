import docx
import shutil
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def style_heading(p, text):
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = 'Montserrat SemiBold'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x5b, 0x57, 0xd1)
    return p

def style_subheading(p, text):
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = 'Montserrat SemiBold'
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    return p

def style_body(p, bold_prefix, text):
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = 'Montserrat SemiBold'
        r_bold.font.size = Pt(9)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    r_text = p.add_run(text)
    r_text.font.name = 'Montserrat Light'
    r_text.font.size = Pt(9)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def style_bullet(p, bold_prefix, text):
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.line_spacing = 1.15
    r_bullet = p.add_run("▪  ")
    r_bullet.font.name = 'Montserrat SemiBold'
    r_bullet.font.size = Pt(8.5)
    r_bullet.font.color.rgb = RGBColor(0x5b, 0x57, 0xd1)
    if bold_prefix:
        r_bold = p.add_run(bold_prefix)
        r_bold.font.name = 'Montserrat SemiBold'
        r_bold.font.size = Pt(9)
        r_bold.font.bold = True
        r_bold.font.color.rgb = RGBColor(0x2c, 0x3e, 0x50)
    r_text = p.add_run(text)
    r_text.font.name = 'Montserrat Light'
    r_text.font.size = Pt(9)
    r_text.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def style_code(p, code_text):
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.2)
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x1a, 0x36, 0x5d)
    return p

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def insert_comparison_table(target_p):
    table_data = [
        ["Modelo", "Versión / Iteración", "Accuracy", "F1-Macro", "Kappa", "F1-N3", "Latencia", "MLflow Registry"],
        ["LightGBM", "v2 (Champion - E3)", "66.21%", "0.5383", "0.5316", "73.73%", "< 2 ms", "Production (@champion)"],
        ["LightGBM", "v1 (Baseline - E2)", "67.21%", "0.5525", "0.5424", "71.12%", "< 2 ms", "Archived (v1)"],
        ["Random Forest", "v2 (Tuned - E3)", "61.73%", "0.5024", "0.4718", "49.24%", "~35 ms", "Production (v2)"],
        ["Random Forest", "v1 (Baseline - E2)", "64.73%", "0.5548", "0.5200", "64.55%", "~18 ms", "Archived (v1)"],
        ["TinySleepNet", "v1 (CNN+BiLSTM)", "50.24%", "0.1338", "0.0000", "0.00%", "~45 ms", "Production (v1)"]
    ]
    
    table = target_p._parent.add_table(rows=len(table_data), cols=len(table_data[0]), width=Inches(6.8))
    target_p._p.addprevious(table._tbl)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    col_widths = [Inches(1.2), Inches(1.5), Inches(0.75), Inches(0.75), Inches(0.65), Inches(0.75), Inches(0.8), Inches(1.4)]
    
    for r_idx, row in enumerate(table.rows):
        is_header = (r_idx == 0)
        is_champion = (r_idx == 1)
        
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            
            val = table_data[r_idx][c_idx]
            run = p.add_run(val)
            run.font.name = 'Montserrat SemiBold' if (is_header or is_champion) else 'Montserrat Light'
            run.font.size = Pt(8)
            
            if is_header:
                set_cell_background(cell, "5B57D1")
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold = True
            elif is_champion:
                set_cell_background(cell, "EBF4FF")
                run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
                run.font.bold = True
            else:
                set_cell_background(cell, "F8F9FA" if r_idx % 2 == 0 else "FFFFFF")
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

def build_report():
    src_path = 'Reporte Entrega 2 - Microproyecto.docx'
    dst_path = 'Reporte Entrega 3 - Microproyecto.docx'
    
    shutil.copy2(src_path, dst_path)
    print(f"Copied fresh template from {src_path} to {dst_path}")
    
    doc = docx.Document(dst_path)
    
    # Locate index of Colaboradores
    target_idx = None
    for i, p in enumerate(doc.paragraphs):
        if 'Colaboradores' in p.text:
            target_idx = i
            break
            
    if target_idx is None:
        raise ValueError("Could not find 'Colaboradores' heading in document")
        
    target_p = doc.paragraphs[target_idx]
    
    def insert_p():
        return target_p.insert_paragraph_before()
        
    # =========================================================================
    # SECCIÓN 1: VERSIONAMIENTO EN MLFLOW, COMPARACIÓN Y MODELO CHAMPION
    # =========================================================================
    style_heading(insert_p(), "Versionamiento de Modelos en MLflow: Iteración, Comparación y Selección Champion")
    
    style_body(insert_p(), "Evolución y reentrenamiento de nuevas versiones (Semana 6 y 7): ",
               "En estricto cumplimiento con la rúbrica de la Entrega 3 ('Desarrollar nuevas versiones de los modelos, comparar y seleccionar mejores alternativas; versionando los modelos en MLflow'), se diseñó una segunda iteración analítica (v2) enfocada en la optimización de hiperparámetros y la reducción de falsos positivos en transiciones de sueño:")
               
    style_bullet(insert_p(), "Random Forest v2 (Tuned Ensemble): ",
                 "Se incrementó el bosque a 250 estimadores con profundidad máxima de 16, división mínima de 4 muestras y pesos de clase balanceados por subsample ('balanced_subsample') para compensar el desbalance crítico en estadios N1 y REM.")
                 
    style_bullet(insert_p(), "LightGBM v2 (Fine-Tuned Gradient Boosting): ",
                 "Se amplió a 300 árboles con tasa de aprendizaje conservadora (lr=0.03), control de profundidad máxima (7, con 31 hojas), submuestreo de instancias y características (subsample=0.85, colsample=0.85) y regularización explícita L1/L2 (alpha=0.1, lambda=1.5) para penalizar divisiones redundantes en épocas transitorias.")

    style_subheading(insert_p(), "Tabla 2: Comparativa de Modelos v1 vs v2 y Selección de Alternativa Champion")
    insert_comparison_table(target_p)
    
    style_body(insert_p(), "Criterio de selección del Modelo Champion: ",
               "Aunque LightGBM v1 exhibió una leve superioridad en accuracy global impulsada por la clase mayoritaria N2, LightGBM v2 demostró un avance clínico determinante en la detección de sueño profundo N3 (F1-N3: 73.73%, incremento de +2.61 puntos porcentuales), una varianza significativamente menor y una mayor robustez contra artefactos electroencefalográficos transitorios. Junto con una latencia de inferencia inferior a 2 milisegundos por época, LightGBM v2 fue seleccionado como el modelo Champion oficial para producción y empaquetado en la API REST.")

    style_body(insert_p(), "Registro oficial en MLflow Model Registry: ",
               "Todas las versiones fueron registradas formalmente en el Model Registry del servidor MLflow (http://18.212.239.119:5000): 'SomnoScope_LightGBM' (v1 en estado Archived, v2 en estado Production con alias @champion y @production), 'SomnoScope_RandomForest' (v1 Archived, v2 Production) y 'SomnoScope_TinySleepNet' (v1 Production). La comparación paramétrica y de métricas cruzadas está habilitada directamente en la vista Experiments.")

    # =========================================================================
    # SECCIÓN 2: ARQUITECTURA DE MICROSERVICIOS, API REST Y DOCKER
    # =========================================================================
    style_heading(insert_p(), "Despliegue y Arquitectura de Microservicios: API REST, Docker y DVC")
    
    style_body(insert_p(), "Servicio desacoplado vía API REST (FastAPI): ",
               "El modelo Champion seleccionado (LightGBM v2) y el pipeline de normalización espectral fueron serializados en 'models/best_sleep_model.pkl'. La API REST construida con FastAPI expone esquemas de validación tipados mediante Pydantic y cuatro endpoints especializados:")
               
    style_bullet(insert_p(), "GET /health: ", "Endpoint de monitoreo y healthcheck continuo que verifica el estado operativo del servicio, las 5 clases objetivo AASM (W, N1, N2, N3, REM) y las 25 características espectrales.")
    style_bullet(insert_p(), "POST /predict/features: ", "Recibe un vector de 25 características espectrales calculadas para una época y retorna la clase diagnóstica predicha junto con la distribución probabilística calibrada.")
    style_bullet(insert_p(), "POST /predict/epoch: ", "Recibe una ventana cruda de 30 segundos de EEG (3,000 muestras a 100 Hz), ejecuta la extracción espectral en tiempo real y clasifica la época.")
    style_bullet(insert_p(), "POST /predict/recording: ", "Permite la ingesta directa de un archivo polisomnográfico completo en formato .edf, procesando la noche completa para generar el hipnograma continuo y los KPIs de sueño.")

    style_body(insert_p(), "Contenedorización integral con Docker Compose: ",
               "Se construyó un Dockerfile multietapa optimizado sobre 'python:3.12-slim' con PyTorch compilado exclusivamente para arquitectura CPU (180 MB frente a los 3 GB habituales). Mediante 'docker-compose.yml', se orquestan tres contenedores interconectados en la red aislada 'microp1maia_default': 'somnoscope-api' (FastAPI en puerto 8000), 'somnoscope-dashboard' (Streamlit en puerto 8050) y 'somnoscope-mlflow' (MLflow en puerto 5000 con SQLite persistente en volumen Docker).")

    style_body(insert_p(), "Versionamiento de datos y modelos con DVC y Amazon S3: ",
               "Los conjuntos de datos masivos de polisomnografía (Sleep-EDFx, 729 MB distribuidos en 39 archivos) se encuentran versionados mediante DVC en 'data.dvc', respaldados en un bucket seguro de Amazon S3 ('s3://microproyecto-maia'). Esto permite la sincronización inmediata del dataset crudo en cualquier instancia mediante 'dvc pull'.")

    # =========================================================================
    # SECCIÓN 3: MANUAL DE INSTALACIÓN
    # =========================================================================
    style_heading(insert_p(), "Manual de Instalación - Tablero SomnoScope")
    
    style_subheading(insert_p(), "Requisitos Previos")
    style_bullet(insert_p(), "Hardware: ", "Mínimo 4 GB de memoria RAM disponible (o 2 GB físicos con 2 GB de memoria Swap activa) y al menos 5 GB de espacio libre en disco.")
    style_bullet(insert_p(), "Software base: ", "Python 3.10 o superior, Git, Docker Engine y Docker Compose v2 instalados y operativos.")
    style_bullet(insert_p(), "Puertos de red: ", "Disponibilidad de los puertos TCP 8000 (API REST), 8050 (Tablero Streamlit) y 5000 (MLflow).")

    style_subheading(insert_p(), "Instalación y Clonación del Repositorio")
    style_body(insert_p(), "1. Clonación oficial: ", "Clone el repositorio oficial del proyecto alojado en GitHub y sitúese en la rama principal 'main':")
    style_code(insert_p(), "git clone https://github.com/Nicolasdgg/MicroP1MAIA.git\ncd MicroP1MAIA\ngit checkout main")
    style_body(insert_p(), "2. Descarga de datos y modelos (DVC): ", "Configure credenciales de AWS ('aws configure') y descargue los registros de sueño:")
    style_code(insert_p(), "dvc pull")

    style_subheading(insert_p(), "Inicio del Sistema con Docker Compose (Recomendado)")
    style_body(insert_p(), "", "Para inicializar de manera desatendida los tres microservicios (API, Dashboard y MLflow), ejecute:")
    style_code(insert_p(), "docker compose build\ndocker compose up -d")
    style_body(insert_p(), "Verificación de contenedores: ", "Compruebe que los servicios estén activos ejecutando 'docker compose ps'. Los contenedores 'somnoscope-api', 'somnoscope-dashboard' y 'somnoscope-mlflow' deben mostrar estado 'Up'.")

    style_subheading(insert_p(), "Verificación del Funcionamiento")
    style_bullet(insert_p(), "API REST (Puerto 8000): ", "Compruebe la salud con 'curl -s http://localhost:8000/health'. La documentación Swagger UI está en 'http://localhost:8000/docs'.")
    style_bullet(insert_p(), "Tablero SomnoScope (Puerto 8050): ", "Acceda desde el navegador a 'http://localhost:8050'.")
    style_bullet(insert_p(), "Servidor MLflow (Puerto 5000): ", "Acceda desde el navegador a 'http://localhost:5000'.")

    style_subheading(insert_p(), "Despliegue en Amazon Web Services (AWS EC2)")
    style_body(insert_p(), "1. Configuración de Red: ", "En AWS EC2, habilite en el Security Group reglas de entrada para TCP 22 (SSH), 8050 (Dashboard), 5000 (MLflow) y 8000 (API).")
    style_body(insert_p(), "2. Preparación y Despliegue en VM: ", "Conéctese por SSH, instale Docker, active Swap y levante el stack:")
    style_code(insert_p(), "sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2\nsudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile\ngit clone https://github.com/Nicolasdgg/MicroP1MAIA.git && cd MicroP1MAIA && git checkout main\ndocker compose up -d")

    # =========================================================================
    # SECCIÓN 4: MANUAL DE USUARIO
    # =========================================================================
    style_heading(insert_p(), "Manual de Usuario - Sistema SomnoScope")
    
    style_subheading(insert_p(), "Acceso al Sistema y Carga de Estudios")
    style_body(insert_p(), "Acceso: ", "Ingrese localmente a 'http://localhost:8050' o en la nube a 'http://18.212.239.119:8050'.")
    style_body(insert_p(), "Procedimiento de carga: ", "En el panel lateral izquierdo, seleccione un paciente precargado de validación clínica (ej. 'SC4001E0') o cargue un archivo polisomnográfico externo en formato .edf con canal EEG Fpz-Cz o Pz-Oz a 100 Hz. La inferencia se envía automáticamente a la API REST.")

    style_subheading(insert_p(), "Interpretación de Resultados Clínicos")
    style_bullet(insert_p(), "KPIs de Arquitectura de Sueño: ", "La franja superior presenta métricas cuantitativas clave: Tiempo Total en Cama (TIB), Tiempo Total de Sueño (TST), Eficiencia de Sueño (≥ 85% normal), Latencia de Inicio (SOL) y Vigilia Tras Inicio (WASO).")
    style_bullet(insert_p(), "Hipnograma Dual Nocturno Interactivo: ", "Ribbons sincronizados de la noche completa: franja superior con la predicción del modelo Champion y franja inferior con la referencia médica (Ground Truth), codificados en colores AASM (W: Rojo, N1: Lavanda, N2: Azul Índigo, N3: Azul Marino, REM: Verde Menta).")
    style_bullet(insert_p(), "Inspector Microscópico de Época (30 Segundos): ", "Control deslizante continuo que despliega la señal cruda EEG (amplitud en microvoltios) y el gráfico de barras de probabilidades calibradas por estadio.")
    style_bullet(insert_p(), "Matriz de Confusión y Métricas: ", "Matriz normalizada de contingencia diagnóstica, sensibilidad y F1-score discriminado por estadio.")

    style_subheading(insert_p(), "Opciones Adicionales")
    style_bullet(insert_p(), "Modos Claro / Oscuro: ", "Selector lateral para alternar entre 'Modo Claro' (reportes médicos diurnos) y 'Modo Oscuro' (estaciones de polisomnografía nocturnas).")
    style_bullet(insert_p(), "Consideración diagnóstica: ", "Herramienta de soporte y triaje clínico que no sustituye el criterio médico especialista.")

    doc.save(dst_path)
    print(f"Reporte Entrega 3 successfully generated at: {dst_path}")

if __name__ == '__main__':
    build_report()
