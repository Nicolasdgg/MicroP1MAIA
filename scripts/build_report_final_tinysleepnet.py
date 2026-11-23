import docx
import shutil
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


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


def set_cell_margins(cell, top=60, bottom=60, left=55, right=55):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def insert_comparison_table(target_p):
    table_data = [
        ["Modelo", "Versión / Rol", "Accuracy", "F1-Macro", "Kappa", "F1-N1 (Prior.)", "F1-N3 (Prof.)", "Latencia", "MLflow Registry"],
        ["TinySleepNet (DL)", "v2 (Normalizado - Champion)", "77.95%", "0.7317", "0.6985", "0.4328", "76.66%", "~4 ms", "Production (@champion)"],
        ["LightGBM", "v2 (Tuned - Challenger)", "77.91%", "0.6942", "0.6886", "0.3284", "80.38%", "< 2 ms", "Candidate (@challenger)"],
        ["LightGBM", "v1 (Baseline)", "77.95%", "0.6942", "0.6879", "0.3216", "80.00%", "< 2 ms", "Archived (v1)"],
        ["Random Forest", "v1 (Baseline)", "76.84%", "0.6690", "0.6696", "0.2841", "75.97%", "~18 ms", "Archived (v1)"],
        ["Random Forest", "v2 (Tuned)", "77.06%", "0.6641", "0.6693", "0.2565", "74.47%", "~35 ms", "Archived (v2)"],
        ["TinySleepNet (DL)", "v1 (Sin normalizar)", "14.64%", "0.0511", "0.0000", "0.0000", "0.00%", "~4 ms", "Archived (v1)"],
    ]

    table = target_p._parent.add_table(rows=len(table_data), cols=len(table_data[0]), width=Inches(6.95))
    target_p._p.addprevious(table._tbl)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    col_widths = [
        Inches(1.00), Inches(1.30), Inches(0.58), Inches(0.58), Inches(0.54),
        Inches(0.65), Inches(0.65), Inches(0.55), Inches(1.10)
    ]

    for r_idx, row in enumerate(table.rows):
        is_header = (r_idx == 0)
        is_champion = (r_idx == 1)

        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=55, bottom=55, left=55, right=55)

            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx >= 2 else WD_ALIGN_PARAGRAPH.LEFT

            val = table_data[r_idx][c_idx]
            run = p.add_run(val)
            run.font.name = 'Montserrat SemiBold' if (is_header or is_champion) else 'Montserrat Light'
            run.font.size = Pt(7.5)

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
    dst_path = 'Reporte Entrega Final con TinySleepNet.docx'

    shutil.copy2(src_path, dst_path)
    print(f"Copied fresh template from {src_path} to {dst_path}")

    doc = docx.Document(dst_path)

    # =========================================================================
    # AJUSTE 1: PARTICIÓN ESTRICTA POR SUJETO Y PREVENCIÓN DE DATA LEAKAGE
    # =========================================================================
    target_dataset_idx = None
    for i, p in enumerate(doc.paragraphs):
        if 'Descripción de conjuntos de datos:' in p.text or 'Sleep-EDF Database Expanded' in p.text:
            target_dataset_idx = i + 1
            break

    if target_dataset_idx is not None:
        p_split = doc.paragraphs[target_dataset_idx].insert_paragraph_before()
        style_body(
            p_split,
            "Estrategia de partición de datos y prevención de fuga (data leakage): ",
            "En registros electroencefalográficos (EEG), las épocas continuas de un mismo paciente presentan una alta autocorrelación morfológica y espectral. Una partición aleatoria convencional a nivel de épocas causaría una severa fuga de datos (data leakage), inflando artificialmente las métricas. Para garantizar validez clínica real, la división se implementó estrictamente a nivel de sujeto mediante GroupShuffleSplit (67% entrenamiento / 33% prueba, random_state=42, groups=subjects). Sobre la cohorte completa de 18 registros (9 sujetos únicos, 18,604 épocas), 6 sujetos conforman el conjunto de entrenamiento (12,537 épocas: SC400, SC402, SC403, SC404, ST701, ST704) y 3 sujetos completamente inéditos conforman el conjunto de prueba (6,067 épocas: SC401, SC405, ST702), asegurando evaluación estricta inter-sujeto sin solapamiento alguno."
        )
        print("[OK] Insertada sección de partición por sujeto (12,537 train / 6,067 test).")

    # Renombrar tabla de colaboradores a Tabla 3
    for p in doc.paragraphs:
        if 'Tabla 2: Matriz de Responsabilidades' in p.text or 'Tabla 2: Matriz de Responsabilidades, Dedicaci' in p.text:
            for r in p.runs:
                if 'Tabla 2' in r.text:
                    r.text = r.text.replace('Tabla 2', 'Tabla 3')
            print("[OK] Renombrada tabla de colaboradores a Tabla 3.")

    # Localizar encabezado Colaboradores
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
    # SECCIÓN 1: VERSIONAMIENTO EN MLFLOW, NORMALIZACIÓN TINYSLEEPNET Y CHAMPION
    # =========================================================================
    style_heading(insert_p(), "Versionamiento en MLflow: Escalado a 18 Registros, TinySleepNet Normalizado y Selección Champion")

    style_body(
        insert_p(),
        "Evolución, diagnóstico del colapso en Deep Learning y reentrenamiento v2 (Semanas 6 y 7): ",
        "En cumplimiento con la rúbrica de la Entrega Final, se amplió el entrenamiento a los 18 registros completos (18,604 épocas; 6,067 épocas de prueba en 3 sujetos inéditos) y se ejecutó una segunda iteración analítica (v2) sobre todos los paradigmas evaluados:"
    )

    style_bullet(
        insert_p(),
        "Corrección crítica en TinySleepNet v2 (Normalización z-score por época + Checkpointing): ",
        "El diagnóstico del colapso de TinySleepNet v1 (Kappa=0.0) reveló que MNE entrega el EEG en voltios (~1e-5 V), cuya varianza (~1e-10) quedaba anulada por el parámetro eps=1e-5 de las capas BatchNorm1d. Al incorporar normalización z-score independiente por cada época de 30s ((x - μ) / (σ + 1e-12), libre de fuga entre sujetos) y checkpointing de mejor estado por F1-Macro (época 10), la red 1D-CNN+BiLSTM desbloqueó todo su potencial espaciotemporal."
    )

    style_bullet(
        insert_p(),
        "Reentrenamiento de Modelos Clásicos (LightGBM y Random Forest v1/v2): ",
        "Al entrenar con la cohorte ampliada de 12,537 épocas, LightGBM v2 (300 árboles, lr=0.03, L1/L2) alcanzó 77.91% de exactitud, 0.6942 de F1-Macro y 0.3284 en F1-N1, consolidándose como un sólido referente tabular (Challenger)."
    )

    style_subheading(insert_p(), "Tabla 2: Comparativa Oficial v1 vs v2 en Cohorte de Prueba Inter-Sujeto (6,067 Épocas Inéditas)")
    insert_comparison_table(target_p)

    style_body(
        insert_p(),
        "Criterio de selección del Modelo Champion y salto cuantitativo en N1 (Prioritario): ",
        "A partir de la evaluación justa sobre las mismas 6,067 épocas de sujetos no vistos, TinySleepNet v2 (Normalizado) es promovido formalmente como el Modelo Champion Oficial (@champion / @production):"
    )

    style_bullet(
        insert_p(),
        "Liderazgo Integral en Métricas Globales: ",
        "TinySleepNet v2 encabeza todas las métricas principales: Accuracy de 77.95%, F1-Macro de 0.7317 (+3.75 puntos porcentuales sobre LightGBM v2) y Cohen's Kappa de 0.6985 (acuerdo clínico sustancial), además de destacar en vigilia (F1-W: 0.8590) y sueño REM (F1-REM: 0.7633)."
    )

    style_bullet(
        insert_p(),
        "Avance decisivo en el Estadio Prioritario N1: ",
        "El alcance del proyecto priorizó la detección del estadio transitorio N1. TinySleepNet v2 elevó el F1-N1 a 0.4328 (43.28%), superando en +10.44 puntos porcentuales al mejor modelo clásico sobre los mismos datos (LightGBM v2: 0.3284) y multiplicando casi por 5× el desempeño inicial de Entrega 2 (0.0874), demostrando que las convoluciones 1D sobre señal normalizada capturan la atenuación alfa y ondas agudas de vértex mucho mejor que los descriptores espectrales agregados."
    )

    style_bullet(
        insert_p(),
        "Rol de LightGBM v2 como Challenger (@challenger): ",
        "LightGBM v2 mantiene un rendimiento destacado en sueño profundo N3 (F1-N3: 80.38%) y latencia sub-milisegundo sobre vectores de 25 características, por lo que se conserva registrado como Challenger y motor del endpoint tabular POST /predict/features, además de habilitarse en el selector comparativo del tablero SomnoScope."
    )

    style_body(
        insert_p(),
        "Gobernanza en MLflow Model Registry: ",
        "Todas las corridas, matrices de confusión y artefactos serializados están auditables en el servidor MLflow con IP elástica estática (http://52.207.88.72:5000): 'SomnoScope_TinySleepNet' (v2 en Production con alias @champion y @production; v1 Archived), 'SomnoScope_LightGBM' (v2 Candidate con alias @challenger; v1 Archived) y 'SomnoScope_RandomForest' (v1/v2 Archived)."
    )

    # =========================================================================
    # SECCIÓN 2: ARQUITECTURA DE MICROSERVICIOS, API REST Y DOCKER
    # =========================================================================
    style_heading(insert_p(), "Despliegue y Arquitectura de Microservicios: API REST, Docker y Elastic IP")

    style_body(
        insert_p(),
        "Servicio desacoplado vía API REST (FastAPI v3.0): ",
        "El Champion TinySleepNet1D normalizado ('models/model_tinysleepnet.pt', 650 KB) y el modelo tabular LightGBM ('models/best_sleep_model.pkl') se sirven simultáneamente mediante FastAPI con esquemas Pydantic:"
    )

    style_bullet(insert_p(), "GET /health: ", "Verifica el estado operativo, reportando TinySleepNet1D Normalizado como modelo Champion activo, LightGBM como respaldo tabular, las 5 clases AASM y las 25 características esperadas.")
    style_bullet(insert_p(), "POST /predict/epoch y POST /predict/recording: ", "Reciben ventanas crudas de 30s (3,000 muestras a 100 Hz) o archivos polisomnográficos .edf completos, ejecutan normalización z-score por época e infieren por lotes con TinySleepNet1D (con opción de seleccionar LightGBM) retornando probabilidades calibradas, hipnograma y KPIs clínicos.")
    style_bullet(insert_p(), "POST /predict/features: ", "Preserva compatibilidad con clientes tabulares recibiendo 25 descriptores espectrales y respondiendo vía LightGBM Challenger.")

    style_body(
        insert_p(),
        "Contenedorización, Elastic IP y Versionamiento de Datos (DVC): ",
        "El Dockerfile multietapa ('python:3.12-slim' con PyTorch CPU-only) orquesta mediante 'docker-compose.yml' tres microservicios: 'somnoscope-api' (puerto 8000), 'somnoscope-dashboard' (Streamlit en puerto 8050, con selector interactivo TinySleepNet Champion vs LightGBM) y 'somnoscope-mlflow' (puerto 5000 con SQLite persistente). La instancia AWS EC2 (t2.medium + 2 GB Swap) opera con Elastic IP permanente (52.207.88.72) y dataset Sleep-EDFx versionado con DVC en Amazon S3 ('s3://microproyecto-maia')."
    )

    # =========================================================================
    # SECCIÓN 3: MANUAL DE INSTALACIÓN
    # =========================================================================
    style_heading(insert_p(), "Manual de Instalación - Tablero SomnoScope")

    style_subheading(insert_p(), "Requisitos Previos y Clonación")
    style_bullet(insert_p(), "Hardware y Software: ", "Mínimo 4 GB RAM (o 2 GB + 2 GB Swap), 5 GB en disco, Python 3.10+, Git, Docker Engine y Docker Compose v2. Puertos TCP abiertos: 22 (SSH), 5000 (MLflow), 8000 (API) y 8050 (Dashboard).")
    style_code(insert_p(), "git clone https://github.com/Nicolasdgg/MicroP1MAIA.git && cd MicroP1MAIA && git checkout main\ndvc pull   # Opcional para sincronizar datos adicionales desde S3\ndocker compose up -d --build")
    style_body(insert_p(), "Verificación de servicios: ", "Ejecute 'docker compose ps' (los 3 contenedores deben mostrar estado 'Up') y valide con 'curl -s http://localhost:8000/health'.")

    style_subheading(insert_p(), "Despliegue en AWS EC2 con Elastic IP")
    style_body(insert_p(), "Preparación en VM Ubuntu 22.04: ", "Habilite TCP 22, 5000, 8000 y 8050 en el Security Group, asocie una Elastic IP para conservar URLs fijas ante reinicios, active Swap y levante el stack:")
    style_code(insert_p(), "sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2\nsudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile\ngit clone https://github.com/Nicolasdgg/MicroP1MAIA.git && cd MicroP1MAIA && sudo docker compose up -d --build")

    # =========================================================================
    # SECCIÓN 4: MANUAL DE USUARIO
    # =========================================================================
    style_heading(insert_p(), "Manual de Usuario - Sistema SomnoScope")

    style_subheading(insert_p(), "Acceso, Selección de Modelo y Carga de Estudios")
    style_body(insert_p(), "Acceso: ", "Localmente en 'http://localhost:8050' o en la nube con IP elástica fija en 'http://52.207.88.72:8050' (Swagger API en ':8000/docs', MLflow en ':5000').")
    style_body(insert_p(), "Operación: ", "En la barra superior seleccione cualquiera de los 18 registros clínicos precargados (ej. 'SC4011 · E0') o active 'Cargar registro EDF' para subir un archivo externo. El selector de modelo permite inferir con el Champion 'TinySleepNet v2 (DL)' por defecto o alternar con 'LightGBM (Clásico)' para contrastar el hipnograma y el F1 de N1 en vivo.")

    style_subheading(insert_p(), "Interpretación de Resultados Clínicos")
    style_bullet(insert_p(), "KPIs de Arquitectura de Sueño: ", "Calcula automáticamente TIB, TST, Eficiencia de Sueño (≥ 85% normal), Latencia de Inicio (SOL) y Vigilia Intra-Sueño (WASO).")
    style_bullet(insert_p(), "Hipnograma Dual e Inspector de Época (30s): ", "Muestra en paralelo la predicción del Champion vs la anotación experta en paleta AASM, junto con el visor microscópico de señal EEG y probabilidades calibradas por estadio.")
    style_bullet(insert_p(), "Matriz de Confusión, F1 por Clase y Panel MLflow: ", "Actualiza dinámicamente la matriz de contingencia y el F1 de cada estadio (destacando N1) junto con la tabla real de corridas registradas en MLflow y selector de Modo Claro / Oscuro. Herramienta de soporte clínico que no sustituye el criterio médico especialista.")

    doc.save(dst_path)
    print(f"[OK] Reporte generado exitosamente en: {dst_path}")


if __name__ == '__main__':
    build_report()
