"""
Generador del Artículo Científico Final de SomnoScope en formato Word (.docx).
Reutiliza la plantilla visual oficial MAIA / Universidad de los Andes (encabezado, pie de página,
portada, márgenes y tipografía Montserrat / paleta índigo #5B57D1) y estructura el documento
estrictamente en las 10 secciones exigidas para el informe final tipo artículo científico.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import docx
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

REPO_ROOT = Path(r"G:\Mi unidad\MAIA\Proyecto_Desarrollo_de_Soluciones\MicroP1MAIA")
FIG_DIR = REPO_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. GENERACIÓN DE FIGURAS DE ALTA RESOLUCIÓN (300 DPI)
# =============================================================================
def generate_architecture_figure(out_path: Path):
    """Genera diagrama vectorial limpio de la arquitectura de extremo a extremo (ciclo ML + MLOps)."""
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # Título interno sutil
    ax.text(5.5, 5.15, "Arquitectura Integral de SomnoScope: Ciclo de Vida de Machine Learning y Despliegue MLOps",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1e293b", family="sans-serif")

    def draw_box(x, y, w, h, title, body, fc, ec, title_c="#1e293b"):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=1.6
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + (h - 0.34) / 2, body, ha="center", va="center",
                fontsize=7.0, color="#334155", linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12))

    # Bandas de fondo por capa
    layers = [
        (0.15, 0.25, 3.15, 4.55, "#f8fafc", "#cbd5e1", "1. DATOS Y PREPROCESAMIENTO"),
        (3.55, 0.25, 3.65, 4.55, "#f5f3ff", "#c4b5fd", "2. MODELADO Y GOBERNANZA MLOPS"),
        (7.45, 0.25, 3.40, 4.55, "#eff6ff", "#93c5fd", "3. DESPLIEGUE PRODUCTIVO (AWS EC2)"),
    ]
    for lx, ly, lw, lh, lfc, lec, ltitle in layers:
        bg = patches.FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor=lfc, edgecolor=lec, linewidth=1.2, linestyle="--")
        ax.add_patch(bg)
        ax.text(lx + lw / 2, ly + lh - 0.18, ltitle, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="#4f46e5")

    # Capa 1: Datos
    draw_box(0.32, 3.35, 2.81, 1.05, "Sleep-EDF Expanded (DVC + S3)",
             "18 estudios · 9 sujetos (6 SC + 3 ST)\n729 MB versionados en Amazon S3\n18,604 épocas válidas de 30 s",
             "#ffffff", "#64748b")
    draw_box(0.32, 1.95, 2.81, 1.08, "Preprocesamiento Clínico (MNE)",
             "Canal estándar EEG Fpz-Cz (100 Hz)\nFiltro pasa-banda 0.5–35 Hz\nExclusión M/? · Recorte ±30 min",
             "#ffffff", "#0284c7")
    draw_box(0.32, 0.48, 2.81, 1.12, "Partición Estricta por Sujeto",
             "GroupShuffleSplit (random_state=42)\nTrain: 6 sujetos (12,537 épocas)\nTest inédito: 3 sujetos (6,067 épocas)",
             "#ecfdf5", "#10b981", "#065f46")

    arrow(1.72, 3.35, 1.72, 3.05)
    arrow(1.72, 1.95, 1.72, 1.62)

    # Capa 2: Modelado
    draw_box(3.75, 3.18, 3.25, 1.22, "Rama Deep Learning (Champion)",
             "Normalización Z-score por época\nTinySleepNet1D (1D-CNN + BiLSTM)\nPesos de clase + Checkpointing (ep. 10)\nAcc 77.95% · F1-Mac 0.732 · F1-N1 0.433",
             "#eef2ff", "#4f46e5", "#312e81")
    draw_box(3.75, 1.78, 3.25, 1.12, "Rama Clásica (Challenger)",
             "25 características Welch PSD + Hjorth\nLightGBM v2 (Tuned) & Random Forest\nAcc 77.91% · F1-Mac 0.694 · F1-N1 0.328",
             "#ffffff", "#6366f1")
    draw_box(3.75, 0.48, 3.25, 1.02, "MLflow Tracking & Registry (:5000)",
             "SQLite persistente + matrices de confusión\n@champion/@production: TinySleepNet v2\n@challenger: LightGBM v2",
             "#fef3c7", "#d97706", "#92400e")

    arrow(3.15, 1.25, 3.73, 3.65, "#4f46e5")
    arrow(3.15, 1.05, 3.73, 2.30, "#6366f1")
    # Flecha limpia desde Rama Clásica hacia MLflow (sin cruzar texto)
    arrow(5.37, 1.78, 5.37, 1.56, "#d97706")
    # Flecha curva lateral desde Rama Deep Learning hacia MLflow bordeando por la derecha sin tocar la caja intermedia
    ax.annotate("", xy=(7.02, 1.18), xytext=(7.02, 3.45),
                arrowprops=dict(arrowstyle="-|>", color="#d97706", lw=1.5, mutation_scale=12,
                                connectionstyle="arc3,rad=-0.18"))

    # Capa 3: Despliegue
    draw_box(7.65, 3.25, 3.00, 1.15, "FastAPI REST v3.0 (:8000)",
             "GET /health · /docs (Swagger UI)\nPOST /predict/epoch & /recording (DL)\nPOST /predict/features (LightGBM)",
             "#ffffff", "#2563eb", "#1e3a8a")
    draw_box(7.65, 1.82, 3.00, 1.15, "Dashboard SomnoScope (:8050)",
             "Streamlit v3.0 + Plotly interactivo\nSelector Champion vs Challenger · Carga .edf\nRecorrido guiado (6 pasos) + KPIs + Hipnograma",
             "#ffffff", "#2563eb", "#1e3a8a")
    draw_box(7.65, 0.48, 3.00, 1.06, "Docker Compose + AWS EC2",
             "Instancia t2.medium (4 GB RAM + 2 GB Swap)\nContenedores: api · dashboard · mlflow\nElastic IP fija: 52.207.88.72",
             "#dbeafe", "#1d4ed8", "#1e3a8a")

    arrow(7.02, 3.75, 7.63, 3.75, "#2563eb")
    arrow(7.02, 2.35, 7.63, 2.35, "#2563eb")
    arrow(9.15, 3.25, 9.15, 2.99, "#2563eb")
    arrow(7.02, 0.98, 7.63, 0.98, "#1d4ed8")

    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_comparison_figure(out_path: Path):
    """Genera gráfica de 2 paneles: Métricas globales y F1-Score por estadio AASM (6,067 épocas)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=300)

    # Panel A: Métricas globales por modelo
    models = ["TinySleepNet\nv2 (Norm.)", "LightGBM\nv2 (Tuned)", "LightGBM\nv1 (Base)", "Random Forest\nv1 (Base)", "TinySleepNet\nv1 (Sin Norm.)"]
    acc = [77.95, 77.91, 77.95, 76.84, 14.64]
    f1m = [73.17, 69.42, 69.42, 66.90, 5.11]
    kap = [69.85, 68.86, 68.79, 66.96, 0.00]

    x = np.arange(len(models))
    w = 0.25
    b1 = ax1.bar(x - w, acc, w, label="Accuracy (%)", color="#4f46e5")
    b2 = ax1.bar(x, f1m, w, label="F1-Macro (×100)", color="#10b981")
    b3 = ax1.bar(x + w, kap, w, label="Kappa (×100)", color="#f59e0b")

    ax1.set_title("A. Desempeño Global Inter-Sujeto (6,067 épocas)", fontsize=9.8, fontweight="bold", color="#1e293b", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.5)
    ax1.set_ylim(0, 95)
    ax1.set_ylabel("Porcentaje / Escala ×100", fontsize=8.5, color="#334155")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=7.5, loc="upper right", frameon=True)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=6.4, color="#1e293b")

    # Panel B: F1-Score por estadio AASM (Champion vs Challenger vs RF vs TSN v1)
    stages = ["W\n(Vigilia)", "N1\n(Prioritario)", "N2\n(Ligero)", "N3\n(Profundo)", "REM"]
    tsn_v2 = [0.859, 0.433, 0.837, 0.767, 0.763]
    lgb_v2 = [0.748, 0.328, 0.874, 0.804, 0.717]
    rf_v1 = [0.729, 0.284, 0.869, 0.760, 0.703]
    tsn_v1 = [0.255, 0.000, 0.000, 0.000, 0.000]

    xs = np.arange(len(stages))
    w2 = 0.20
    c1 = ax2.bar(xs - 1.5 * w2, tsn_v2, w2, label="TinySleepNet v2 (Champion)", color="#4f46e5")
    c2 = ax2.bar(xs - 0.5 * w2, lgb_v2, w2, label="LightGBM v2 (Challenger)", color="#0ea5e9")
    c3 = ax2.bar(xs + 0.5 * w2, rf_v1, w2, label="Random Forest v1", color="#94a3b8")
    c4 = ax2.bar(xs + 1.5 * w2, tsn_v1, w2, label="TinySleepNet v1 (Sin norm.)", color="#ef4444")

    # Resaltar columna N1
    ax2.axvspan(0.55, 1.45, color="#fef3c7", alpha=0.45, zorder=0)
    ax2.annotate("+10.4 pp en N1\nvs. LightGBM", xy=(1 - 1.5 * w2, 0.433), xytext=(0.65, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3),
                 fontsize=7.3, fontweight="bold", color="#92400e",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#fffbeb", ec="#f59e0b"))

    ax2.set_title("B. F1-Score por Estadio AASM (Foco en N1)", fontsize=9.8, fontweight="bold", color="#1e293b", pad=10)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(stages, fontsize=8.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("F1-Score", fontsize=8.5, color="#334155")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(fontsize=7.2, loc="upper right", frameon=True)
    for bar in c1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#312e81")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# 2. UTILIDADES DE ESTILO WORD (HOMOLOGADAS A LA PLANTILLA MAIA)
# =============================================================================
def set_cell_bg(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=60, bottom=60, left=70, right=70):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Montserrat SemiBold"
    r.font.size = Pt(12.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x5B, 0x57, 0xD1)
    return p


def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3.5)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Montserrat SemiBold"
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    return p


def add_body(doc, text, bold_prefix=None, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Montserrat SemiBold"
        rb.font.size = Pt(9.5)
        rb.font.bold = True
        rb.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    rt = p.add_run(text)
    rt.font.name = "Montserrat Light"
    rt.font.size = Pt(9.5)
    rt.font.italic = italic
    rt.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p


def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.line_spacing = 1.15
    rb0 = p.add_run("▪  ")
    rb0.font.name = "Montserrat SemiBold"
    rb0.font.size = Pt(8.5)
    rb0.font.color.rgb = RGBColor(0x5B, 0x57, 0xD1)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Montserrat SemiBold"
        rb.font.size = Pt(9.5)
        rb.font.bold = True
        rb.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    rt = p.add_run(text)
    rt.font.name = "Montserrat Light"
    rt.font.size = Pt(9.5)
    rt.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.name = "Montserrat Light"
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return p


def add_figure(doc, img_path: Path, caption: str, width_in: float = 6.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Inches(width_in))
    add_caption(doc, caption)


def add_styled_table(doc, headers, rows, col_widths, highlight_row_idx=None, font_pt=8.0):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    all_rows = [headers] + rows

    for r_idx, r_vals in enumerate(all_rows):
        is_hdr = (r_idx == 0)
        is_hl = (highlight_row_idx is not None and r_idx == highlight_row_idx + 1)
        row = table.rows[r_idx]
        for c_idx, val in enumerate(r_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=55, bottom=55, left=55, right=55)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.name = "Montserrat SemiBold" if (is_hdr or is_hl) else "Montserrat Light"
            run.font.size = Pt(font_pt)
            if is_hdr:
                set_cell_bg(cell, "5B57D1")
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.bold = True
            elif is_hl:
                set_cell_bg(cell, "EBF4FF")
                run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
                run.font.bold = True
            else:
                set_cell_bg(cell, "F8F9FA" if r_idx % 2 == 0 else "FFFFFF")
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return table


# =============================================================================
# 3. CONSTRUCCIÓN DEL ARTÍCULO CIENTÍFICO
# =============================================================================
def build_article():
    arch_fig = FIG_DIR / "arquitectura_somnoscope_articulo.png"
    comp_fig = FIG_DIR / "comparativa_modelos_estadios_articulo.png"
    cm_fig = FIG_DIR / "confusion_matrix_tinysleepnet_v2.png"

    print("[1/3] Generando figuras de arquitectura y resultados (300 DPI)...")
    generate_architecture_figure(arch_fig)
    generate_comparison_figure(comp_fig)

    src_template = REPO_ROOT / "Reporte Entrega 2 - Microproyecto.docx"
    dst_path = REPO_ROOT / "Articulo Cientifico Final - SomnoScope.docx"
    shutil.copy2(src_template, dst_path)

    doc = docx.Document(dst_path)

    # Conservar la portada visual (Table 0) con los logos institucionales MAIA / Uniandes
    # y limpiar todos los elementos posteriores del cuerpo original de Entrega 2.
    body = doc._body._body
    t0_elem = doc.tables[0]._tbl
    keep_until_seen = False
    to_remove = []
    for child in list(body):
        if child is t0_elem:
            keep_until_seen = True
            continue
        if child.tag.endswith("sectPr"):
            continue
        if keep_until_seen:
            to_remove.append(child)
    for elem in to_remove:
        body.remove(elem)

    # Actualizar texto de la portada (Table 0) con Título del Artículo, Autores y Afiliación (sin correos)
    t0 = doc.tables[0]
    p_title = t0.rows[0].cells[0].paragraphs[0]
    p_title.clear()
    r_main = p_title.add_run("SomnoScope")
    r_main.font.name = "Montserrat SemiBold"
    r_main.font.size = Pt(28)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(0x4A, 0x6F, 0xA5)

    p_sub = t0.rows[0].cells[0].add_paragraph()
    p_sub.paragraph_format.space_before = Pt(4)
    p_sub.paragraph_format.space_after = Pt(8)
    r_sub = p_sub.add_run(
        "Estadificación Automática del Sueño mediante Redes Convolucionales-Recurrentes "
        "Normalizadas sobre Electroencefalografía Monocanal y Arquitectura MLOps Productiva"
    )
    r_sub.font.name = "Montserrat SemiBold"
    r_sub.font.size = Pt(12.5)
    r_sub.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    c_auth = t0.rows[1].cells[0]
    c_auth.paragraphs[0].clear()
    r_au = c_auth.paragraphs[0].add_run("Arturo Molina · Catalina García · Daniel Franco · Nicolás Gómez")
    r_au.font.name = "Montserrat SemiBold"
    r_au.font.size = Pt(10.5)
    r_au.font.color.rgb = RGBColor(0x4A, 0x6F, 0xA5)

    p_aff = c_auth.add_paragraph()
    p_aff.paragraph_format.space_before = Pt(4)
    r_aff = p_aff.add_run(
        "Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes, Bogotá, Colombia\n"
        "Proyecto: Despliegue de Soluciones de Inteligencia Artificial"
    )
    r_aff.font.name = "Montserrat Light"
    r_aff.font.size = Pt(9.5)
    r_aff.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Salto de página tras portada para iniciar el artículo limpio en página 2
    doc.add_page_break()

    # =========================================================================
    # TÍTULO Y AUTORES (ENCABEZADO ACADÉMICO DEL ARTÍCULO)
    # =========================================================================
    p_art_t = doc.add_paragraph()
    p_art_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_art_t.paragraph_format.space_before = Pt(2)
    p_art_t.paragraph_format.space_after = Pt(6)
    rt = p_art_t.add_run(
        "SomnoScope: Estadificación Automática del Sueño mediante Redes Convolucionales-Recurrentes "
        "Normalizadas sobre Electroencefalografía Monocanal y Arquitectura MLOps Productiva"
    )
    rt.font.name = "Montserrat SemiBold"
    rt.font.size = Pt(14)
    rt.font.bold = True
    rt.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    p_art_a = doc.add_paragraph()
    p_art_a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_art_a.paragraph_format.space_after = Pt(2)
    ra = p_art_a.add_run("Arturo Molina, Catalina García, Daniel Franco y Nicolás Gómez")
    ra.font.name = "Montserrat SemiBold"
    ra.font.size = Pt(10)
    ra.font.color.rgb = RGBColor(0x5B, 0x57, 0xD1)

    p_art_f = doc.add_paragraph()
    p_art_f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_art_f.paragraph_format.space_after = Pt(10)
    rf = p_art_f.add_run("Maestría en Inteligencia Artificial (MAIA), Universidad de los Andes, Bogotá, Colombia")
    rf.font.name = "Montserrat Light"
    rf.font.size = Pt(9)
    rf.font.italic = True
    rf.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # =========================================================================
    # 3. RESUMEN (ESTRICTO: 200-300 PALABRAS) + PALABRAS CLAVE (3-5)
    # =========================================================================
    add_h1(doc, "Resumen")
    resumen_text = (
        "La evaluación diagnóstica de los trastornos del sueño depende de la polisomnografía (PSG), "
        "un estudio nocturno en el que especialistas clasifican manualmente cerca de mil ventanas de 30 segundos "
        "(épocas) en cinco estadios definidos por la Academia Americana de Medicina del Sueño (AASM): vigilia (W), "
        "sueño no-REM (N1, N2, N3) y sueño de movimientos oculares rápidos (REM). Este proceso consume entre una y "
        "dos horas por paciente y sufre elevada variabilidad inter-evaluador, especialmente en el estadio transitorio N1. "
        "Como respuesta, se diseñó y desplegó SomnoScope, una plataforma clínica de apoyo al diagnóstico basada en "
        "inteligencia artificial (IA) que clasifica automáticamente señales de electroencefalografía (EEG) monocanal "
        "y opera bajo una arquitectura reproducible de Operaciones de Aprendizaje Automático (MLOps). Metodológicamente, "
        "se utilizó una submuestra estratificada de Sleep-EDF Expanded (18 registros de 9 sujetos, 18,604 épocas en el "
        "canal EEG Fpz-Cz), garantizando partición estricta por sujeto sin fuga de datos (6 sujetos para entrenamiento, "
        "12,537 épocas; 3 sujetos inéditos para prueba, 6,067 épocas). Se compararon modelos clásicos basados en 25 "
        "características espectrales (Random Forest y LightGBM) frente a una red neuronal convolucional unidimensional "
        "con memoria bidireccional (TinySleepNet1D). Al corregir el colapso de entrenamiento causado por la escala "
        "física en voltios mediante normalización z-score por época y guardado del mejor punto de control, "
        "TinySleepNet v2 alcanzó 77.95% de exactitud, 0.7317 de F1-Macro, 0.6985 de Kappa de Cohen y 0.4328 de F1 en "
        "el estadio prioritario N1, superando en +10.44 puntos porcentuales al mejor modelo tabular. El sistema quedó "
        "desplegado en Amazon Web Services mediante contenedores Docker (FastAPI, Streamlit y MLflow)."
    )
    res_word_count = len(resumen_text.split())
    print(f"[Check] Conteo de palabras del Resumen: {res_word_count} palabras (requisito: 200–300).")
    assert 200 <= res_word_count <= 300, f"Resumen fuera de rango: {res_word_count}"

    # Caja sombreada elegante para el Resumen
    tbl_abs = doc.add_table(rows=1, cols=1)
    tbl_abs.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_abs = tbl_abs.rows[0].cells[0]
    c_abs.width = Inches(6.95)
    set_cell_bg(c_abs, "F8F9FA")
    set_cell_margins(c_abs, top=110, bottom=110, left=140, right=140)
    p_ab = c_abs.paragraphs[0]
    p_ab.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_ab.paragraph_format.line_spacing = 1.15
    p_ab.paragraph_format.space_after = Pt(4)
    r_ab = p_ab.add_run(resumen_text)
    r_ab.font.name = "Montserrat Light"
    r_ab.font.size = Pt(9.0)
    r_ab.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    p_kw = c_abs.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(0)
    rk1 = p_kw.add_run("Palabras clave: ")
    rk1.font.name = "Montserrat SemiBold"
    rk1.font.size = Pt(9.0)
    rk1.font.bold = True
    rk1.font.color.rgb = RGBColor(0x5B, 0x57, 0xD1)
    rk2 = p_kw.add_run(
        "Estadificación automática del sueño; Electroencefalografía (EEG); Aprendizaje profundo; "
        "TinySleepNet; Operaciones de Aprendizaje Automático (MLOps)."
    )
    rk2.font.name = "Montserrat Light"
    rk2.font.size = Pt(9.0)
    rk2.font.italic = True
    rk2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # =========================================================================
    # 4. INTRODUCCIÓN
    # =========================================================================
    add_h1(doc, "1. Introducción")
    add_body(
        doc,
        "Los trastornos del sueño —como el insomnio crónico, la apnea obstructiva del sueño y la narcolepsia— "
        "constituyen un problema creciente de salud pública estrechamente asociado con enfermedades cardiovasculares, "
        "deterioro cognitivo y desregulación metabólica. El estándar de oro clínico para su evaluación es la "
        "polisomnografía (PSG), un estudio nocturno que registra múltiples señales bioeléctricas, principalmente "
        "electroencefalografía (EEG), electrooculografía (EOG) y electromiografía (EMG) [2]. Para interpretar una "
        "noche de registro, un técnico o médico especialista debe inspeccionar visualmente entre 800 y 1,200 ventanas "
        "consecutivas de 30 segundos (denominadas épocas) y asignar a cada una un estadio según los criterios de la "
        "Academia Americana de Medicina del Sueño (AASM): vigilia (W), sueño ligero (N1 y N2), sueño profundo de ondas "
        "lentas (N3) y sueño de movimientos oculares rápidos (REM) [2]."
    )
    add_body(
        doc,
        "A pesar de su valor diagnóstico, la estadificación manual presenta dos cuellos de botella críticos: primero, "
        "es una labor intensiva en tiempo que requiere entre una y dos horas por estudio, limitando la capacidad de "
        "atención en unidades de sueño; segundo, exhibe una variabilidad inter-evaluador documentada incluso entre "
        "expertos certificados. Estudios multicéntricos de la propia AASM reportan una concordancia global cercana al "
        "82.6%, pero que desciende drásticamente a alrededor del 63% en el estadio transitorio N1 debido a su corta "
        "duración y similitud morfológica con la vigilia relajada y el inicio de N2 [13]."
    )
    add_body(
        doc,
        "En este contexto, el objetivo general del proyecto consistió en diseñar, validar y desplegar una plataforma "
        "experimental de estadificación automática de sueño (SomnoScope) basada en aprendizaje automático sobre EEG "
        "de un solo canal, orientada a reducir el tiempo de revisión clínica sin reemplazar el criterio médico "
        "especialista. Como objetivos específicos se definieron: (i) construir un flujo de preprocesamiento clínico "
        "con partición estricta por sujeto para evitar la fuga de datos (data leakage); (ii) comparar de manera "
        "reproducible modelos clásicos basados en ingeniería de características espectrales frente a arquitecturas "
        "de aprendizaje profundo (Deep Learning), poniendo especial atención en la clase minoritaria N1; y "
        "(iii) desplegar el modelo ganador en una arquitectura desacoplada de microservicios con trazabilidad de "
        "experimentos y una interfaz clínica interactiva."
    )
    add_body(doc, "Las contribuciones principales de este trabajo son:", bold_prefix="Contribuciones del trabajo: ")
    add_bullet(
        doc,
        "Diagnóstico y corrección de escala en bioseñales para redes profundas: ",
        "Se identificó que la señal EEG entregada en voltios (~10⁻⁵ V) provoca el colapso de las capas de "
        "normalización por lotes (Batch Normalization), y se demostró que una normalización z-score independiente "
        "por época (libre de fuga entre sujetos) combinada con guardado del mejor punto de control (checkpointing) "
        "eleva el F1-Macro de TinySleepNet1D de 0.0511 a 0.7317 y el F1 en N1 de 0.0000 a 0.4328."
    )
    add_bullet(
        doc,
        "Comparativa controlada inter-sujeto: ",
        "Se evaluaron seis variantes supervisadas (Random Forest v1/v2, LightGBM v1/v2 y TinySleepNet v1/v2) bajo "
        "exactamente las mismas 6,067 épocas de pacientes completamente inéditos, demostrando superioridad del "
        "enfoque profundo en métricas globales y en N1."
    )
    add_bullet(
        doc,
        "Sistema MLOps desplegado de extremo a extremo: ",
        "Se empaquetó y desplegó SomnoScope en la nube (Amazon Web Services) con tres contenedores orquestados "
        "(API REST en FastAPI, tablero clínico en Streamlit con recorrido guiado interactivo y servidor MLflow), "
        "versionamiento de datos con Data Version Control (DVC) e IP elástica permanente."
    )

    # =========================================================================
    # 5. REVISIÓN DEL ESTADO DEL ARTE
    # =========================================================================
    add_h1(doc, "2. Revisión del Estado del Arte")
    add_body(
        doc,
        "La automatización de la estadificación del sueño ha evolucionado desde reglas heurísticas inspiradas en el "
        "manual clásico de Rechtschaffen y Kales [1] y su posterior actualización por la AASM [2], hacia paradigmas "
        "de aprendizaje supervisado clásico y redes neuronales profundas. La disponibilidad del conjunto público "
        "Sleep-EDF Expanded en PhysioNet [3], [4] ha servido como referencia estandarizada para comparar métodos "
        "sobre EEG monocanal (particularmente la derivación frontal-central Fpz-Cz)."
    )
    add_body(
        doc,
        "En una primera generación de soluciones, los modelos clásicos de aprendizaje automático combinaron "
        "extracción manual de descriptores espectrales —como la Densidad Espectral de Potencia (PSD) estimada por "
        "el método de Welch en bandas fisiológicas Delta, Theta, Alpha, Sigma y Beta— y parámetros temporales de "
        "Hjorth, alimentados a algoritmos de ensamble como Bosques Aleatorios (Random Forest) [9] o potenciación "
        "del gradiente basada en árboles (LightGBM) [8]. Estos enfoques destacan por su rapidez de entrenamiento, "
        "baja latencia de inferencia e interpretabilidad de variables, pero dependen fuertemente de la calidad de "
        "las características diseñadas manualmente y presentan dificultades para capturar microeventos transitorios "
        "como los husos de sueño, complejos K y ondas agudas de vértex propias del estadio N1."
    )
    add_body(
        doc,
        "Para superar esta limitación, la literatura reciente consolidó arquitecturas de aprendizaje profundo de "
        "extremo a extremo capaces de aprender representaciones directamente desde la señal cruda de EEG. "
        "DeepSleepNet [5] introdujo un extractor convolucional unidimensional (1D-CNN) de doble rama acoplado a "
        "redes recurrentes bidireccionales de memoria a corto y largo plazo (BiLSTM), alcanzando exactitudes "
        "cercanas al 82% en subconjuntos de Sleep-EDF. Posteriormente, TinySleepNet [6] simplificó drásticamente "
        "la arquitectura eliminando ramas redundantes y aplicando regularización y aumento de datos, logrando "
        "rendimientos competitivos (83.1%–85.4% de exactitud y ~0.46–0.51 de F1 en N1 sobre cohortes amplias de "
        "20 a 78 sujetos) con un tamaño de modelo mucho más compacto. Más recientemente, AttnSleep [7] incorporó "
        "mecanismos de atención multiescala y recalibración adaptativa para ponderar las bandas de frecuencia más "
        "relevantes, mitigando parcialmente el desbalance de clases."
    )
    add_body(
        doc,
        "A partir de esta revisión se identifican tres brechas clave que motivan el presente proyecto: "
        "(1) gran parte de los trabajos académicos permanecen como scripts o cuadernos de laboratorio sin integración "
        "en servicios productivos ni interfaces auditables que muestren la señal cruda junto a la certeza del modelo; "
        "(2) numerosos reportes aplicados omiten validar estrictamente por sujeto o subestiman la sensibilidad de las "
        "redes profundas a la escala física de los archivos clínicos en formato European Data Format (EDF); y "
        "(3) existe poca documentación práctica que articule el entrenamiento profundo con prácticas formales de "
        "MLOps (versionamiento de datos, registro de modelos Champion/Challenger, pruebas automatizadas y despliegue "
        "en contenedores). SomnoScope aborda directamente estas tres brechas integrando rigor metodológico e ingeniería "
        "de despliegue clínico.",
        bold_prefix="Análisis propio y brechas identificadas: "
    )

    # =========================================================================
    # 6. METODOLOGÍA (SIGUIENDO EL CICLO DEL MACHINE LEARNING)
    # =========================================================================
    add_h1(doc, "3. Metodología")
    add_body(
        doc,
        "El desarrollo siguió de forma estructurada el ciclo de vida completo de un sistema de Machine Learning: "
        "adquisición y versionamiento de datos, preprocesamiento clínico y partición anti-fuga, ingeniería de "
        "características y normalización, entrenamiento comparativo con registro en MLflow, validación automatizada "
        "y despliegue en microservicios."
    )

    add_h2(doc, "3.1 Datos, Preprocesamiento y Partición Estricta por Sujeto")
    add_body(
        doc,
        "La fuente de datos corresponde a Sleep-EDF Database Expanded [3], [4], compuesta por estudios de personas "
        "sanas (cohorte Sleep Cassette) y sujetos con dificultades leves de sueño (cohorte Sleep Telemetry). Con el "
        "fin de garantizar tiempos reproducibles de experimentación y portabilidad en contenedores dentro del entorno "
        "académico, se construyó y versionó mediante DVC en Amazon S3 (s3://microproyecto-maia) una submuestra "
        "estratificada de 729 MB integrada por 18 registros polisomnográficos completos (36 archivos .edf: 18 de "
        "señal PSG y 18 hipnogramas expertos), correspondientes a 9 sujetos únicos con dos noches registradas por "
        "paciente: 6 sujetos de Sleep Cassette (SC400 a SC405) y 3 sujetos de Sleep Telemetry (ST701, ST702 y ST704)."
    )
    add_body(
        doc,
        "El módulo de carga implementado con MNE-Python [10] estandariza los nombres de canales y selecciona la "
        "derivación clínica de referencia EEG Fpz-Cz muestreada a 100 Hz (estándar en los benchmarks de Sleep-EDF "
        "[5], [6]). Sobre la señal continua se aplica un filtro pasa-banda FIR entre 0.5 y 35.0 Hz para eliminar "
        "deriva basal y ruido eléctrico de red, se mapean las etiquetas originales de Rechtschaffen & Kales a las "
        "cinco clases AASM (fusionando estadios 3 y 4 en N3 y excluyendo épocas de movimiento 'M' y desconocidas '?'), "
        "se segmenta en épocas exactas de 30 segundos (3,000 muestras) y se recorta la vigilia diurna prolongada "
        "conservando 30 minutos antes del primer episodio de sueño y 30 minutos después del último. Este proceso "
        "produjo un corpus limpio de 18,604 épocas válidas."
    )
    add_body(
        doc,
        "En bioseñales, dividir aleatoriamente por épocas introduce una severa fuga de datos (data leakage) debido "
        "a que ventanas de un mismo individuo comparten impedancia, anatomía craneal y firmas espectrales propias. "
        "Para garantizar generalización clínica realista, se aplicó GroupShuffleSplit agrupando estrictamente por "
        "identificador de paciente (test_size=0.33, random_state=42): 6 sujetos conformaron el conjunto de "
        "entrenamiento (SC400, SC402, SC403, SC404, ST701 y ST704; 12,537 épocas, 67.4%) y 3 sujetos completamente "
        "inéditos conformaron el conjunto de prueba (SC401, SC405 y ST702; 6,067 épocas, 32.6%, distribuidas en "
        "888 W, 477 N1, 2,820 N2, 953 N3 y 929 REM), con solapamiento nulo entre pacientes.",
        bold_prefix="Prevención de fuga de datos: "
    )

    add_h2(doc, "3.2 Enfoques de Modelado y Normalización de Señales")
    add_bullet(
        doc,
        "Paradigma Clásico (Random Forest y LightGBM): ",
        "Para cada época de 30 segundos se extrajo un vector de 25 características: potencias espectrales absolutas "
        "y relativas (Welch PSD) en Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Sigma (12–16 Hz) y Beta "
        "(16–30 Hz); tres ratios clínicos (Delta/Theta, Theta/Alpha y ondas lentas/rápidas); siete estadísticos "
        "temporales (media, desviación estándar, varianza, asimetría, curtosis, amplitud pico a pico y tasa de "
        "cruces por cero); y los tres parámetros de Hjorth (actividad, movilidad y complejidad). Se evaluaron "
        "versiones baseline (v1) y optimizadas con regularización L1/L2 y ponderación de clases balanceada (v2) "
        "para Random Forest [9] y LightGBM [8]."
    )
    add_bullet(
        doc,
        "Paradigma de Aprendizaje Profundo (TinySleepNet1D): ",
        "Se implementó en PyTorch [11] una arquitectura compacta (650 KB en disco) inspirada en TinySleepNet [6], "
        "compuesta por tres bloques convolucionales 1D (32, 64 y 128 filtros con BatchNorm1d, activación ReLU, "
        "MaxPool/AdaptiveAvgPool y Dropout) seguidos de una capa recurrente BiLSTM (64 unidades ocultas por "
        "dirección) y un clasificador denso de cinco salidas. Para compensar el desbalance severo de N1 se empleó "
        "entropía cruzada ponderada inversamente a la frecuencia de clase y optimizador AdamW (lr=0.001, batch=64, "
        "10 épocas). Crucialmente, se incorporó normalización z-score independiente por época: "
        "x_norm = (x − μ_época) / (σ_época + 10⁻¹²), junto con guardado automático del mejor punto de control "
        "(checkpointing) según F1-Macro en validación. Al calcular μ y σ exclusivamente dentro de cada ventana de "
        "30 segundos, el procedimiento es idéntico en entrenamiento e inferencia y 100% libre de fuga de datos."
    )

    add_h2(doc, "3.3 Arquitectura de la Solución, Herramientas y Despliegue")
    add_body(
        doc,
        "El sistema completo se implementó en Python 3.12 integrando PyTorch, LightGBM, scikit-learn, MNE-Python, "
        "FastAPI, Streamlit, Plotly y MLflow [12]. La Figura 1 ilustra la arquitectura de extremo a extremo "
        "orquestada mediante Docker Compose sobre una instancia Amazon EC2 (Ubuntu 22.04 LTS, t2.medium con 4 GB "
        "de RAM, 2 GB de memoria Swap y 20 GB SSD) asociada a una Elastic IP estática (52.207.88.72) para asegurar "
        "URLs permanentes ante reinicios del laboratorio."
    )
    add_figure(
        doc, arch_fig,
        "Figura 1. Arquitectura integral de SomnoScope: flujo de datos (DVC/S3), partición estricta por sujeto, "
        "entrenamiento dual registrado en MLflow y despliegue en tres contenedores Docker sobre AWS EC2.",
        width_in=6.75
    )
    add_body(
        doc,
        "El despliegue desacopla tres microservicios en una red interna Docker: (1) somnoscope-api (FastAPI v3.0, "
        "puerto 8000), que sirve el modelo Champion TinySleepNet1D para inferencia sobre señal cruda (/predict/epoch "
        "y /predict/recording con archivos .edf completos), conserva LightGBM para el contrato tabular "
        "(/predict/features) y expone monitoreo continuo (/health) y documentación Swagger UI (/docs); "
        "(2) somnoscope-dashboard (Streamlit v3.0, puerto 8050), que permite explorar los 18 registros precargados "
        "o cargar archivos .edf externos, alternar en vivo entre TinySleepNet v2 (Champion) y LightGBM (Challenger), "
        "probar transferencia entre derivaciones (EEG Fpz-Cz y respaldo cross-channel EEG Pz-Oz), visualizar los "
        "KPIs clínicos (TIB, TST, Eficiencia, SOL, WASO), el hipnograma dual, el inspector microscópico de 30s, "
        "la matriz de confusión por sujeto y un recorrido guiado interactivo de 6 pasos diseñado para evaluadores; "
        "y (3) somnoscope-mlflow (puerto 5000), con base de datos SQLite persistente en volumen Docker para "
        "gobernanza del Model Registry (@champion, @production y @challenger)."
    )

    # =========================================================================
    # 7. RESULTADOS Y EVALUACIÓN
    # =========================================================================
    add_h1(doc, "4. Resultados y Evaluación")
    add_body(
        doc,
        "Dado el fuerte desbalance de clases inherente al sueño humano —donde N2 concentra casi la mitad de la "
        "noche mientras N1 representa menos del 8% de las épocas de prueba—, la evaluación no se apoyó únicamente "
        "en la exactitud global (Accuracy), sino principalmente en el F1-Score Macro (promedio no ponderado entre "
        "los cinco estadios), el coeficiente Kappa de Cohen (concordancia corregida por azar) y el F1-Score "
        "discriminado por clase con énfasis en el estadio prioritario N1. La Tabla 1 consolida el desempeño "
        "verificado de las seis variantes sobre exactamente las mismas 6,067 épocas de prueba inter-sujeto."
    )

    add_h2(doc, "Tabla 1. Comparativa de Desempeño Inter-Sujeto sobre 6,067 Épocas Inéditas (SC401, SC405, ST702)")
    t1_headers = ["Modelo", "Versión / Rol", "Acc.", "F1-Mac", "Kappa", "F1-W", "F1-N1*", "F1-N2", "F1-N3", "F1-REM"]
    t1_rows = [
        ["TinySleepNet1D", "v2 Norm. (Champion)", "77.95%", "0.7317", "0.6985", "0.859", "0.433", "0.837", "0.767", "0.763"],
        ["LightGBM", "v2 Tuned (Challenger)", "77.91%", "0.6942", "0.6886", "0.748", "0.328", "0.874", "0.804", "0.717"],
        ["LightGBM", "v1 Baseline", "77.95%", "0.6942", "0.6879", "0.759", "0.322", "0.873", "0.800", "0.718"],
        ["Random Forest", "v1 Baseline", "76.84%", "0.6690", "0.6696", "0.729", "0.284", "0.869", "0.760", "0.703"],
        ["Random Forest", "v2 Tuned", "77.06%", "0.6641", "0.6693", "0.748", "0.256", "0.867", "0.745", "0.704"],
        ["TinySleepNet1D", "v1 Sin normalizar", "14.64%", "0.0511", "0.0000", "0.255", "0.000", "0.000", "0.000", "0.000"],
    ]
    add_styled_table(
        doc, t1_headers, t1_rows,
        col_widths=[1.12, 1.33, 0.56, 0.56, 0.56, 0.54, 0.58, 0.54, 0.56, 0.60],
        highlight_row_idx=0, font_pt=7.5
    )
    add_caption(doc, "*N1 corresponde al estadio prioritario definido en el alcance del proyecto. Todos los modelos se evaluaron sobre la misma partición por sujeto.")

    add_figure(
        doc, comp_fig,
        "Figura 2. Comparativa de métricas globales (Panel A) y F1-Score discriminado por estadio AASM (Panel B) "
        "en la cohorte de prueba de 6,067 épocas. Se destaca la ganancia de +10.44 puntos porcentuales de TinySleepNet v2 en N1 frente a LightGBM v2.",
        width_in=6.75
    )

    if cm_fig.exists():
        add_figure(
            doc, cm_fig,
            "Figura 3. Matriz de confusión del modelo Champion TinySleepNet v2 (Normalizado, época 10) sobre las 6,067 "
            "épocas de prueba de sujetos inéditos (SC401, SC405 y ST702).",
            width_in=4.7
        )

    add_h2(doc, "Comparación con Trabajos Previos y Análisis Crítico")
    t2_headers = ["Trabajo / Referencia", "Cohorte Sleep-EDF", "Contexto Temporal", "Accuracy", "F1-Macro", "Kappa", "F1-N1"]
    t2_rows = [
        ["DeepSleepNet [5]", "20 sujetos (39 noches)", "Secuencias (Seq2Seq)", "82.0%", "0.769", "0.76", "0.466"],
        ["TinySleepNet original [6]", "20 sujetos (39 noches)", "Secuencias (Seq2Seq)", "85.4%", "0.805", "0.80", "0.514"],
        ["AttnSleep [7]", "20 sujetos (39 noches)", "Época individual", "84.4%", "0.781", "0.79", "0.420"],
        ["SomnoScope — TinySleepNet v2", "9 sujetos (18 noches)", "Época individual", "77.95%", "0.732", "0.70", "0.433"],
    ]
    add_styled_table(
        doc, t2_headers, t2_rows,
        col_widths=[1.55, 1.35, 1.15, 0.72, 0.72, 0.68, 0.78],
        highlight_row_idx=3, font_pt=7.5
    )
    add_caption(doc, "Tabla 2. Referenciación de SomnoScope frente a trabajos del estado del arte sobre Sleep-EDF (canal EEG Fpz-Cz).")

    add_body(
        doc,
        "El contraste empírico arroja tres hallazgos determinantes. Primero, la normalización z-score por época "
        "resolvió por completo el colapso numérico de TinySleepNet v1 (donde la varianza ~10⁻¹⁰ V² era dominada "
        "por el término ε = 10⁻⁵ de BatchNorm1d), saltando de 0.0511 a 0.7317 en F1-Macro. Segundo, aunque LightGBM "
        "v2 obtuvo una exactitud global prácticamente idéntica (77.91% vs. 77.95%) y lideró marginalmente en N2 "
        "(0.874) y sueño profundo N3 (0.804 vs. 0.767) —estadios definidos por potencia delta sostenida fácilmente "
        "capturable por Welch PSD—, TinySleepNet v2 superó con claridad a todos los enfoques clásicos en los estadios "
        "de mayor complejidad morfológica: Vigilia (+11.1 pp), REM (+4.6 pp) y, críticamente, N1 (+10.44 pp, alcanzando "
        "0.4328). Tercero, en la Tabla 2 se observa que nuestro F1 en N1 (0.433) ya iguala o supera al reportado por "
        "AttnSleep [7] (0.420) operando época a época, aun entrenando con menos de la mitad de los sujetos; la brecha "
        "restante en exactitud global frente a DeepSleepNet y TinySleepNet original (~4–7 pp) se explica directamente "
        "por el menor número de sujetos en entrenamiento (6 vs. 19 en validación cruzada) y por la ausencia de "
        "modelado secuencial inter-época.",
        bold_prefix="Análisis crítico de los resultados: "
    )

    # =========================================================================
    # 8. DISCUSIÓN
    # =========================================================================
    add_h1(doc, "5. Discusión")
    add_body(
        doc,
        "Los resultados validan que las representaciones aprendidas mediante convoluciones unidimensionales sobre "
        "EEG normalizado capturan patrones transitorios que los descriptores espectrales agregados pierden al "
        "promediar ventanas de 30 segundos. Desde el punto de vista clínico, alcanzar un coeficiente Kappa cercano "
        "a 0.70 (acuerdo sustancial) y un F1-N1 de 0.433 en pacientes inéditos —cuando la concordancia entre "
        "especialistas humanos en N1 ronda el 63% [13]— demuestra viabilidad real como herramienta de pre-etiquetado "
        "y triaje asistido. Asimismo, la coexistencia de TinySleepNet v2 (Champion) y LightGBM v2 (Challenger) "
        "dentro del mismo ecosistema MLOps aporta resiliencia operativa y trazabilidad completa: las pruebas "
        "automatizadas (8/8 en pytest) verificaron integridad tanto en inferencia sobre señal cruda como en "
        "contratos tabulares."
    )
    add_body(doc, "Las principales fortalezas del enfoque propuesto incluyen:", bold_prefix="Fortalezas: ")
    add_bullet(
        doc, "Rigor metodológico sin fuga de datos: ",
        "Tanto la partición por sujeto (GroupShuffleSplit) como la normalización estrictamente intra-época "
        "garantizan métricas honestas extrapolables a nuevos pacientes."
    )
    add_bullet(
        doc, "Auditabilidad clínica y usabilidad: ",
        "El tablero SomnoScope expone la señal cruda de 30 s, probabilidades calibradas, comparación dual de modelos "
        "y un recorrido guiado interactivo, permitiendo al especialista auditar rápidamente épocas dudosas."
    )
    add_bullet(
        doc, "Eficiencia computacional: ",
        "El modelo Champion ocupa solo 650 KB, corre sobre CPU estándar sin requerir GPU y procesa una noche "
        "completa de ~900 épocas en menos de tres segundos."
    )

    add_body(doc, "En contraste, es importante reconocer las debilidades y limitaciones actuales:", bold_prefix="Debilidades y limitaciones: ")
    add_bullet(
        doc, "Tamaño muestral de entrenamiento: ",
        "La cohorte versionada en DVC contiene 9 sujetos (6 de entrenamiento y 3 de prueba), lo que introduce "
        "varianza dependiente del subconjunto de pacientes y limita la cobertura de edades extremas y patologías "
        "severas como apnea obstructiva."
    )
    add_bullet(
        doc, "Clasificación época-a-época y modalidad monocanal: ",
        "Al evaluar cada ventana de 30 s de forma independiente y únicamente con EEG (sin EOG ni EMG), persisten "
        "confusiones esperables entre N1, vigilia relajada y REM (que clínicamente se discriminan observando "
        "movimientos oculares rápidos y atonía muscular submentoniana)."
    )
    add_bullet(
        doc, "Alcance experimental: ",
        "Aunque el sistema permite cargar archivos .edf externos e inferir en derivación Pz-Oz como respaldo "
        "cross-channel, las métricas oficiales corresponden a Fpz-Cz y el prototipo carece de validación clínica "
        "prospectiva o certificación regulatoria."
    )

    # =========================================================================
    # 9. CONCLUSIONES Y TRABAJO FUTURO
    # =========================================================================
    add_h1(doc, "6. Conclusiones y Trabajo Futuro")
    add_body(
        doc,
        "Este trabajo demostró el diseño, validación rigurosa y puesta en producción de SomnoScope, una solución "
        "integral de IA para estadificación automática de sueño sobre EEG monocanal. Al sustituir la ingesta cruda "
        "en voltios por normalización z-score por época con checkpointing de mejor estado, TinySleepNet1D v2 superó "
        "de manera consistente a Random Forest y LightGBM sobre 6,067 épocas de pacientes inéditos (77.95% de "
        "exactitud, 0.7317 de F1-Macro, 0.6985 de Kappa y 0.4328 de F1 en N1), quedando promovido oficialmente "
        "como Champion en MLflow y operativo en producción sobre AWS EC2 con Docker Compose y Elastic IP."
    )
    add_body(
        doc,
        "En términos prácticos, la plataforma habilita aplicaciones inmediatas como herramienta de pre-estadificación "
        "en laboratorios de sueño, apoyo docente en neurofisiología y base tecnológica para servicios de telemedicina "
        "y tamizaje ambulatorio del sueño. Entre las lecciones aprendidas destacan que: (i) en bioseñales, el "
        "entendimiento de la escala física (voltios vs. microvoltios) es tan crítico como la arquitectura neuronal; "
        "(ii) la partición estricta por sujeto es innegociable para evitar sobreestimaciones engañosas; y "
        "(iii) gobernar modelos con MLflow y contenedores Docker acelera significativamente el paso del experimento "
        "a un servicio verificable."
    )
    add_body(doc, "Como líneas de trabajo futuro para extender esta solución se proponen:", bold_prefix="Trabajo futuro: ")
    add_bullet(
        doc, "Escalamiento del corpus y validación cruzada k-fold por sujeto: ",
        "Extender el entrenamiento a los benchmarks completos Sleep-EDF-20 (39 registros) y Sleep-EDF-78 (153 "
        "registros) empleando validación cruzada estratificada por sujeto (GroupKFold) para reducir la varianza."
    )
    add_bullet(
        doc, "Modelado secuencial inter-época (Sequence-to-Sequence): ",
        "Incorporar ventanas de contexto de 15 a 20 épocas consecutivas sobre las representaciones CNN para "
        "aprender las transiciones fisiológicas de la macro-arquitectura del sueño y acercarse al 84–85% de exactitud."
    )
    add_bullet(
        doc, "Fusión multimodal (EEG + EOG + EMG): ",
        "Aprovechar el parámetro in_channels ya previsto en TinySleepNet1D para integrar simultáneamente EEG "
        "Fpz-Cz, EEG Pz-Oz, EOG horizontal y EMG submentoniano, atacando directamente la frontera clínica entre "
        "N1, vigilia y REM."
    )

    # =========================================================================
    # 10. REFERENCIAS (ESTILO IEEE CONSISTENTE)
    # =========================================================================
    add_h1(doc, "7. Referencias")
    refs = [
        "[1] A. Rechtschaffen and A. Kales, Eds., A Manual of Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human Subjects. Washington, DC, USA: U.S. Government Printing Office, 1968.",
        "[2] R. B. Berry et al., \"The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications,\" American Academy of Sleep Medicine, Darien, IL, USA, Version 2.4, 2017.",
        "[3] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, \"Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG,\" IEEE Trans. Biomed. Eng., vol. 47, no. 9, pp. 1185–1194, Sep. 2000. [Dataset: Sleep-EDF Database Expanded, PhysioNet].",
        "[4] A. L. Goldberger et al., \"PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,\" Circulation, vol. 101, no. 23, pp. e215–e220, Jun. 2000.",
        "[5] A. Supratak, H. Dong, C. Wu, and Y. Guo, \"DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 11, pp. 1998–2008, Nov. 2017.",
        "[6] A. Supratak and Y. Guo, \"TinySleepNet: An efficient deep learning model for sleep stage scoring based on raw single-channel EEG,\" in Proc. 42nd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), Montreal, QC, Canada, Jul. 2020, pp. 641–644.",
        "[7] E. Eldele et al., \"An attention-based deep learning approach for sleep stage classification with single-channel EEG,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 29, pp. 809–818, 2021.",
        "[8] G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 3146–3154.",
        "[9] L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, Oct. 2001.",
        "[10] A. Gramfort et al., \"MEG and EEG data analysis with MNE-Python,\" Front. Neurosci., vol. 7, art. 267, Dec. 2013.",
        "[11] A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019, pp. 8024–8035.",
        "[12] M. Zaharia et al., \"Accelerating the machine learning lifecycle with MLflow,\" IEEE Data Eng. Bull., vol. 41, no. 4, pp. 39–45, Dec. 2018.",
        "[13] R. S. Rosenberg and S. Van Hout, \"The American Academy of Sleep Medicine inter-scorer reliability program: Sleep stage scoring,\" J. Clin. Sleep Med., vol. 9, no. 1, pp. 81–87, Jan. 2013.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.left_indent = Inches(0.28)
        p.paragraph_format.first_line_indent = Inches(-0.28)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(ref)
        r.font.name = "Montserrat Light"
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    doc.save(dst_path)

    total_words = sum(len(p.text.split()) for p in doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                total_words += sum(len(p.text.split()) for p in cell.paragraphs)

    print(f"[OK] Artículo científico generado exitosamente: {dst_path}")
    print(f"     Párrafos: {len(doc.paragraphs)} | Tablas: {len(doc.tables)} | Palabras aprox: {total_words}")


if __name__ == "__main__":
    build_article()
