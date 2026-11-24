"""
Generador del Artículo Científico Final de SomnoScope (Formato Académico PDF + Rúbrica ≤12 pp):
'Articulo Cientifico Final - SomnoScope (23 Sujetos + CV).docx'

Cumple estrictamente las Recomendaciones de Formato:
  - Papel: Carta (8.5 x 11 pulgadas)
  - Márgenes: 2.5 cm en los cuatro bordes
  - Fuente: Times New Roman, tamaño 10 ptos (cuerpo)
  - Interlineado: Sencillo (1.0)
  - Alineación: Justificada
  - Títulos numerados en negrita (1, 2, 3...) y subsecciones (3.1, 3.2...)
  - Ecuaciones centradas con numeración entre paréntesis a la derecha: (1), (2), (3)
  - Figuras 300 DPI interpretables en escala de grises, título DEBAJO con "Elaboración propia."
  - Tablas numeradas consecutivamente, título ENCIMA
  - Citas estilo IEEE numérico en estricto orden de aparición
  - Extensión controlada: <= 12 páginas (rúbrica de evaluación)
"""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import docx
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = Path(r"G:\Mi unidad\MAIA\Proyecto_Desarrollo_de_Soluciones\MicroP1MAIA")
FIG_DIR = REPO_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. FIGURAS CIENTÍFICAS (300 DPI, INTERPRETABLES EN ESCALA DE GRISES)
# ==============================================================================
def generate_architecture_figure(out_path: Path):
    """Figura 1: Arquitectura integral SomnoScope (alto contraste + legible en escala de grises)."""
    fig, ax = plt.subplots(figsize=(10.5, 4.9), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    def draw_box(x, y, w, h, title, body, fc, ec, title_c="#111827", lw=1.4):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=title_c, family="serif")
        ax.text(x + w / 2, y + (h - 0.31) / 2, body, ha="center", va="center",
                fontsize=6.9, color="#1f2937", linespacing=1.30, family="serif")

    def arrow(x1, y1, x2, y2, color="#1f2937"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=11))

    layers = [
        (0.12, 0.15, 3.08, 4.55, "#f9fafb", "#4b5563", "1. DATOS Y VALIDACIÓN CRUZADA"),
        (3.48, 0.15, 3.55, 4.55, "#f3f4f6", "#1f2937", "2. MODELADO Y GOBERNANZA MLOPS"),
        (7.30, 0.15, 3.08, 4.55, "#f9fafb", "#4b5563", "3. DESPLIEGUE PRODUCTIVO (AWS)"),
    ]
    for lx, ly, lw, lh, lfc, lec, ltitle in layers:
        bg = patches.FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=lfc, edgecolor=lec, linewidth=1.1, linestyle="--")
        ax.add_patch(bg)
        ax.text(lx + lw / 2, ly + lh - 0.18, ltitle, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="#111827", family="serif")

    draw_box(0.28, 3.25, 2.76, 1.02, "Sleep-EDF-20 + Telemetry",
             "23 sujetos únicos · 45 noches\n20 Sleep Cassette + 3 Telemetry\n48,378 épocas válidas de 30 s",
             "#ffffff", "#374151")
    draw_box(0.28, 1.85, 2.76, 1.05, "Preprocesamiento Clínico (MNE)",
             "Canal estándar EEG Fpz-Cz (100 Hz)\nFiltro pasa-banda 0.5–35 Hz (FIR)\nExclusión M/? · Recorte ±30 min",
             "#ffffff", "#374151")
    draw_box(0.28, 0.35, 2.76, 1.12, "3-Fold GroupKFold por Sujeto",
             "0% solapamiento entre pacientes\n23/23 sujetos evaluados OOF\nPesos inversos solo en train",
             "#e5e7eb", "#111827", lw=1.6)

    arrow(1.66, 3.25, 1.66, 2.94)
    arrow(1.66, 1.85, 1.66, 1.51)

    draw_box(3.66, 3.12, 3.19, 1.15, "Deep Learning: TinySleepNet v2 (Champion)",
             "Z-score por época (Ec. 1) · 1D-CNN + BiLSTM\nCrossEntropyLoss ponderada (Ec. 2) · 0.62 MB\nCV3 F1-Mac 0.716±0.05 · 1.65 s / 1,000 ép.",
             "#e0e7ff", "#1e1b4b", "#1e1b4b", lw=1.8)
    draw_box(3.66, 1.72, 3.19, 1.08, "ML Clásico: LightGBM & Random Forest",
             "25 características Welch PSD + Hjorth\nclass_weight='balanced' (Ec. 2)\nCV3 F1-Mac 0.694 (LGBM v1) · 4.00 s / noche",
             "#ffffff", "#374151")
    draw_box(3.66, 0.35, 3.19, 1.02, "MLflow Nested CV & Registry (:5000)",
             "Parent Runs (Media±DE) + 3 Child Folds\n@champion/@production: TinySleepNet v2\n@challenger: LightGBM v2",
             "#f3f4f6", "#111827", lw=1.5)

    arrow(3.06, 1.10, 3.63, 3.60)
    arrow(3.06, 0.90, 3.63, 2.25)
    arrow(5.25, 1.72, 5.25, 1.42)
    ax.annotate("", xy=(6.88, 1.05), xytext=(6.88, 3.35),
                arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.3, mutation_scale=11,
                                connectionstyle="arc3,rad=-0.16"))

    draw_box(7.46, 3.22, 2.76, 1.05, "FastAPI REST v3.0 (:8000)",
             "GET /health · /docs (OpenAPI)\nPOST /predict/epoch & /recording\nPOST /predict/features (LGBM)",
             "#ffffff", "#374151")
    draw_box(7.46, 1.80, 2.76, 1.08, "Dashboard SomnoScope (:8050)",
             "Streamlit v3.0 + Plotly interactivo\nSelector Champion vs. Challenger\nCarga .edf · Hipnograma · KPIs",
             "#ffffff", "#374151")
    draw_box(7.46, 0.35, 2.76, 1.05, "Docker Compose + AWS EC2",
             "t2.medium (2 vCPU, 4 GB RAM, 2 GB Swap)\nContenedores: api · dashboard · mlflow\nElastic IP fija: 52.207.88.72",
             "#e5e7eb", "#111827", lw=1.6)

    arrow(6.88, 3.70, 7.43, 3.70)
    arrow(6.88, 2.28, 7.43, 2.28)
    arrow(8.84, 3.22, 8.84, 2.92)
    arrow(6.88, 0.86, 7.43, 0.86)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_cv_and_scaling_figure(cv: dict, out_path: Path):
    """Figura 2: Panel A (3-Fold CV Media±DE) + Panel B (Escalamiento 9 vs 23 sujetos). Legible en escala de grises con tramas."""
    s = cv["summary_mean_std"]
    models = [
        ("TinySleepNet v2\n(Champion)", "TinySleepNet v2 (Normalizado)"),
        ("LightGBM v1\n(Base)", "LightGBM v1"),
        ("LightGBM v2\n(Tuned)", "LightGBM v2"),
        ("Random Forest\nv1 (Base)", "Random Forest v1"),
    ]
    labels = [m[0] for m in models]
    f1m = [s[m[1]]["f1_macro_mean"] * 100 for m in models]
    f1m_sd = [s[m[1]]["f1_macro_std"] * 100 for m in models]
    kap = [s[m[1]]["cohen_kappa_mean"] * 100 for m in models]
    kap_sd = [s[m[1]]["cohen_kappa_std"] * 100 for m in models]
    n1 = [s[m[1]]["f1_N1_mean"] * 100 for m in models]
    n1_sd = [s[m[1]]["f1_N1_std"] * 100 for m in models]

    plt.rcParams["font.family"] = "serif"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=300)

    x = np.arange(len(labels))
    w = 0.25
    b1 = ax1.bar(x - w, f1m, w, yerr=f1m_sd, capsize=3, label="F1-Macro (×100)",
                 color="#312e81", edgecolor="#111827", linewidth=0.7, error_kw={"elinewidth": 1.0})
    b2 = ax1.bar(x, kap, w, yerr=kap_sd, capsize=3, label="Kappa de Cohen (×100)",
                 color="#64748b", edgecolor="#111827", linewidth=0.7, hatch="///", error_kw={"elinewidth": 1.0})
    b3 = ax1.bar(x + w, n1, w, yerr=n1_sd, capsize=3, label="F1-N1 (×100)",
                 color="#cbd5e1", edgecolor="#111827", linewidth=0.7, hatch="...", error_kw={"elinewidth": 1.0})

    for bar, val, sd in zip(b1, f1m, f1m_sd):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + sd + 1.4, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111827")
    for bar, val, sd in zip(b3, n1, n1_sd):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + sd + 1.4, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111827")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.0)
    ax1.set_ylim(0, 98)
    ax1.set_ylabel("Métrica (escala 0–100)", fontsize=8.5, fontweight="bold")
    ax1.set_title("A. Validación Cruzada 3-Fold GroupKFold (23 sujetos)", fontsize=9.2, fontweight="bold", pad=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8")
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.3, loc="upper right", framealpha=0.96)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Head-to-head scaling on 7 unseen subjects (15,247 epochs)
    metrics_h2h = ["Accuracy", "F1-Macro", "Kappa", "F1-N3", "F1-REM", "F1-N1"]
    m9 = [72.00, 67.35, 64.09, 74.66, 71.10, 33.63]
    m23 = [76.24, 71.38, 68.93, 86.56, 74.40, 31.84]
    x2 = np.arange(len(metrics_h2h))
    w2 = 0.34
    ax2.bar(x2 - w2 / 2, m9, w2, label="TinySleepNet (9 sujetos / 18 noches)",
            color="#cbd5e1", edgecolor="#111827", linewidth=0.7, hatch="///")
    bars23 = ax2.bar(x2 + w2 / 2, m23, w2, label="TinySleepNet (23 sujetos / 45 noches)",
                     color="#312e81", edgecolor="#111827", linewidth=0.7)

    for i, (v9, v23) in enumerate(zip(m9, m23)):
        diff = v23 - v9
        sign = "+" if diff >= 0 else ""
        ax2.text(x2[i] + w2 / 2, v23 + 1.5, f"{sign}{diff:.1f}",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111827")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_h2h, fontsize=8.0)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Porcentaje / Escala ×100", fontsize=8.5, fontweight="bold")
    ax2.set_title("B. Escalamiento en 7 Sujetos Inéditos (15,247 épocas)", fontsize=9.2, fontweight="bold", pad=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, color="#94a3b8")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.3, loc="upper right", framealpha=0.96)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_compact_confusion_matrices(oof_details: dict, out_path: Path):
    """Figura 3: 4 matrices de confusión OOF (todos los modelos) en 1 figura 2x2, normalizadas por fila, escala fija [0,1]."""
    plt.rcParams["font.family"] = "serif"
    labels = ["W", "N1", "N2", "N3", "REM"]
    cms = oof_details["oof_normalized_cms"]

    panels = [
        ("(a) Random Forest v1 (F1-Mac: 0.681)", np.array(cms["Random Forest v1"])),
        ("(b) LightGBM v1 (F1-Mac: 0.694)", np.array(cms["LightGBM v1"])),
        ("(c) LightGBM v2 Tuned (F1-Mac: 0.692)", np.array(cms["LightGBM v2"])),
        ("(d) TinySleepNet v2 Champion (F1-Mac: 0.716)", np.array(cms["TinySleepNet v2 (Normalizado)"])),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.6), dpi=300, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.022, 0.73])

    for idx, (ax, (title, mat)) in enumerate(zip(axes.flat, panels)):
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=(idx == 0),
            cbar_ax=(cbar_ax if idx == 0 else None),
            cbar_kws={"label": "Proporción normalizada por fila (sensibilidad)"} if idx == 0 else None,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.4,
            linecolor="#e2e8f0",
            annot_kws={"size": 8.5, "weight": "bold"},
        )
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
        if idx in (0, 2):
            ax.set_ylabel("Estadio Real (AASM)", fontsize=8.8, fontweight="bold")
        else:
            ax.set_ylabel("")
        if idx in (2, 3):
            ax.set_xlabel("Estadio Predicho", fontsize=8.8, fontweight="bold")
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=8.3)

    cbar_ax.tick_params(labelsize=8.2)
    cbar_ax.set_ylabel("Proporción normalizada por fila", fontsize=8.6, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.89, top=0.94, bottom=0.08, wspace=0.12, hspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==============================================================================
# 2. UTILIDADES DE FORMATO WORD (PDF: CARTA, 2.5 CM, TIMES NEW ROMAN 10, SENCILLO)
# ==============================================================================
def set_cell_shading(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=45, bottom=45, left=75, right=75):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_academic_table_borders(table):
    """Bordes limpios estilo revista IEEE/académica (reglas horizontales superior/inferior, sin rejilla vertical)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def set_borderless_table(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/><w:bottom w:val="none"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/>'
        f'<w:insideH w:val="none"/><w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r1 = p.add_run("SomnoScope — Artículo Científico Final (MAIA, Universidad de los Andes)   |   Página ")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(8.5)
    r1.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    r_pg = p.add_run()
    r_pg.font.name = "Times New Roman"
    r_pg.font.size = Pt(8.5)
    r_pg.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)
    r_pg._r.append(fld1)
    r_pg._r.append(instr)
    r_pg._r.append(fld2)


def add_h1(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_h2(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_body(doc, text: str, bold_prefix: str = None, space_after: float = 4.0, italic: bool = False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.0
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(10)
        rb.font.bold = True
        rb.font.italic = italic
        rb.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.font.italic = italic
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_bullet(doc, bold_prefix: str, text: str):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.0
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(10)
        rb.font.bold = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    return p


def add_equation(doc, formula_parts, eq_number: str):
    """
    Ecuación centrada con numeración entre paréntesis alineada a la derecha (formato PDF).
    formula_parts: lista de tuplas (texto, italic, sub, sup)
    """
    tbl = doc.add_table(rows=1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borderless_table(tbl)
    widths = [Inches(0.55), Inches(5.43), Inches(0.55)]
    for i, w in enumerate(widths):
        tbl.rows[0].cells[i].width = w
        set_cell_margins(tbl.rows[0].cells[i], top=30, bottom=30, left=20, right=20)
        tbl.rows[0].cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    p_mid = tbl.rows[0].cells[1].paragraphs[0]
    p_mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mid.paragraph_format.space_before = Pt(1)
    p_mid.paragraph_format.space_after = Pt(1)
    p_mid.paragraph_format.line_spacing = 1.0
    for txt, is_it, is_sub, is_sup in formula_parts:
        run = p_mid.add_run(txt)
        run.font.name = "Cambria Math"
        run.font.size = Pt(10)
        run.font.italic = is_it
        run.font.subscript = is_sub
        run.font.superscript = is_sup

    p_num = tbl.rows[0].cells[2].paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(1)
    p_num.paragraph_format.space_after = Pt(1)
    r_num = p_num.add_run(f"({eq_number})")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(10)


def add_academic_table(doc, caption_title: str, headers: list, rows: list, col_widths=None, highlight_row=None):
    """Tabla académica con título ENCIMA (regla PDF)."""
    pc = doc.add_paragraph()
    pc.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pc.paragraph_format.space_before = Pt(5)
    pc.paragraph_format.space_after = Pt(2)
    pc.paragraph_format.line_spacing = 1.0
    pc.paragraph_format.keep_with_next = True
    parts = caption_title.split(".", 1)
    r_lbl = pc.add_run(parts[0] + ".")
    r_lbl.font.name = "Times New Roman"
    r_lbl.font.size = Pt(9.5)
    r_lbl.font.bold = True
    if len(parts) > 1:
        r_desc = pc.add_run(parts[1])
        r_desc.font.name = "Times New Roman"
        r_desc.font.size = Pt(9.5)

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_academic_table_borders(table)

    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        if col_widths:
            cell.width = Inches(col_widths[j])
        set_cell_shading(cell, "E5E7EB")
        set_cell_margins(cell, top=40, bottom=40, left=55, right=55)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)
        r.font.bold = True

    for i, row in enumerate(rows):
        is_hl = (highlight_row is not None and i == highlight_row)
        bg = "F3F4F6" if is_hl else ("FFFFFF" if i % 2 == 0 else "FAFAFA")
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            if col_widths:
                cell.width = Inches(col_widths[j])
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=32, bottom=32, left=55, right=55)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.3)
            r.font.bold = is_hl

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3)
    return table


def add_figure_block(doc, img_path: Path, caption_label: str, caption_text: str, width_in: float = 6.1):
    """Figura centrada con título DEBAJO e indicación de elaboración propia (regla PDF)."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(5)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(5)
    p_cap.paragraph_format.line_spacing = 1.0
    r1 = p_cap.add_run(caption_label + " ")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(9)
    r1.font.bold = True
    r2 = p_cap.add_run(caption_text)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(9)
    r2.font.italic = True


# ==============================================================================
# 3. CONSTRUCCIÓN DEL ARTÍCULO CIENTÍFICO COMPLETO
# ==============================================================================
def build_article():
    cv = json.loads((REPO_ROOT / "reports" / "cv_groupkfold_23subjects.json").read_text(encoding="utf-8"))
    oof_det = json.loads((REPO_ROOT / "reports" / "cv_oof_details_23subjects.json").read_text(encoding="utf-8"))
    s = cv["summary_mean_std"]
    lat = oof_det["latencies_cpu"]
    cnt = oof_det["class_counts"]
    pct = oof_det["class_pcts"]

    # Generar las 3 figuras 300 DPI interpretables en escala de grises
    fig1_path = FIG_DIR / "arquitectura_somnoscope_cv23.png"
    fig2_path = FIG_DIR / "comparativa_cv23_estadios.png"
    fig3_path = FIG_DIR / "confusion_matrix_cv_all4_23subj.png"
    generate_architecture_figure(fig1_path)
    generate_cv_and_scaling_figure(cv, fig2_path)
    generate_compact_confusion_matrices(oof_det, fig3_path)

    doc = docx.Document()

    # Configurar página Carta (8.5 x 11 in) y márgenes exactos de 2.5 cm
    for sec in doc.sections:
        sec.page_width = Inches(8.5)
        sec.page_height = Inches(11.0)
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)
        add_page_number_footer(sec)

    # Configurar estilo Normal: Times New Roman 10 pt, interlineado sencillo, justificado
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ------------------------------------------------------------------
    # TÍTULO Y AUTORES (ESTILO ARTÍCULO CIENTÍFICO EN PÁGINA 1)
    # ------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(5)
    p_title.paragraph_format.line_spacing = 1.08
    rt = p_title.add_run(
        "SomnoScope: Clasificación Automática de Estadios del Sueño sobre Electroencefalografía "
        "Monocanal con Validación Cruzada por Sujeto y Arquitectura MLOps Productiva"
    )
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(13)
    rt.font.bold = True

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(2)
    ra = p_auth.add_run("Arturo Molina, Catalina García, Daniel Franco y Nicolás Gómez")
    ra.font.name = "Times New Roman"
    ra.font.size = Pt(10)
    ra.font.bold = True

    p_aff = doc.add_paragraph()
    p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_aff.paragraph_format.space_before = Pt(0)
    p_aff.paragraph_format.space_after = Pt(7)
    p_aff.paragraph_format.line_spacing = 1.0
    r_af = p_aff.add_run(
        "Maestría en Inteligencia Artificial (MAIA), Universidad de los Andes, Bogotá, Colombia\n"
        "[correo.molina@uniandes.edu.co; correo.garcia@uniandes.edu.co; "
        "correo.franco@uniandes.edu.co; correo.gomez@uniandes.edu.co]"
    )
    r_af.font.name = "Times New Roman"
    r_af.font.size = Pt(9)
    r_af.font.italic = True

    # ------------------------------------------------------------------
    # RESUMEN (200-300 PALABRAS EXACTAS)
    # ------------------------------------------------------------------
    resumen_text = (
        "La estadificación manual de estudios de polisomnografía (PSG) constituye el estándar clínico "
        "para diagnosticar trastornos del sueño, pero exige entre dos y cuatro horas de inspección visual "
        "por especialista y presenta variabilidad inter-evaluador cercana al 20%. Este trabajo presenta "
        "SomnoScope, una solución integral de aprendizaje automático y operaciones de aprendizaje automático "
        "(MLOps) que automatiza la clasificación de los cinco estadios del sueño definidos por la Academia "
        "Americana de Medicina del Sueño (AASM) —Vigilia (W), N1, N2, N3 y sueño de Movimientos Oculares "
        "Rápidos (REM)— empleando un único canal de electroencefalografía (EEG, Fpz-Cz). Se evaluaron "
        "cuatro modelos (Random Forest, LightGBM v1, LightGBM v2 y la red profunda TinySleepNet v2 basada "
        "en convoluciones unidimensionales y memoria bidireccional a corto y largo plazo, 1D-CNN+BiLSTM) "
        "sobre 23 sujetos de Sleep-EDF Expanded (45 noches, 48,378 épocas de 30 s) mediante validación "
        "cruzada estricta 3-Fold GroupKFold agrupada por paciente, garantizando cero solapamiento entre "
        "sujetos de entrenamiento y prueba. Antes de comparar modelos, se definió un criterio multiobjetivo "
        "basado en F1-Macro, desempeño en estadios críticos (N1 y REM) y latencia en CPU. TinySleepNet v2 "
        "con normalización Z-score por época y pesos inversos por clase obtuvo el mejor desempeño en los "
        "tres folds (exactitud 76.27 ± 5.21%, F1-Macro 0.7164 ± 0.0507, Kappa 0.6870 ± 0.0695, F1-N1 "
        "0.3779 y F1-REM 0.7638), procesando 1,000 épocas en 1.65 s en CPU. La solución se desplegó en "
        "AWS EC2 mediante Docker Compose (FastAPI, Streamlit y MLflow) para inferencia clínica interactiva."
    )
    wc = len(resumen_text.split())
    assert 200 <= wc <= 300, f"El Resumen tiene {wc} palabras (debe estar entre 200 y 300)."
    print(f"[Check] Conteo de palabras del Resumen: {wc} palabras (rango: 200-300).")

    add_body(doc, resumen_text, bold_prefix="Resumen— ", space_after=3.5)
    add_body(
        doc,
        "Clasificación de estadios del sueño; electroencefalografía (EEG); validación cruzada por sujeto; "
        "TinySleepNet; operaciones de aprendizaje automático (MLOps).",
        bold_prefix="Palabras clave— ",
        space_after=7.0,
        italic=True,
    )

    # ------------------------------------------------------------------
    # 1. INTRODUCCIÓN
    # ------------------------------------------------------------------
    add_h1(doc, "1. Introducción")
    add_body(
        doc,
        "Los trastornos del sueño —como la apnea obstructiva del sueño, el insomnio crónico y la narcolepsia— "
        "constituyen un problema prioritario de salud pública debido a su asociación directa con hipertensión, "
        "deterioro cognitivo y accidentes laborales [1]. El estándar de referencia para su diagnóstico es la "
        "polisomnografía (PSG), un examen nocturno multiseñal en el que un especialista clasifica manualmente "
        "cada segmento o época de 30 segundos en uno de cinco estadios definidos por la Academia Americana de "
        "Medicina del Sueño (AASM, por sus siglas en inglés): Vigilia (W), sueño ligero (N1), sueño intermedio "
        "(N2), sueño profundo de ondas lentas (N3) y sueño de Movimientos Oculares Rápidos (REM) [1]. En una "
        "noche típica de ocho horas, esta labor implica inspeccionar cerca de 1,000 épocas, demandando entre dos "
        "y cuatro horas de trabajo experto por paciente y presentando una concordancia inter-evaluador humana "
        "promedio cercana al 80%–83%, con discrepancias marcadas en la fase transicional N1 [2]."
    )
    add_body(
        doc,
        "Frente a este cuello de botella asistencial, las técnicas de inteligencia artificial (IA) permiten "
        "pre-etiquetar estudios completos en segundos. Sin embargo, persisten dos brechas críticas para su "
        "adopción clínica real: (i) una porción relevante de estudios académicos evalúa sus modelos mediante "
        "particiones aleatorias de épocas que mezclan registros del mismo paciente entre entrenamiento y prueba "
        "(fuga de datos o data leakage), sobrestimando artificialmente el desempeño; y (ii) la mayoría de los "
        "prototipos permanecen aislados en cuadernos de código sin trazabilidad de experimentos, APIs "
        "documentadas ni interfaces auditables para el personal clínico."
    )
    add_body(
        doc,
        "El objetivo general de este trabajo es diseñar, validar rigurosamente y desplegar en producción "
        "SomnoScope, una plataforma integral de estadificación automática del sueño sobre electroencefalografía "
        "(EEG) monocanal. Sus contribuciones principales son: (1) la evaluación comparativa libre de fuga "
        "intra-sujeto de cuatro arquitecturas clásicas y profundas sobre 23 sujetos de Sleep-EDF Expanded "
        "(48,378 épocas) bajo validación cruzada 3-Fold GroupKFold; (2) el diagnóstico y corrección de un "
        "colapso numérico en capas de normalización por lotes (Batch Normalization) sobre voltajes crudos de "
        "EEG mediante normalización Z-score por época; (3) la cuantificación empírica del escalamiento de datos "
        "(9 vs. 23 sujetos) sobre pacientes estrictamente inéditos; y (4) un despliegue productivo reproducible "
        "en la nube (AWS EC2) bajo prácticas de operaciones de aprendizaje automático (MLOps) con FastAPI, "
        "Streamlit y MLflow."
    )

    # ------------------------------------------------------------------
    # 2. REVISIÓN DEL ESTADO DEL ARTE
    # ------------------------------------------------------------------
    add_h1(doc, "2. Revisión del estado del arte")
    add_body(
        doc,
        "La automatización de la estadificación del sueño ha evolucionado desde modelos tabulares clásicos "
        "basados en ingeniería de características hasta arquitecturas de aprendizaje profundo (Deep Learning) "
        "de extremo a extremo. Los enfoques tradicionales extraen descriptores de potencia espectral mediante "
        "el periodograma de Welch —cuantificando bandas clínicas delta (0.5–4 Hz), theta (4–8 Hz), alfa "
        "(8–12 Hz), sigma o husos de sueño (12–16 Hz) y beta (16–30 Hz)— junto con parámetros temporales de "
        "Hjorth, alimentando clasificadores de ensamble como Random Forest [3] o Gradient Boosting (LightGBM) "
        "[4]. Estos modelos ofrecen alta interpretabilidad y bajo costo computacional, pero su representación "
        "estática por época limita la captura de eventos transitorios breves como complejos K o microdespertares."
    )
    add_body(
        doc,
        "Con la adopción de redes neuronales profundas, Supratak et al. [5] introdujeron DeepSleepNet y "
        "posteriormente TinySleepNet [6], arquitecturas híbridas que combinan redes neuronales convolucionales "
        "unidimensionales (1D-CNN) para extraer patrones morfológicos directamente de la señal cruda con capas "
        "recurrentes de memoria bidireccional a corto y largo plazo (BiLSTM) para modelar la dinámica temporal, "
        "alcanzando exactitudes de 82%–85% sobre EEG monocanal en Sleep-EDF mediante validación cruzada de "
        "20 folds con contextos multiepoca (secuencias de 15 a 25 épocas consecutivas). En paralelo, Phan et al. "
        "[7] propusieron SeqSleepNet con mecanismos de atención jerárquica sobre espectrogramas tiempo-frecuencia, "
        "Perslev et al. [8] desarrollaron U-Sleep (una red completamente convolucional entrenada sobre múltiples "
        "cohortes internacionales que prescinde de recurrencia), y Eldele et al. [9] integraron atención "
        "multi-cabezal y convoluciones multiescala en AttnSleep."
    )
    add_body(
        doc,
        "Análisis crítico de brechas identificadas: La revisión bibliográfica evidencia tres vacíos que guían "
        "el diseño de SomnoScope. Primero, los modelos de máximo desempeño reportado suelen depender de "
        "infraestructura con GPU, ventanas secuenciales largas o grandes volúmenes multicéntricos, lo que "
        "dificulta su despliegue económico en entornos hospitalarios con recursos de CPU limitados. Segundo, "
        "el estadio N1 continúa siendo el talón de Aquiles universal en EEG monocanal (F1 típicamente entre "
        "0.30 y 0.50 incluso en arquitecturas avanzadas [6], [9]), debido tanto a su baja prevalencia como a "
        "su similitud electrofisiológica con Vigilia y REM en ausencia de electrooculografía (EOG) y "
        "electromiografía (EMG). Tercero, la literatura rara vez integra la validación estadística por sujeto "
        "con una arquitectura MLOps completa orientada a mitigar deuda técnica en producción [10] —registro "
        "de modelos, comparación Champion/Challenger en vivo, API REST y auditoría visual de hipnogramas—."
    )

    # ------------------------------------------------------------------
    # 3. METODOLOGÍA
    # ------------------------------------------------------------------
    add_h1(doc, "3. Metodología")
    add_body(
        doc,
        "La metodología sigue el ciclo de vida completo de aprendizaje automático: adquisición y curación "
        "clínica de datos, preprocesamiento de señales, partición libre de fuga por paciente, modelado y "
        "manejo del desbalance de clases, evaluación cruzada y despliegue MLOps (Figura 1)."
    )

    add_h2(doc, "3.1 Datos, consistencia de la cohorte y distribución de estadios")
    add_body(
        doc,
        "Se utilizó el repositorio público de acceso abierto Sleep-EDF Database Expanded de PhysioNet [11], "
        "compuesto por dos subestudios: Sleep Cassette (SC, adultos sanos monitorizados en domicilio durante "
        "dos noches consecutivas) y Sleep Telemetry (ST, pacientes con dificultad leve para conciliar el sueño "
        "registrados en laboratorio). Con el fin de asegurar absoluta consistencia entre los datos descritos y "
        "los realmente evaluados, se diferencian explícitamente dos cohortes utilizadas en el proyecto:"
    )
    add_bullet(
        doc,
        "Cohorte principal de evaluación (23 sujetos, 45 noches, 48,378 épocas): ",
        "Integra los 20 sujetos completos del estándar de referencia Sleep-EDF-20 (SC400–SC419; 39 noches, ya "
        "que SC413 dispone de una sola noche en PhysioNet) más 3 sujetos de Sleep Telemetry (ST701, ST702 y "
        "ST704; 6 noches). Sobre este conjunto exacto de 48,378 épocas de 30 s se ejecutó la validación cruzada "
        "3-Fold GroupKFold y se entrenaron los artefactos finales promovidos a producción."
    )
    add_bullet(
        doc,
        "Cohorte base de desarrollo inicial (9 sujetos, 18 noches, 18,919 épocas): ",
        "Subconjunto versionado originalmente con Data Version Control (DVC) en Amazon S3 (SC400–SC405 y "
        "ST701, ST702, ST704) para viabilizar iteraciones rápidas. Este subconjunto se emplea únicamente en el "
        "experimento auxiliar de escalamiento (Sección 4.4) para cuantificar el impacto de ampliar la muestra "
        "de 9 a 23 sujetos sobre 7 pacientes estrictamente inéditos."
    )
    add_body(
        doc,
        "La Tabla 1 resume la composición exacta de las 48,378 épocas evaluadas y su distribución por fold. "
        "Destaca un desbalance severo de clases: N2 concentra el 43.07% de las épocas, mientras que el estadio "
        "transicional N1 representa apenas el 6.64% (una razón aproximada de 6.5:1 frente a N2)."
    )

    # Tabla 1: Distribución real exacta de clases y folds
    t1_headers = ["Dimensión / Estadio", "Subgrupo / Descripción", "Épocas (n)", "Proporción (%)", "Sujetos / Observación"]
    t1_rows = [
        ["Vigilia (W)", "Despierto (recortado ±30 min)", f"{cnt['W']:,}", f"{pct['W']:.2f}%", "Transición inicial/despertares"],
        ["Estadio N1", "Sueño ligero transicional", f"{cnt['N1']:,}", f"{pct['N1']:.2f}%", "Clase minoritaria crítica"],
        ["Estadio N2", "Sueño intermedio (husos/K)", f"{cnt['N2']:,}", f"{pct['N2']:.2f}%", "Clase mayoritaria"],
        ["Estadio N3", "Sueño profundo (ondas lentas)", f"{cnt['N3']:,}", f"{pct['N3']:.2f}%", "Fusión R&K S3 + S4 (AASM)"],
        ["Estadio REM", "Movimientos oculares rápidos", f"{cnt['REM']:,}", f"{pct['REM']:.2f}%", "Alta similitud espectral con N1"],
        ["Fold 1 (Prueba OOF)", "SC403, SC406, SC409, SC412, SC416, SC419, ST701, ST704", "16,922", "34.98%", "8 sujetos (Train: 15 subj., 31,456 ép.)"],
        ["Fold 2 (Prueba OOF)", "SC401, SC404, SC407, SC410, SC414, SC415, SC418, ST702", "16,235", "33.56%", "8 sujetos (Train: 15 subj., 32,143 ép.)"],
        ["Fold 3 (Prueba OOF)", "SC400, SC402, SC405, SC408, SC411, SC413, SC417", "15,221", "31.46%", "7 sujetos (Train: 16 subj., 33,157 ép.)"],
        ["Total Evaluado (OOF)", "20 Sleep Cassette + 3 Sleep Telemetry (45 noches)", "48,378", "100.00%", "23/23 sujetos únicos (0% solapamiento)"],
    ]
    add_academic_table(
        doc,
        "Tabla 1. Distribución exacta de estadios AASM y particiones del esquema 3-Fold GroupKFold sobre los 23 sujetos evaluados.",
        t1_headers, t1_rows, col_widths=[1.25, 2.05, 0.72, 0.78, 1.73], highlight_row=8
    )

    add_h2(doc, "3.2 Preprocesamiento clínico y manejo del desbalance de clases")
    add_body(
        doc,
        "Cada archivo en formato European Data Format (.edf) se procesó en Python mediante MNE-Python [12] "
        "extrayendo exclusivamente el canal estándar EEG Fpz-Cz muestreado a 100 Hz (3,000 muestras por época "
        "de 30 s). Se aplicó un filtro digital pasa-banda de respuesta al impulso finita (FIR) entre 0.5 y 35 Hz "
        "para eliminar derivas de línea base y ruido de alta frecuencia. Las etiquetas originales en estándar "
        "Rechtschaffen y Kales (R&K) se unificaron al estándar AASM [1] fusionando S3 y S4 en N3, descartando "
        "épocas anotadas como Movement time o desconocidas (?), y recortando las largas colas de vigilia diurna "
        "a ±30 minutos alrededor del período principal de sueño, siguiendo la convención establecida en [5], [6]."
    )
    add_body(
        doc,
        "Durante la experimentación inicial con Deep Learning se detectó un hallazgo técnico determinante: "
        "MNE-Python carga las amplitudes de EEG en Voltios (~10⁻⁵ V), por lo que su varianza (~10⁻¹⁰ V²) queda "
        "cinco órdenes de magnitud por debajo del parámetro de estabilidad numérica (ε = 10⁻⁵) de las capas "
        "BatchNorm1d en PyTorch [13], anulando los gradientes y colapsando las predicciones hacia una sola clase. "
        "Para resolverlo sin introducir fuga estadística entre muestras, se incorporó normalización Z-score "
        "independiente por cada época i y muestra temporal t, como se muestra en la Ecuación (1):"
    )
    add_equation(
        doc,
        [
            ("x̃", True, False, False), ("i,t", True, True, False),
            ("  =  ( ", False, False, False),
            ("x", True, False, False), ("i,t", True, True, False),
            (" − ", False, False, False),
            ("μ", True, False, False), ("i", True, True, False),
            (" ) / ( ", False, False, False),
            ("σ", True, False, False), ("i", True, True, False),
            (" + ", False, False, False),
            ("ε", True, False, False), (" ),   con  ", False, False, False),
            ("ε", True, False, False), (" = 10", False, False, False), ("−12", False, False, True),
        ],
        "1",
    )
    add_body(
        doc,
        "donde μᵢ y σᵢ son la media y desviación estándar calculadas exclusivamente sobre las 3,000 muestras de "
        "la época i. Respecto al manejo del desbalance de clases (Tabla 1), se aplicó explícitamente ponderación "
        "inversa por frecuencia de clase en los cuatro modelos evaluados, calculada estrictamente sobre el "
        "subconjunto de entrenamiento de cada fold según la Ecuación (2):"
    )
    add_equation(
        doc,
        [
            ("w", True, False, False), ("c", True, True, False),
            ("  =  ", False, False, False),
            ("N", True, False, False), ("train", False, True, False),
            (" / ( ", False, False, False),
            ("K", True, False, False), (" · ", False, False, False),
            ("n", True, False, False), ("c", True, True, False),
            (" ),   ", False, False, False),
            ("c", True, False, False), (" ∈ {W, N1, N2, N3, REM},  ", False, False, False),
            ("K", True, False, False), (" = 5", False, False, False),
        ],
        "2",
    )
    add_body(
        doc,
        "donde N_train es el número total de épocas de entrenamiento del fold y n_c el conteo de la clase c en "
        "dicho entrenamiento. En Random Forest y LightGBM (v1 y v2) esto se implementó mediante el parámetro "
        "class_weight='balanced', mientras que en TinySleepNet v2 se incorporó en la función de pérdida "
        "nn.CrossEntropyLoss(weight=class_weights). Dado que N1 representa el 6.64% de las épocas, recibe un "
        "peso aproximado ~3.0× superior al promedio y ~6.5× superior a N2, penalizando proporcionalmente sus "
        "errores de omisión sin replicar muestras sintéticas ni filtrar información del bloque de prueba."
    )

    add_h2(doc, "3.3 Uso de la partición de validación/prueba y prevención de fuga de información")
    add_body(
        doc,
        "En clasificación de PSG, dividir épocas aleatoriamente provoca fuga de datos severa porque el modelo "
        "memoriza la firma espectral basal de cada individuo. Para garantizar generalización clínica real hacia "
        "pacientes nuevos, toda partición se agrupó estrictamente por identificador de sujeto (subject_id), "
        "asegurando que las dos noches de un mismo individuo pertenezcan siempre al mismo bloque. En la etapa "
        "inicial de desarrollo se empleó un esquema hold-out GroupShuffleSplit (70% entrenamiento / 30% "
        "validación-prueba, ~67%/33% efectivo por indivisibilidad de sujetos); sin embargo, depender de una "
        "única partición del ~30% conlleva dos limitaciones: alta varianza según qué pacientes caigan en prueba "
        "y acoplamiento entre la selección de modelo y la estimación final de desempeño."
    )
    add_body(
        doc,
        "Para superar esa limitación, el protocolo definitivo adopta validación cruzada 3-Fold GroupKFold sobre "
        "los 23 sujetos (Tabla 1), donde en cada iteración aproximadamente el 67% de los sujetos (15–16 "
        "pacientes) conforma el conjunto de entrenamiento y el ~33% restante (7–8 pacientes completamente inéditos) "
        "actúa como conjunto de evaluación fuera de muestra (Out-Of-Fold, OOF). Al completar los 3 folds, el 100% "
        "de los 23 sujetos (48,378 épocas) ha sido evaluado sin haber participado en el entrenamiento del modelo "
        "que lo predice. La prevención de fuga se sustenta en tres controles: (1) aislamiento total por subject_id "
        "(0% de intersección entre entrenamiento y evaluación); (2) normalización Z-score por época (Ecuación (1)) "
        "y extracción espectral independiente por ventana de 30 s, sin estimadores ajustados sobre el corpus "
        "global; y (3) cálculo de pesos de clase (Ecuación (2)) restringido únicamente a y_train de cada fold."
    )
    add_body(
        doc,
        "Con transparencia metodológica, se deja constancia de un matiz relevante respecto al uso del ~33% de "
        "evaluación en cada fold: mientras los modelos clásicos (Random Forest v1, LightGBM v1 y LightGBM v2) "
        "fueron entrenados con hiperparámetros fijos predefinidos y evaluados una sola vez sobre el fold externo "
        "sin utilizarlo durante el ajuste, en TinySleepNet v2 dicho fold externo se empleó simultáneamente para "
        "seleccionar el mejor punto de guardado (checkpointing por máximo F1-Macro a lo largo de 8 épocas fijas) "
        "y reportar las métricas del fold. Aunque los hiperparámetros arquitectónicos (tasa de aprendizaje, tamaño "
        "de lote y arquitectura) permanecieron fijos y nunca se modificaron en función del fold, reutilizar el "
        "mismo bloque para seleccionar la época óptima de checkpoint puede introducir un sesgo optimista leve; "
        "por ello, en la Sección 6 se propone adoptar validación cruzada anidada (nested GroupKFold) con "
        "partición triple estricta (entrenamiento/validación interna/prueba externa)."
    )

    add_h2(doc, "3.4 Modelos evaluados, herramientas y despliegue MLOps en AWS")
    add_body(
        doc,
        "Se compararon cuatro modelos pertenecientes a dos paradigmas complementarios sobre exactamente los "
        "mismos 3 folds: (i) rama clásica tabular, alimentada con 25 características fisiológicas extraídas por "
        "época (potencias absolutas y relativas en bandas delta, theta, alfa, sigma y beta vía método de Welch, "
        "razones espectrales delta/beta, theta/alfa y delta/alfa, entropía espectral, parámetros de Hjorth "
        "—actividad, movilidad y complejidad— y estadísticos temporales), entrenando Random Forest v1 (100 "
        "árboles, profundidad máxima 12) [3], LightGBM v1 (150 estimadores, tasa de aprendizaje 0.05) [4] y "
        "LightGBM v2 regularizado (300 estimadores, tasa 0.03, profundidad 7, submuestreo 0.85, regularización "
        "L1=0.1 y L2=1.5); y (ii) rama de aprendizaje profundo, implementando TinySleepNet1D v2 en PyTorch [13] "
        "directamente sobre la señal cruda normalizada (1×3,000 muestras), combinando tres bloques convolucionales "
        "1D con núcleos multiescala (64, 128 y 128 filtros), BatchNorm1d, ReLU, MaxPooling y Dropout (0.5), "
        "seguidos de una capa BiLSTM (64 unidades ocultas por dirección, 128 dimensiones concatenadas) y una capa "
        "lineal de clasificación entrenada con AdamW (lr=10⁻³, weight_decay=10⁻⁴, lote 64, 8 épocas)."
    )
    add_body(
        doc,
        "Toda la solución se construyó en Python 3.11+ y se desplegó en una instancia AWS EC2 t2.medium "
        "(2 vCPU, 4 GB RAM, 2 GB Swap, Ubuntu 22.04 LTS, Elastic IP fija 52.207.88.72) orquestada con Docker "
        "Compose en tres microservicios acoplados (Figura 1): (1) MLflow (:5000) [14], que gestiona corridas "
        "anidadas de validación cruzada (Parent Runs con Media ± DE y 3 Child Folds) y el registro formal de "
        "versiones mediante alias @champion/@production (TinySleepNet v2) y @challenger (LightGBM v2); "
        "(2) FastAPI v3.0 (:8000) [15], que expone endpoints documentados bajo OpenAPI (/health, /predict/epoch "
        "y /predict/recording servidos por el modelo Champion, y /predict/features por el Challenger) validados "
        "con 8 pruebas automatizadas en Pytest; y (3) Streamlit v3.0 (:8050), un tablero clínico interactivo "
        "con carga de archivos .edf reales, selector en vivo Champion vs. Challenger, hipnograma nocturno en "
        "Plotly, indicadores clínicos (eficiencia del sueño, latencia, distribución de estadios) y recorrido "
        "guiado de seis pasos."
    )

    add_figure_block(
        doc,
        fig1_path,
        "Figura 1.",
        "Arquitectura integral de SomnoScope siguiendo el ciclo de vida MLOps: ingesta y preprocesamiento clínico "
        "(MNE-Python), validación cruzada 3-Fold GroupKFold, modelado paralelo (Deep Learning vs. ML clásico), "
        "gobernanza en MLflow y despliegue contenedorizado en AWS EC2. Elaboración propia.",
        width_in=6.25,
    )

    # ------------------------------------------------------------------
    # 4. RESULTADOS Y EVALUACIÓN
    # ------------------------------------------------------------------
    add_h1(doc, "4. Resultados y evaluación")

    add_h2(doc, "4.1 Criterios de evaluación y selección del modelo campeón")
    add_body(
        doc,
        "Previo a ejecutar la comparación cuantitativa entre modelos, se definió un criterio formal multiobjetivo "
        "de selección del modelo campeón alineado con la realidad clínica y operacional del servicio:"
    )
    add_bullet(
        doc,
        "Métrica principal (F1-Macro en 3-Fold GroupKFold): ",
        "Se priorizó F1-Macro sobre la exactitud global (Accuracy) porque las cinco clases presentan un desbalance "
        "de hasta 6.5:1 (Tabla 1); un modelo que ignore N1 puede mantener alta exactitud siendo clínicamente "
        "inaceptable. Como medida secundaria de concordancia inter-evaluador corregida por azar respecto al "
        "estándar AASM se evaluó el índice Kappa de Cohen (κ), ambos definidos en la Ecuación (3):"
    )
    add_equation(
        doc,
        [
            ("F1-Macro = (1/", False, False, False), ("K", True, False, False),
            (") Σ", False, False, False), ("c=1..K", False, True, False),
            (" [ 2·TP", False, False, False), ("c", True, True, False),
            (" / (2·TP", False, False, False), ("c", True, True, False),
            (" + FP", False, False, False), ("c", True, True, False),
            (" + FN", False, False, False), ("c", True, True, False),
            (") ],     ", False, False, False),
            ("κ", True, False, False), (" = ( ", False, False, False),
            ("p", True, False, False), ("o", True, True, False),
            (" − ", False, False, False),
            ("p", True, False, False), ("e", True, True, False),
            (" ) / ( 1 − ", False, False, False),
            ("p", True, False, False), ("e", True, True, False), (" )", False, False, False),
        ],
        "3",
    )
    add_bullet(
        doc,
        "Desempeño por estadio clínico crítico: ",
        "Ante desempeños globales cercanos, se prioriza el modelo con mejor balance en los estadios de mayor "
        "dificultad diagnóstica y relevancia clínica —particularmente N1 (sueño transicional) y REM (arquitectura "
        "del sueño y parasomnias)— sin degradar severamente N2, N3 ni Vigilia, y demostrando consistencia en "
        "la mayoría de los folds."
    )
    add_bullet(
        doc,
        "Latencia de inferencia y huella operacional en CPU: ",
        "Dado que la plataforma se despliega en instancias CPU económicas (AWS EC2 t2.medium sin GPU), el modelo "
        "campeón debe ofrecer latencia de extremo a extremo inferior a 100 ms por época individual (para la API "
        "/predict/epoch) y procesar una noche completa de 1,000 épocas en menos de 5 segundos, con un tamaño de "
        "artefacto compacto que evite saturación de memoria RAM."
    )

    add_h2(doc, "4.2 Comparación en 3-Fold GroupKFold y verificación de coherencia del campeón")
    add_body(
        doc,
        "La Tabla 2 presenta el consolidado oficial (Media ± Desviación Estándar sobre los 3 folds, cubriendo los "
        "23 sujetos inéditos) junto con las latencias reales de inferencia de extremo a extremo medidas en CPU "
        "(incluyendo el tiempo de extracción de las 25 características espectrales de Welch —5.46 ms/época y "
        "3.93 s por 1,000 épocas— en los modelos clásicos, frente a la normalización Z-score directa en "
        "TinySleepNet v2). La Figura 2A ilustra visualmente el comportamiento comparativo con barras de dispersión."
    )

    # Tabla 2: Resultados consolidados CV + Latencia real
    def fmt_ms(m_key, k, pct_scale=False):
        mu = s[m_key][f"{k}_mean"]
        sd = s[m_key][f"{k}_std"]
        return f"{mu*100:.2f}±{sd*100:.2f}%" if pct_scale else f"{mu:.4f}±{sd:.4f}"

    t2_headers = [
        "Modelo (3-Fold CV)", "Accuracy", "F1-Macro", "Kappa (κ)",
        "F1-N1", "F1-REM", "F1-N3", "Latencia 1 ép. / 1,000 ép.", "Peso"
    ]
    t2_rows = [
        [
            "Random Forest v1",
            fmt_ms("Random Forest v1", "accuracy", True),
            fmt_ms("Random Forest v1", "f1_macro"),
            fmt_ms("Random Forest v1", "cohen_kappa"),
            f"{s['Random Forest v1']['f1_N1_mean']:.4f}",
            f"{s['Random Forest v1']['f1_REM_mean']:.4f}",
            f"{s['Random Forest v1']['f1_N3_mean']:.4f}",
            f"{lat['rf_e2e_1ep_ms']:.1f} ms / {lat['rf_e2e_1000ep_s']:.2f} s",
            "23.51 MB"
        ],
        [
            "LightGBM v1 (Base)",
            fmt_ms("LightGBM v1", "accuracy", True),
            fmt_ms("LightGBM v1", "f1_macro"),
            fmt_ms("LightGBM v1", "cohen_kappa"),
            f"{s['LightGBM v1']['f1_N1_mean']:.4f}",
            f"{s['LightGBM v1']['f1_REM_mean']:.4f}",
            f"{s['LightGBM v1']['f1_N3_mean']:.4f}",
            f"{lat['lgb1_e2e_1ep_ms']:.1f} ms / {lat['lgb1_e2e_1000ep_s']:.2f} s",
            "~4.9 MB"
        ],
        [
            "LightGBM v2 (Tuned)",
            fmt_ms("LightGBM v2", "accuracy", True),
            fmt_ms("LightGBM v2", "f1_macro"),
            fmt_ms("LightGBM v2", "cohen_kappa"),
            f"{s['LightGBM v2']['f1_N1_mean']:.4f}",
            f"{s['LightGBM v2']['f1_REM_mean']:.4f}",
            f"{s['LightGBM v2']['f1_N3_mean']:.4f}",
            f"{lat['lgb2_e2e_1ep_ms']:.1f} ms / {lat['lgb2_e2e_1000ep_s']:.2f} s",
            "4.94 MB"
        ],
        [
            "TinySleepNet v2 (Champion)",
            fmt_ms("TinySleepNet v2 (Normalizado)", "accuracy", True),
            fmt_ms("TinySleepNet v2 (Normalizado)", "f1_macro"),
            fmt_ms("TinySleepNet v2 (Normalizado)", "cohen_kappa"),
            f"{s['TinySleepNet v2 (Normalizado)']['f1_N1_mean']:.4f}",
            f"{s['TinySleepNet v2 (Normalizado)']['f1_REM_mean']:.4f}",
            f"{s['TinySleepNet v2 (Normalizado)']['f1_N3_mean']:.4f}",
            f"{lat['tsn_e2e_1ep_ms']:.1f} ms / {lat['tsn_e2e_1000ep_s']:.2f} s",
            "0.62 MB"
        ],
    ]
    add_academic_table(
        doc,
        "Tabla 2. Resultados consolidados de 3-Fold GroupKFold (23 sujetos, 48,378 épocas, Media ± DE) y latencia real de inferencia en CPU.",
        t2_headers, t2_rows, col_widths=[1.32, 0.78, 0.80, 0.78, 0.48, 0.50, 0.48, 0.94, 0.45], highlight_row=3
    )

    add_body(
        doc,
        "Confirmación de coherencia con el criterio de selección: TinySleepNet v2 (Normalizado) satisface de forma "
        "plenamente coherente las tres dimensiones del criterio definido en la Sección 4.1. En la métrica principal, "
        "lidera con F1-Macro de 0.7164 ± 0.0507 (+2.22 pp sobre LightGBM v1, +2.40 pp sobre LightGBM v2 y +3.52 pp "
        "sobre Random Forest v1) y alcanza el mayor índice Kappa (0.6870 ± 0.0695, concordancia sustancial). En "
        "estadios críticos, supera ampliamente a todos los modelos clásicos en N1 (0.3779 vs. 0.3233 de LightGBM v1, "
        "+5.46 pp), REM (0.7638 vs. 0.6868, +7.70 pp) y Vigilia (0.8460 vs. 0.7891, +5.69 pp), mientras los modelos "
        "de árboles conservan una ventaja moderada en N2 (0.8481 vs. 0.8083) y N3 (0.8239 vs. 0.7858). En el eje "
        "operacional, TinySleepNet v2 resulta además el más veloz de extremo a extremo en CPU: procesa una época "
        "en 9.47 ms y una noche completa de 1,000 épocas en 1.65 s (2.4× más rápido que LightGBM, cuyo cómputo "
        "espectral de Welch consume 3.93 s por noche), ocupando apenas 0.62 MB en disco frente a 4.94 MB de "
        "LightGBM y 23.51 MB de Random Forest. Cabe aclarar que las latencias fueron medidas en entorno local CPU "
        "sin concurrencia multiusuario; el comportamiento bajo carga clínica concurrente en producción se proyecta "
        "evaluar posteriormente ([latencia p95 en producción hospitalaria concurrente pendiente de medición])."
    )

    # Tabla 3: Desglose por fold del Champion
    folds = cv["folds"]
    oof_t = cv["oof_aggregate_tinysleepnet_v2"]
    t3_headers = ["Fold (Sujetos inéditos)", "Épocas", "Mejor Ép.", "Accuracy", "F1-Macro", "Kappa", "F1-W", "F1-N1", "F1-N2", "F1-N3", "F1-REM"]
    t3_rows = []
    for f in folds:
        mt = f["metrics_by_model"]["TinySleepNet v2 (Normalizado)"]
        t3_rows.append([
            f"Fold {f['fold']} ({len(f['val_subjects'])} subj.)",
            f"{f['val_epochs']:,}",
            str(int(mt["best_epoch"])),
            f"{mt['accuracy']*100:.2f}%",
            f"{mt['f1_macro']:.4f}",
            f"{mt['cohen_kappa']:.4f}",
            f"{mt['f1_W']:.4f}",
            f"{mt['f1_N1']:.4f}",
            f"{mt['f1_N2']:.4f}",
            f"{mt['f1_N3']:.4f}",
            f"{mt['f1_REM']:.4f}",
        ])
    t3_rows.append([
        "Media ± DE (3 Folds)", "16,126", "7.7",
        f"{s['TinySleepNet v2 (Normalizado)']['accuracy_mean']*100:.2f}±{s['TinySleepNet v2 (Normalizado)']['accuracy_std']*100:.2f}%",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_macro_mean']:.4f}±{s['TinySleepNet v2 (Normalizado)']['f1_macro_std']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['cohen_kappa_mean']:.4f}±{s['TinySleepNet v2 (Normalizado)']['cohen_kappa_std']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_W_mean']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_N1_mean']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_N2_mean']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_N3_mean']:.4f}",
        f"{s['TinySleepNet v2 (Normalizado)']['f1_REM_mean']:.4f}",
    ])
    t3_rows.append([
        "OOF Agregado (23/23)", "48,378", "—",
        f"{oof_t['accuracy']*100:.2f}%",
        f"{oof_t['f1_macro']:.4f}",
        f"{oof_t['cohen_kappa']:.4f}",
        f"{oof_t['f1_W']:.4f}",
        f"{oof_t['f1_N1']:.4f}",
        f"{oof_t['f1_N2']:.4f}",
        f"{oof_t['f1_N3']:.4f}",
        f"{oof_t['f1_REM']:.4f}",
    ])
    add_academic_table(
        doc,
        "Tabla 3. Desempeño detallado de TinySleepNet v2 (Champion) por fold y agregado Out-Of-Fold sobre 48,378 épocas.",
        t3_headers, t3_rows, col_widths=[1.22, 0.50, 0.44, 0.72, 0.76, 0.76, 0.42, 0.42, 0.42, 0.42, 0.45], highlight_row=4
    )
    add_body(
        doc,
        "Como muestra la Tabla 3, TinySleepNet v2 ocupó el primer lugar en F1-Macro y F1-N1 en los tres folds "
        "(3/3 victorias). La dispersión observada entre Fold 1 (72.44%) y Fold 3 (82.21%) responde directamente "
        "a la heterogeneidad inter-sujeto: Fold 1 y Fold 2 contienen los sujetos de Sleep Telemetry (ST701, ST702, "
        "ST704), cuyas grabaciones hospitalarias con dificultad leve de conciliación presentan mayor fragmentación "
        "del sueño que los adultos sanos de Sleep Cassette agrupados mayoritariamente en Fold 3."
    )

    add_figure_block(
        doc,
        fig2_path,
        "Figura 2.",
        "Panel A: Desempeño comparativo en 3-Fold GroupKFold (Media ± DE sobre 23 sujetos). Panel B: Impacto "
        "del escalamiento de datos (9 vs. 23 sujetos) evaluado sobre los mismos 7 pacientes estrictamente "
        "inéditos (15,247 épocas). Elaboración propia.",
        width_in=6.35,
    )

    add_h2(doc, "4.3 Comparación de matrices de confusión de todos los modelos")
    add_body(
        doc,
        "La Figura 3 reúne en un único panel compacto las matrices de confusión agregadas Out-Of-Fold (48,378 "
        "épocas, 23 sujetos) de los cuatro modelos evaluados, normalizadas por fila (sensibilidad por estadio) y "
        "trazadas exactamente bajo la misma escala [0.00, 1.00] para facilitar la comparación directa."
    )
    add_figure_block(
        doc,
        fig3_path,
        "Figura 3.",
        "Matrices de confusión agregadas Out-Of-Fold (3-Fold GroupKFold, 48,378 épocas, 23 sujetos) para los "
        "cuatro modelos evaluados, normalizadas por fila y bajo escala común [0.00, 1.00]. Elaboración propia.",
        width_in=5.85,
    )
    add_body(
        doc,
        "Al comparar los patrones de error más frecuentes entre modelos en la Figura 3 destacan tres hallazgos "
        "clínicamente consistentes: (1) Estadio N1 (foco principal de error): en los tres clasificadores clásicos "
        "la sensibilidad en N1 es reducida (0.29 en Random Forest v1, 0.39 en LightGBM v1 y 0.38 en LightGBM v2), "
        "desviando masivamente las épocas de N1 hacia REM (0.40 en RF, 0.34 en LightGBM v1 y 0.35 en LightGBM v2) "
        "y hacia Vigilia (0.16–0.19). Esto ocurre porque N1 y REM comparten actividad EEG mixta de baja amplitud "
        "en bandas theta/alfa que resulta casi indistinguible en espectros estáticos de Welch sin canales EOG/EMG. "
        "TinySleepNet v2 eleva la sensibilidad de N1 hasta 0.53 (+14 pp sobre LightGBM v1 y +24 pp sobre RF v1), "
        "reduciendo la confusión N1→REM a 0.27 y N1→W a 0.10 gracias a sus filtros convolucionales y la memoria "
        "secuencial BiLSTM intra-época. (2) Estadio REM: los modelos basados en árboles alcanzan sensibilidades "
        "de 0.72 (LightGBM v1/v2) a 0.75 (Random Forest v1) en REM, confundiendo entre 12% y 16% con N1 y 6%–7% "
        "con N2; TinySleepNet v2 incrementa la sensibilidad en REM a 0.81 (+9 pp frente a LightGBM), minimizando "
        "los errores REM→N2 a 0.03 y REM→W a 0.02. (3) Compensación en N2 y N3: TinySleepNet v2 maximiza el "
        "acierto en sueño profundo N3 (0.92 frente a 0.84–0.88 en clásicos), aunque tiende a sobre-predecir N3 "
        "sobre épocas fronterizas de N2 (confusión N2→N3 de 0.13 vs. 0.06–0.07 en árboles), lo que explica su "
        "menor sensibilidad en N2 (0.71 frente a 0.79–0.81 en los modelos clásicos)."
    )

    add_h2(doc, "4.4 Experimento auxiliar de escalamiento: 9 sujetos vs. 23 sujetos")
    add_body(
        doc,
        "Para aislar el efecto del volumen y la diversidad de pacientes sobre la generalización del modelo "
        "campeón, se ejecutó una comparación controlada cabeza a cabeza (Tabla 4 y Figura 2B) entre TinySleepNet "
        "v2 entrenado con la cohorte base inicial (9 sujetos, 18 noches) y el entrenado con la cohorte ampliada "
        "(23 sujetos, 45 noches), evaluando ambos exactamente sobre los mismos 7 sujetos estrictamente inéditos "
        "(SC401, SC408, SC409, SC412, SC413, SC415 y SC417; 15,247 épocas) excluidos del entrenamiento de ambos."
    )
    t4_headers = ["Configuración TinySleepNet v2", "Accuracy", "F1-Macro", "Kappa (κ)", "F1-W", "F1-N1", "F1-N2", "F1-N3", "F1-REM"]
    t4_rows = [
        ["Entrenado en 9 sujetos (18 noches)", "72.00%", "0.6735", "0.6409", "0.8490", "0.3363", "0.7245", "0.7466", "0.7110"],
        ["Entrenado en 23 sujetos (45 noches)", "76.24%", "0.7138", "0.6893", "0.8067", "0.3184", "0.8343", "0.8656", "0.7440"],
        ["Ganancia neta por escalamiento (Δ)", "+4.24 pp", "+4.03 pp", "+0.0484", "−4.23 pp", "−1.79 pp", "+10.98 pp", "+11.90 pp", "+3.30 pp"],
    ]
    add_academic_table(
        doc,
        "Tabla 4. Comparación controlada Head-to-Head sobre los mismos 7 sujetos estrictamente inéditos (15,247 épocas).",
        t4_headers, t4_rows, col_widths=[1.88, 0.60, 0.60, 0.60, 0.56, 0.56, 0.58, 0.58, 0.57], highlight_row=2
    )
    add_body(
        doc,
        "Escalar de 9 a 23 sujetos produjo mejoras sustanciales y consistentes en la generalización global "
        "(+4.24 pp en Accuracy, +4.03 pp en F1-Macro y +0.0484 en Kappa), con saltos notables en N3 (+11.90 pp), "
        "N2 (+10.98 pp) y REM (+3.30 pp). En N1 se registró una leve caída (−1.79 pp) en este subconjunto "
        "particular de 7 pacientes, lo cual confirma la necesidad de estrategias complementarias de desbalance "
        "y contexto inter-época para dicho estadio."
    )

    add_h2(doc, "4.5 Comparación con la literatura y análisis crítico")
    add_body(
        doc,
        "La Tabla 5 contextualiza los resultados de SomnoScope frente a trabajos de referencia en Sleep-EDF."
    )
    t5_headers = ["Trabajo / Modelo", "Protocolo y Cohorte", "Contexto", "Accuracy", "F1-Macro", "Kappa", "Despliegue MLOps"]
    t5_rows = [
        ["DeepSleepNet [5]", "20-Fold CV (Sleep-EDF-20)", "Secuencia (25 ép.)", "82.0%", "0.769", "0.76", "No reportado"],
        ["TinySleepNet [6]", "20-Fold CV (Sleep-EDF-20)", "Secuencia (15 ép.)", "85.4%", "0.805", "0.80", "No reportado"],
        ["AttnSleep [9]", "20-Fold CV (Sleep-EDF-20)", "Época única + Attn", "84.4%", "0.781", "0.79", "No reportado"],
        ["U-Sleep [8]", "Multicéntrico (19,924 PSG)", "Noche completa", "—", "0.790", "—", "Prototipo web"],
        ["SomnoScope (Este trabajo)", "3-Fold GroupKFold (23 subj., SC+ST)", "Época única (30 s)", "76.27%", "0.716", "0.69", "AWS EC2 + API + MLflow"],
    ]
    add_academic_table(
        doc,
        "Tabla 5. Comparación contextualizada de SomnoScope frente al estado del arte en Sleep-EDF (EEG monocanal Fpz-Cz).",
        t5_headers, t5_rows, col_widths=[1.25, 1.35, 0.90, 0.56, 0.58, 0.45, 1.44], highlight_row=4
    )
    add_body(
        doc,
        "Análisis crítico: La brecha de ~6–9 pp frente a modelos de referencia como TinySleepNet original [6] o "
        "AttnSleep [9] obedece a tres factores metodológicos verificables y deliberados: (i) SomnoScope clasifica "
        "cada época de 30 s de forma independiente (para soportar inferencia unitaria vía API /predict/epoch en "
        "9.47 ms), mientras que [5], [6] explotan secuencias de 15–25 épocas consecutivas (7.5–12.5 minutos de "
        "contexto fisiológico); (ii) la validación 3-Fold entrena con ~67% de los sujetos por fold (15–16 "
        "pacientes) frente al 95% (19 pacientes) de un esquema 20-Fold, además de incluir sujetos de Sleep "
        "Telemetry con mayor fragmentación del sueño; y (iii) el entrenamiento se restringió a 8 épocas en CPU "
        "sin aumento de datos. A cambio, el sistema ofrece un ecosistema MLOps completo, auditable y operativo en "
        "nube con un modelo de apenas 0.62 MB."
    )

    # ------------------------------------------------------------------
    # 5. DISCUSIÓN
    # ------------------------------------------------------------------
    add_h1(doc, "5. Discusión")
    add_body(
        doc,
        "Los resultados obtenidos demuestran que el aprendizaje profundo sobre señales crudas de EEG supera "
        "consistentemente a los ensambles basados en características espectrales manuales cuando se resuelve "
        "correctamente el acondicionamiento numérico de la señal. La normalización Z-score por época (Ecuación (1)) "
        "no solo evitó el colapso de BatchNorm1d ante voltajes de microvoltios, sino que aportó invarianza frente "
        "a diferencias de impedancia entre electrodos y pacientes, habilitando a TinySleepNet v2 para vencer en "
        "los tres folds de GroupKFold y liderar simultáneamente en F1-Macro, Kappa, F1-N1, F1-REM y velocidad de "
        "inferencia nocturna en CPU."
    )
    add_body(
        doc,
        "Al mismo tiempo, los resultados revelan fortalezas y debilidades complementarias que justifican el "
        "diseño Champion/Challenger implementado en SomnoScope. Como fortalezas sobresalen: (1) la reproducibilidad "
        "estadística libre de fuga intra-sujeto en el 100% de los 23 pacientes; (2) la eficiencia computacional "
        "en CPU (1.65 s por noche completa, 0.62 MB de peso), que democratiza el despliegue sin GPU en clínicas "
        "de recursos limitados; y (3) la trazabilidad integral en MLflow y la visualización auditable en el "
        "tablero clínico. Como limitaciones principales se reconocen: (1) la dificultad intrínseca de N1 en EEG "
        "monocanal sin EOG/EMG (F1-N1 de 0.3779), donde persiste un 27% de confusión hacia REM; (2) el uso del "
        "fold externo para selección del mejor checkpoint de TinySleepNet v2 (Sección 3.3), que exige migración "
        "hacia validación cruzada anidada para eliminar cualquier sesgo optimista residual; (3) la ausencia de "
        "modelado secuencial inter-época que permita imponer transiciones fisiológicas suaves en el hipnograma; y "
        "(4) la falta de validación prospectiva en cohortes hospitalarias locales con comorbilidades severas "
        "(apnea grave o epilepsia nocturna), cuya evaluación requerirá aprobación ética institucional."
    )
    add_body(
        doc,
        "En términos de aplicabilidad práctica, SomnoScope se concibe como una herramienta de apoyo a la decisión "
        "clínica (Human-in-the-Loop): al generar en menos de dos segundos un hipnograma preliminar con probabilidades "
        "por época, el especialista puede concentrar su revisión en las épocas de transición ambiguas (especialmente "
        "N1/REM y N2/N3), reduciendo drásticamente el tiempo de lectura sin delegar la responsabilidad diagnóstica."
    )

    # ------------------------------------------------------------------
    # 6. CONCLUSIONES Y TRABAJO FUTURO
    # ------------------------------------------------------------------
    add_h1(doc, "6. Conclusiones y trabajo futuro")
    add_body(
        doc,
        "Este trabajo diseñó, validó y desplegó SomnoScope, una plataforma MLOps de extremo a extremo para la "
        "estadificación automática del sueño sobre EEG monocanal. Mediante validación cruzada estricta 3-Fold "
        "GroupKFold sobre 23 sujetos de Sleep-EDF Expanded (45 noches, 48,378 épocas), TinySleepNet v2 "
        "(Normalizado) demostró ser el modelo campeón coherente con el criterio multiobjetivo predefinido, "
        "ganando en 3/3 folds con exactitud de 76.27 ± 5.21%, F1-Macro de 0.7164 ± 0.0507, Kappa de 0.6870 ± "
        "0.0695, F1-N1 de 0.3779 y F1-REM de 0.7638, además de ser 2.4× más veloz en inferencia nocturna CPU "
        "(1.65 s por 1,000 épocas) que LightGBM. El experimento auxiliar de escalamiento confirmó que ampliar la "
        "cohorte de 9 a 23 sujetos aporta +4.24 pp en Accuracy y +11.90 pp en F1-N3 sobre pacientes inéditos. "
        "Como lecciones aprendidas destacan la importancia crítica de auditar las escalas físicas reales de las "
        "señales biomédicas antes de alimentar capas de normalización en Deep Learning, la necesidad de validar "
        "siempre agrupando por sujeto, y el valor de integrar MLOps desde etapas tempranas."
    )
    add_body(doc, "Como líneas concretas de trabajo futuro para evolucionar la solución se proponen:", space_after=2.5)

    add_bullet(
        doc,
        "Incorporación de parada temprana (early stopping) como parámetro de entrenamiento: ",
        "Integrar early stopping automático tanto en TinySleepNet v2 (paciencia de 5–10 épocas con restauración "
        "de mejores pesos) como en LightGBM (parámetro early_stopping_rounds=30), monitoreando el F1-Macro de "
        "validación (junto con la pérdida de entropía cruzada ponderada) sobre una partición interna de sujetos "
        "disjuntos. Esto previene el sobreajuste hacia patrones idiosincráticos de los pacientes de entrenamiento, "
        "estabiliza la generalización entre folds y evita épocas redundantes de cómputo."
    )
    add_bullet(
        doc,
        "Estrategias avanzadas frente al desbalance de clases (complementando los pesos inversos actuales): ",
        "Aunque la versión actual ya aplica ponderación inversa por clase (Ecuación (2)), en una siguiente "
        "iteración se evaluarán pérdidas focales (Focal Loss), remuestreo consciente de sujeto aplicado "
        "exclusivamente sobre el subconjunto de entrenamiento (sobremuestreo de secuencias con N1 o SMOTE en "
        "espacio tabular, nunca sobre validación/prueba) y ajuste post-hoc de umbrales de probabilidad por clase, "
        "con el objetivo específico de elevar el desempeño en los estadios minoritario y transicional (N1 y REM), "
        "midiendo el avance mediante F1-N1, F1-REM, F1-Macro y Balanced Accuracy."
    )
    add_bullet(
        doc,
        "Validación cruzada anidada (Nested GroupKFold) y modelado secuencial inter-época: ",
        "Desacoplar por completo la selección de hiperparámetros/época óptima de la evaluación final mediante "
        "validación cruzada anidada por sujeto (train / val interna / test externo), incorporando secuencias de "
        "15–20 épocas consecutivas en el bloque BiLSTM o un suavizado probabilístico mediante Modelos Ocultos de "
        "Markov (HMM) para reducir transiciones clínicamente improbables."
    )
    add_bullet(
        doc,
        "Fusión multicanal (EEG+EOG+EMG), expansión a Sleep-EDF-78 y validación clínica prospectiva: ",
        "Incorporar el canal EOG horizontal disponible en Sleep-EDF para desambiguar N1 frente a REM, escalar el "
        "entrenamiento a los 78 sujetos completos de Sleep-EDF Expanded y realizar una evaluación prospectiva con "
        "pruebas de estrés concurrente en entorno hospitalario ([cohorte clínica local pendiente de convenio])."
    )

    # ------------------------------------------------------------------
    # REFERENCIAS (ESTILO IEEE NUMÉRICO EN ORDEN DE APARICIÓN [1]-[15])
    # ------------------------------------------------------------------
    add_h1(doc, "Referencias")
    refs = [
        "[1] R. B. Berry et al., The AASM Manual for the Scoring of Sleep and Associated Events: Rules, Terminology and Technical Specifications, Version 2.6. Darien, IL, USA: American Academy of Sleep Medicine, 2020.",
        "[2] R. S. Rosenberg y S. Van Hout, \"The American Academy of Sleep Medicine inter-scorer reliability program: Sleep stage scoring,\" J. Clin. Sleep Med., vol. 9, no. 1, pp. 81–87, Ene. 2013.",
        "[3] L. Breiman, \"Random forests,\" Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001.",
        "[4] G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" en Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 3146–3154.",
        "[5] A. Supratak, H. Dong, C. Wu y Y. Guo, \"DeepSleepNet: A model for automatic sleep stage scoring based on raw single-channel EEG,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 11, pp. 1998–2008, Nov. 2017.",
        "[6] A. Supratak y Y. Guo, \"TinySleepNet: An efficient deep learning model for sleep stage scoring based on raw single-channel EEG,\" en Proc. 42nd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), Montreal, QC, Canadá, 2020, pp. 641–644.",
        "[7] H. Phan, F. Andreotti, N. Cooray, O. Y. Chén y M. De Vos, \"SeqSleepNet: End-to-end hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 27, no. 3, pp. 400–410, Mar. 2019.",
        "[8] M. Perslev et al., \"U-Sleep: Resilient high-frequency sleep staging,\" npj Digit. Med., vol. 4, no. 1, Art. no. 72, Abr. 2021.",
        "[9] E. Eldele et al., \"An attention-based deep learning approach for sleep stage classification with single-channel EEG,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 29, pp. 809–818, 2021.",
        "[10] D. Sculley et al., \"Hidden technical debt in machine learning systems,\" en Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 28, Montreal, QC, Canadá, 2015, pp. 2503–2511.",
        "[11] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen y J. J. L. Oberyé, \"Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG,\" IEEE Trans. Biomed. Eng., vol. 47, no. 9, pp. 1185–1194, Sep. 2000. (Dataset: Sleep-EDF Database Expanded, PhysioNet).",
        "[12] A. Gramfort et al., \"MEG and EEG data analysis with MNE-Python,\" Front. Neurosci., vol. 7, Art. no. 267, Dic. 2013.",
        "[13] A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" en Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, Vancouver, BC, Canadá, 2019, pp. 8024–8035.",
        "[14] M. Zaharia et al., \"Accelerating the machine learning lifecycle with MLflow,\" IEEE Data Eng. Bull., vol. 41, no. 4, pp. 39–45, Dic. 2018.",
        "[15] S. Ramírez, \"FastAPI: Modern, high-performance web framework for building APIs with Python,\" Software Documentation, v0.115, 2024. Disponible en: https://fastapi.tiangolo.com/",
    ]
    for rtxt in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Cm(0.65)
        p.paragraph_format.first_line_indent = Cm(-0.65)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(rtxt)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.8)

    out_docx = REPO_ROOT / "Articulo Cientifico Final - SomnoScope (23 Sujetos + CV).docx"
    doc.save(str(out_docx))

    total_words = sum(len(p.text.split()) for p in doc.paragraphs)
    # Estimación conservadora de páginas (Times New Roman 10pt sencillo Carta 2.5cm ≈ 550-600 palabras/pág. + 3 figuras + 5 tablas)
    est_pages = (total_words / 540.0) + 1.8 + 1.4
    print(f"[OK] Artículo Científico generado en: {out_docx}")
    print(f"     Párrafos: {len(doc.paragraphs)} | Tablas (incl. 3 ec.): {len(doc.tables)} | Palabras cuerpo: {total_words}")
    print(f"     Estimación de páginas (límite rúbrica <= 12): ~{est_pages:.1f} páginas")


if __name__ == "__main__":
    build_article()
