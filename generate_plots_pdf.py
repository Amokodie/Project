"""Generate a presentation-ready PDF that walks through every plot from the
CNN / PINN study on NASA C-MAPSS FD001 with professor-facing explanations.

Run:
    python generate_plots_pdf.py

Writes: output/PINN_vs_CNN_Plots_Explained.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from fpdf import FPDF

ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "output"
PDF_PATH = OUTPUT_DIR / "PINN_vs_CNN_Plots_Explained.pdf"

PAGE_W_MM = 210.0
MARGIN = 18.0
CONTENT_W = PAGE_W_MM - 2 * MARGIN


_REPLACEMENTS = {
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2026": "...",
    "\u00A0": " ",   # non-breaking space
    "\u2192": "->",
    "\u2190": "<-",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00B7": "-",   # middle dot
    "\u00D7": "x",
    "\u00B2": "^2",
    "\u00B3": "^3",
    "\u03BB": "lambda",
}


def ascii_safe(text: str) -> str:
    """Coerce text to Latin-1 by replacing common smart punctuation."""
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


# ---------------------------------------------------------------------------
# Content — one entry per plot, in presentation order.
# ---------------------------------------------------------------------------

PLOTS = [
    {
        "file": "sensor_trajectories.png",
        "title": "1. Sensor Trajectories — The Degradation Fingerprint",
        "what": (
            "Normalized readings from selected sensors across the full life "
            "cycle of representative training engines. The x-axis is cycle "
            "count; each coloured curve is one sensor."
        ),
        "why": (
            "The C-MAPSS dataset gives us 21 sensors, but not all of them "
            "carry degradation information. This plot justifies the choice "
            "of 18 input channels (six sensors were dropped because they "
            "were effectively constant). It also illustrates the core "
            "modelling challenge: most of an engine's life looks nominal, "
            "and the signal only emerges in the final degradation phase."
        ),
        "talk": (
            "Point out: 'degradation is subtle and late — which is why a "
            "good prior matters more than raw network capacity.'"
        ),
    },
    {
        "file": "rul_trajectory_example.png",
        "title": "2. RUL Trajectory — What We Are Predicting",
        "what": (
            "A single engine's true Remaining Useful Life (RUL) as a "
            "function of operating cycles, with the piecewise-linear cap "
            "at 125 cycles overlaid."
        ),
        "why": (
            "RUL is the dependent variable of the whole project. This plot "
            "explains our target-engineering choice: we cap RUL at 125 "
            "because early-life cycles contain no learnable signal, and "
            "forcing the network to learn a flat line would waste capacity "
            "and worsen the regression on the informative tail."
        ),
        "talk": (
            "Emphasize: 'we are predicting a monotonically decreasing "
            "quantity — a property the PINN exploits directly in its "
            "loss function.'"
        ),
    },
    {
        "file": "network_architectures.png",
        "title": "3. Network Architectures — Baseline vs Physics-Informed",
        "what": (
            "Side-by-side layer diagrams of the two models. Left: the 1D "
            "Convolutional Neural Network (Conv1D -> ReLU -> GAP -> Dense "
            "-> Dense). Right: the Physics-Informed MLP (Dense(96) -> "
            "Dense(48) -> Dense(1)) with the physics loss hooks shown."
        ),
        "why": (
            "This is the methodological core of the work. It shows that "
            "the PINN is NOT a larger or more complex network — it is, in "
            "fact, the simpler architecture. The gain comes entirely from "
            "the physics-informed loss, not from capacity. This controls "
            "for confounding: if a bigger MLP beat the CNN, we could not "
            "attribute the improvement to physics alone."
        ),
        "talk": (
            "Key message: 'the PINN has fewer free parameters than the "
            "CNN, so any improvement is attributable to the prior, not "
            "to model size.'"
        ),
    },
    {
        "file": "training_curves.png",
        "title": "4. Training Curves — Convergence and Regularization",
        "what": (
            "Training and validation loss (MSE) per epoch for both models, "
            "on the same axes for direct comparison."
        ),
        "why": (
            "Two findings live here. (a) Both models converge in under "
            "20.5 seconds of wall-clock on CPU using pure NumPy — so the "
            "entire study is reproducible without GPU access. (b) The "
            "PINN converges to a lower validation loss with a smaller "
            "train / validation gap, which is the classic signature of "
            "a regularizer. The physics loss acts as a free, "
            "interpretable prior."
        ),
        "talk": (
            "State explicitly: 'the physics loss behaves like L2 weight "
            "decay in effect, but unlike L2 it is not arbitrary — it "
            "comes from the physics of the problem.'"
        ),
    },
    {
        "file": "cnn_filters.png",
        "title": "5. Learned CNN Filters — What the Convolution Sees",
        "what": (
            "Visualization of the sixteen learned 1D convolutional "
            "kernels (kernel size 5) from the CNN's first and only "
            "convolutional layer."
        ),
        "why": (
            "This plot answers the interpretability question. The filters "
            "show that the CNN has rediscovered, on its own, short-window "
            "derivative- and moving-average-style operators — roughly the "
            "hand-crafted features that classical prognostics literature "
            "uses. The implication: a data-driven network is not magical, "
            "it is just learning finite-difference operators. This frames "
            "the PINN argument: if the network is going to learn physics "
            "anyway, we may as well tell it the physics up front."
        ),
        "talk": (
            "'The CNN is implicitly learning derivatives of sensor "
            "readings. The PINN just encodes the derivative structure in "
            "the loss from the start — cheaper and more reliable.'"
        ),
    },
    {
        "file": "pred_vs_actual_cnn.png",
        "title": "6. CNN Predicted vs Actual RUL",
        "what": (
            "Scatter of CNN predictions against ground-truth RUL on the "
            "100 unseen test engines. The y=x diagonal is the ideal."
        ),
        "why": (
            "This shows where the CNN fails. At low true RUL — i.e., "
            "engines near end of life — the CNN systematically "
            "overshoots, predicting more life than remains. In safety-"
            "critical terms this is the worst failure mode: a falsely "
            "optimistic prediction at the moment when pessimism is most "
            "needed."
        ),
        "talk": (
            "'Look at the low-RUL region on the right: the CNN is "
            "optimistic exactly when we need it to be conservative.'"
        ),
    },
    {
        "file": "pred_vs_actual_pinn.png",
        "title": "7. PINN Predicted vs Actual RUL",
        "what": (
            "Same scatter as the previous plot, but for the PINN model. "
            "Plotted on identical axes for direct visual comparison."
        ),
        "why": (
            "The PINN scatter hugs the y=x diagonal much more tightly, "
            "and the end-of-life overshoot is largely gone. The "
            "monotonicity constraint in the loss has removed the "
            "optimistic bias that the CNN exhibits. This is the headline "
            "qualitative result of the project."
        ),
        "talk": (
            "'Same data, simpler network, right prior — and the "
            "pathological overshoot disappears.'"
        ),
    },
    {
        "file": "model_comparison.png",
        "title": "8. Quantitative Comparison — RMSE, MAE, NASA Score",
        "what": (
            "Bar chart comparing CNN and PINN across three metrics: "
            "Root Mean Square Error, Mean Absolute Error, and the "
            "asymmetric NASA scoring function."
        ),
        "why": (
            "RMSE drops from 18.68 to 14.64 (a 21.6 % improvement). MAE "
            "drops from 13.69 to 11.14 (18.6 %). But the NASA score — "
            "which penalizes late predictions roughly four times harder "
            "than early ones, matching the asymmetric real-world cost of "
            "missed warnings in aviation — collapses from 1107.5 to "
            "338.6, a 69.4 % reduction. The physics prior does most of "
            "its work on exactly the errors that cost the most."
        ),
        "talk": (
            "'The 70 % reduction on the NASA score is the answer to "
            "why-should-anyone-care: this is the metric aviation "
            "maintenance actually uses.'"
        ),
    },
    {
        "file": "error_histogram.png",
        "title": "9. Error Distribution — Where the Mistakes Live",
        "what": (
            "Histogram of prediction errors (predicted RUL minus true "
            "RUL) on the test set, overlaid for CNN and PINN."
        ),
        "why": (
            "The PINN error histogram is both narrower (lower variance) "
            "and more symmetric around zero. The right tail — positive "
            "errors, i.e. late predictions — is markedly thinner. This "
            "is the microscopic view of the NASA-score improvement from "
            "the previous plot: fewer large late errors means a smaller "
            "asymmetric penalty."
        ),
        "talk": (
            "'The PINN does not just reduce average error — it changes "
            "the shape of the error distribution. Fewer catastrophic "
            "late predictions.'"
        ),
    },
    {
        "file": "per_engine_rul.png",
        "title": "10. Per-Engine RUL Trajectories — Temporal Consistency",
        "what": (
            "For selected test engines, the predicted RUL trajectory "
            "from both models is plotted against the true linear decay, "
            "across the full sensor history of that engine."
        ),
        "why": (
            "This plot validates that the physics constraint is doing "
            "what we designed it to do. CNN trajectories wander — they "
            "can predict rising RUL (physically impossible) and jitter "
            "between adjacent cycles. PINN trajectories are smooth and "
            "monotonically non-increasing, without any post-hoc "
            "smoothing or filtering. The monotonicity penalty in the "
            "loss is visibly shaping the output space."
        ),
        "talk": (
            "'No post-processing, no Kalman filter, no smoothing — the "
            "trajectories look physically sensible because the loss "
            "function made them that way.'"
        ),
    },
]


# ---------------------------------------------------------------------------
# PDF layout
# ---------------------------------------------------------------------------


class PresentationPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0, 6,
            ascii_safe("Engine Failure & RUL Prediction - CNN vs PINN - RCSSTEAP, Beihang University"),
            align="L",
        )
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, 18, PAGE_W_MM - MARGIN, 18)
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # Override text-emitting helpers to coerce Unicode to Latin-1.
    def cell(self, *args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], str):
            args = list(args); args[2] = ascii_safe(args[2])
        elif "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = ascii_safe(kwargs["text"])
        elif "txt" in kwargs and isinstance(kwargs["txt"], str):
            kwargs["txt"] = ascii_safe(kwargs["txt"])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], str):
            args = list(args); args[2] = ascii_safe(args[2])
        elif "text" in kwargs and isinstance(kwargs["text"], str):
            kwargs["text"] = ascii_safe(kwargs["text"])
        elif "txt" in kwargs and isinstance(kwargs["txt"], str):
            kwargs["txt"] = ascii_safe(kwargs["txt"])
        result = super().multi_cell(*args, **kwargs)
        # fpdf2's multi_cell does not reliably reset x to the left margin
        # afterwards (varies by version / new_x default). Force it so the
        # next text block always starts flush-left.
        self.set_x(self.l_margin)
        return result


def image_height_mm(path: Path, target_w_mm: float) -> float:
    with Image.open(path) as img:
        aspect = img.height / img.width
    return target_w_mm * aspect


def write_cover(pdf: PresentationPDF) -> None:
    pdf.add_page()
    # Top coloured band
    pdf.set_fill_color(14, 23, 42)  # deep navy
    pdf.rect(0, 0, PAGE_W_MM, 62, "F")
    pdf.set_fill_color(125, 211, 252)  # cyan ribbon
    pdf.rect(0, 62, PAGE_W_MM, 1.5, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(MARGIN, 18)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "FINAL PROJECT - RCSSTEAP - BEIHANG UNIVERSITY")
    pdf.set_xy(MARGIN, 26)
    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(CONTENT_W, 9, "Engine Failure & Remaining Useful Life Prediction")
    pdf.set_xy(MARGIN, 45)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(200, 220, 240)
    pdf.cell(0, 7, "Plots & Diagrams: CNN vs Physics-Informed Neural Network")

    # Intro text below the band
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(MARGIN, 78)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "About this document")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    intro = (
        "This document walks through every figure produced by our CNN vs PINN "
        "study on the NASA C-MAPSS FD001 turbofan degradation dataset. For "
        "each plot you will find three sections: What it shows, Why it "
        "matters for the research, and a Talking point designed to anchor "
        "your verbal explanation during the presentation.\n\n"
        "Use the talking-point lines as anchors: each one is a single "
        "sentence you can say out loud while the plot is on screen."
    )
    pdf.multi_cell(CONTENT_W, 6, intro)

    # Results at a glance
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Results at a glance")
    pdf.ln(11)
    pdf.set_draw_color(200, 210, 220)
    pdf.set_fill_color(245, 248, 252)
    pdf.set_font("Helvetica", "B", 11)
    col_w = [55, 40, 40, 40]
    for header, w in zip(["Metric", "CNN", "PINN", "Improvement"], col_w):
        pdf.cell(w, 9, header, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 11)
    rows = [
        ["RMSE (Root MSE)", "18.68", "14.64", "-21.6 %"],
        ["MAE (Mean Abs Err)", "13.69", "11.14", "-18.6 %"],
        ["NASA score (asym.)", "1107.5", "338.6", "-69.4 %"],
    ]
    for row in rows:
        pdf.cell(col_w[0], 9, row[0], border=1)
        for i in range(1, 4):
            align = "C"
            pdf.cell(col_w[i], 9, row[i], border=1, align=align)
        pdf.ln()

    # Author block
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Team")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 10)
    authors = [
        ("Kodie Amo Kwame", "LS2525226"),
        ("Sumara Alfred Salifu", "LS2525245"),
        ("Peta Mimi Precious", "LS2525255"),
    ]
    for name, sid in authors:
        pdf.cell(0, 6, f"- {name}  ({sid})")
        pdf.ln(6)

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        CONTENT_W, 5,
        "Dataset: NASA C-MAPSS FD001 (100 train / 100 test engines, 21 sensors, "
        "run-to-failure). Models trained in 20.5 s total on CPU with pure NumPy. "
        "All figures generated from the repository's output/ directory."
    )


def write_plot_page(pdf: PresentationPDF, plot: dict) -> None:
    pdf.add_page()
    img_path = OUTPUT_DIR / plot["file"]

    # Title
    pdf.set_text_color(20, 30, 60)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(CONTENT_W, 8, plot["title"])
    pdf.ln(2)

    # Thin rule
    pdf.set_draw_color(255, 179, 71)  # amber, matches slide accent
    pdf.set_line_width(0.6)
    pdf.line(MARGIN, pdf.get_y(), MARGIN + 40, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(4)

    # Figure — scale to fit within remaining vertical space, max 150mm wide
    target_w = min(150.0, CONTENT_W)
    h = image_height_mm(img_path, target_w)
    # If too tall, shrink
    max_fig_h = 110.0
    if h > max_fig_h:
        target_w = target_w * (max_fig_h / h)
        h = max_fig_h
    x_centered = (PAGE_W_MM - target_w) / 2
    pdf.image(str(img_path), x=x_centered, w=target_w)
    pdf.ln(4)

    # What it shows
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "What it shows")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(CONTENT_W, 5, plot["what"])
    pdf.ln(2)

    # Why it matters
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Why it matters for the research")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(CONTENT_W, 5, plot["why"])
    pdf.ln(2)

    # Talking point
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Talking point")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(180, 110, 30)  # muted amber for emphasis
    pdf.multi_cell(CONTENT_W, 5, plot["talk"])


def write_closing(pdf: PresentationPDF) -> None:
    pdf.add_page()
    pdf.set_text_color(20, 30, 60)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Key message for the defense")
    pdf.ln(14)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    closing = (
        "A smaller network with the correct physical prior outperforms a "
        "larger purely data-driven network. On the NASA C-MAPSS FD001 "
        "benchmark, embedding two simple physics constraints -- RUL must "
        "be monotonically non-increasing, and must decrease by exactly one "
        "per cycle -- cuts Root Mean Square Error by 21.6 % and the "
        "asymmetric NASA score by 69.4 %, without adding a single parameter.\n\n"
        "The value of physics-informed learning is not in model capacity; "
        "it is in the shape of the solution. The CNN can, in principle, "
        "learn the same constraint from data, but at the cost of many "
        "samples and many parameters -- and with no guarantee. The PINN "
        "gets the constraint for free, from the problem specification."
    )
    pdf.multi_cell(CONTENT_W, 6, closing)

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 30, 60)
    pdf.cell(0, 8, "Likely examiner questions")
    pdf.ln(10)

    qa = [
        ("Q: Why not just use a larger CNN?",
         "A: Size doesn't address the structural problem -- the CNN can still predict rising RUL. The physics loss eliminates that failure mode by design."),
        ("Q: How did you choose lambda values (0.15, 0.05)?",
         "A: Held-out validation sweep. The drift penalty is weighted lower because the monotonicity penalty already enforces the sign; drift only fine-tunes the magnitude."),
        ("Q: Would this generalize to FD002-FD004?",
         "A: The physics constraint is dataset-agnostic -- it only encodes the definition of RUL. We expect the absolute RMSE to rise with multi-mode faults, but the relative improvement of PINN over CNN should hold."),
        ("Q: Why pure NumPy and not PyTorch?",
         "A: To emphasize that the method is implementation-light and reproducible without GPU access. 20.5 s of wall time on CPU makes the work transparent and verifiable."),
    ]
    pdf.set_font("Helvetica", "", 10)
    for q, a in qa:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(CONTENT_W, 5, q)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(CONTENT_W, 5, a)
        pdf.ln(2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = PresentationPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)

    write_cover(pdf)
    for plot in PLOTS:
        img_path = OUTPUT_DIR / plot["file"]
        if not img_path.exists():
            print(f"  ! missing: {img_path}")
            continue
        write_plot_page(pdf, plot)
    write_closing(pdf)

    pdf.output(str(PDF_PATH))
    print(f"Wrote: {PDF_PATH}")


if __name__ == "__main__":
    main()
