"""
performance.py
---------------
The "Model Performance" tab: shows the comparison table of all four
models, highlights the automatically-selected best model, and lets the
user open the confusion matrix image.
"""

import os
import tkinter as tk
from tkinter import ttk

from utils.image_viewer import open_image

BG = "#f4f6f9"
CARD_BG = "#ffffff"
ACCENT = "#2c3e50"
TEXT_MUTED = "#7f8c8d"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM_IMAGE_PATH = os.path.join(BASE_DIR, "visualizations", "confusion_matrix.png")


class PerformanceFrame(tk.Frame):
    def __init__(self, parent, metadata):
        super().__init__(parent, bg=BG)
        self.metadata = metadata
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self, text="Model Performance", font=("Segoe UI", 20, "bold"),
            bg=BG, fg=ACCENT, anchor="w",
        )
        title.pack(fill="x", padx=30, pady=(25, 10))

        best_model = self.metadata.get("best_model_name", "N/A")
        subtitle = tk.Label(
            self,
            text=(
                f"Selected Model: {best_model}  "
                f"(chosen automatically using F1-score as the selection metric)"
            ),
            font=("Segoe UI", 10, "italic"), bg=BG, fg=TEXT_MUTED, anchor="w",
        )
        subtitle.pack(fill="x", padx=30, pady=(0, 15))

        # Comparison table
        table_card = tk.Frame(self, bg=CARD_BG, highlightbackground="#dcdde1",
                               highlightthickness=1)
        table_card.pack(fill="x", padx=30, pady=(0, 15))

        columns = ("model", "accuracy", "precision", "recall", "f1_score")
        headers = {
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1_score": "F1 Score",
        }
        tree = ttk.Treeview(table_card, columns=columns, show="headings", height=5)
        for col in columns:
            tree.heading(col, text=headers[col])
            tree.column(col, width=140, anchor="center")
        tree.pack(fill="x", padx=15, pady=15)

        comparison = self.metadata.get("model_comparison", {})
        for model_name, metrics in comparison.items():
            marker = " (Selected)" if model_name == best_model else ""
            tree.insert(
                "", "end",
                values=(
                    model_name + marker,
                    f"{metrics['accuracy'] * 100:.2f}%",
                    f"{metrics['precision'] * 100:.2f}%",
                    f"{metrics['recall'] * 100:.2f}%",
                    f"{metrics['f1_score'] * 100:.2f}%",
                ),
            )

        # Confusion matrix summary
        cm = self.metadata.get("confusion_matrix", {})
        cm_card = tk.Frame(self, bg=CARD_BG, highlightbackground="#dcdde1",
                            highlightthickness=1)
        cm_card.pack(fill="x", padx=30, pady=(0, 15))

        tk.Label(cm_card, text="Confusion Matrix (Selected Model)",
                 font=("Segoe UI", 12, "bold"), bg=CARD_BG, fg=ACCENT).pack(
            anchor="w", padx=15, pady=(15, 5))

        cm_text = (
            f"True Positive (TP): {cm.get('tp', 0)}     "
            f"True Negative (TN): {cm.get('tn', 0)}     "
            f"False Positive (FP): {cm.get('fp', 0)}     "
            f"False Negative (FN): {cm.get('fn', 0)}"
        )
        tk.Label(cm_card, text=cm_text, font=("Segoe UI", 10), bg=CARD_BG,
                 fg=ACCENT).pack(anchor="w", padx=15, pady=(0, 10))

        tk.Button(
            cm_card, text="Open Confusion Matrix Image", font=("Segoe UI", 9, "bold"),
            bg="#2980b9", fg="white", relief="flat", padx=15, pady=6,
            command=lambda: open_image(CM_IMAGE_PATH),
        ).pack(anchor="w", padx=15, pady=(0, 15))

        note = tk.Label(
            self,
            text=(
                "Accuracy, Precision, Recall and F1-score are computed on a held-out "
                "test set. The model with the highest F1-score is selected automatically "
                "- this is not a guarantee of real-world performance."
            ),
            font=("Segoe UI", 9, "italic"), bg=BG, fg=TEXT_MUTED,
            wraplength=760, justify="left",
        )
        note.pack(fill="x", padx=30, pady=(5, 10), anchor="w")
