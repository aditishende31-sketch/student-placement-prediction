"""
feature_importance.py
------------------------
The "Feature Importance / About" tab: shows a sorted feature-importance
table, a button to open the graph, and general information about the app.
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
FI_IMAGE_PATH = os.path.join(BASE_DIR, "visualizations", "feature_importance.png")


class FeatureImportanceFrame(tk.Frame):
    def __init__(self, parent, metadata):
        super().__init__(parent, bg=BG)
        self.metadata = metadata
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self, text="Feature Importance / About", font=("Segoe UI", 20, "bold"),
            bg=BG, fg=ACCENT, anchor="w",
        )
        title.pack(fill="x", padx=30, pady=(25, 10))

        table_card = tk.Frame(self, bg=CARD_BG, highlightbackground="#dcdde1",
                               highlightthickness=1)
        table_card.pack(fill="x", padx=30, pady=(0, 15))

        tk.Label(table_card, text="Feature Importance (Random Forest)",
                 font=("Segoe UI", 12, "bold"), bg=CARD_BG, fg=ACCENT).pack(
            anchor="w", padx=15, pady=(15, 5))

        columns = ("feature", "importance")
        tree = ttk.Treeview(table_card, columns=columns, show="headings", height=8)
        tree.heading("feature", text="Feature")
        tree.heading("importance", text="Importance")
        tree.column("feature", width=250, anchor="w")
        tree.column("importance", width=120, anchor="center")
        tree.pack(fill="x", padx=15, pady=(0, 10))

        for feature, importance in self.metadata.get("feature_importance", []):
            tree.insert("", "end", values=(feature.replace("_", " "), f"{importance:.4f}"))

        tk.Button(
            table_card, text="Open Feature Importance Graph", font=("Segoe UI", 9, "bold"),
            bg="#2980b9", fg="white", relief="flat", padx=15, pady=6,
            command=lambda: open_image(FI_IMAGE_PATH),
        ).pack(anchor="w", padx=15, pady=(0, 15))

        note = tk.Label(
            self,
            text=(
                "Feature importance indicates how useful each feature was to the trained "
                "model when making predictions on this synthetic dataset. It does NOT "
                "establish that a feature causes placement or non-placement."
            ),
            font=("Segoe UI", 9, "italic"), bg=BG, fg=TEXT_MUTED,
            wraplength=760, justify="left",
        )
        note.pack(fill="x", padx=30, pady=(0, 15), anchor="w")

        about_card = tk.Frame(self, bg=CARD_BG, highlightbackground="#dcdde1",
                               highlightthickness=1)
        about_card.pack(fill="x", padx=30, pady=(0, 15))

        tk.Label(about_card, text="About This Project", font=("Segoe UI", 12, "bold"),
                 bg=CARD_BG, fg=ACCENT).pack(anchor="w", padx=15, pady=(15, 5))

        about_text = (
            "Student Placement Predictor is an academic machine-learning mini-project.\n"
            "It uses a synthetically generated dataset (not real recruitment data) and "
            "compares four classification algorithms - Logistic Regression, Decision Tree, "
            "Random Forest, and K-Nearest Neighbors - selecting the best one automatically "
            "based on F1-score.\n\n"
            "Predictions made here are for educational demonstration only and do not "
            "guarantee actual employment or placement outcomes."
        )
        tk.Label(about_card, text=about_text, font=("Segoe UI", 9), bg=CARD_BG,
                 fg=ACCENT, wraplength=760, justify="left").pack(
            anchor="w", padx=15, pady=(0, 15))
