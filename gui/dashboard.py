"""
dashboard.py
-------------
The Dashboard tab: shows dataset stats, best model, and prediction count
in simple, clearly formatted "cards".
"""

import tkinter as tk
from tkinter import ttk

from utils.database import count_predictions

BG = "#f4f6f9"
CARD_BG = "#ffffff"
ACCENT = "#2c3e50"
TEXT_MUTED = "#7f8c8d"


class DashboardFrame(tk.Frame):
    def __init__(self, parent, metadata):
        super().__init__(parent, bg=BG)
        self.metadata = metadata
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self, text="Dashboard", font=("Segoe UI", 20, "bold"),
            bg=BG, fg=ACCENT, anchor="w",
        )
        title.pack(fill="x", padx=30, pady=(25, 15))

        cards_frame = tk.Frame(self, bg=BG)
        cards_frame.pack(fill="x", padx=30)

        summary = self.metadata.get("dataset_summary", {})
        total = summary.get("total_records", 0)
        placed = summary.get("placed_count", 0)
        not_placed = summary.get("not_placed_count", 0)
        best_model = self.metadata.get("best_model_name", "N/A")
        best_metrics = self.metadata.get("model_comparison", {}).get(best_model, {})
        accuracy = best_metrics.get("accuracy", 0)
        predictions_made = count_predictions()

        cards = [
            ("Total Dataset Records", str(total), "#3498db"),
            ("Placed Students", str(placed), "#27ae60"),
            ("Not Placed Students", str(not_placed), "#e74c3c"),
            ("Best Model", best_model, "#8e44ad"),
            ("Best Model Accuracy", f"{accuracy * 100:.1f}%", "#f39c12"),
            ("Predictions Made", str(predictions_made), "#16a085"),
        ]

        for i, (label, value, color) in enumerate(cards):
            row, col = divmod(i, 3)
            card = tk.Frame(cards_frame, bg=CARD_BG, highlightbackground="#dcdde1",
                             highlightthickness=1)
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            cards_frame.grid_columnconfigure(col, weight=1)

            bar = tk.Frame(card, bg=color, width=6)
            bar.pack(side="left", fill="y")

            inner = tk.Frame(card, bg=CARD_BG)
            inner.pack(side="left", fill="both", expand=True, padx=15, pady=15)

            tk.Label(inner, text=value, font=("Segoe UI", 18, "bold"),
                     bg=CARD_BG, fg=ACCENT, anchor="w").pack(anchor="w")
            tk.Label(inner, text=label, font=("Segoe UI", 10),
                     bg=CARD_BG, fg=TEXT_MUTED, anchor="w").pack(anchor="w")

        info = tk.Label(
            self,
            text=(
                "This dashboard summarizes the synthetic training dataset and the "
                "currently selected ML model. Predictions made in this app are for "
                "academic demonstration only and do not guarantee real placement outcomes."
            ),
            font=("Segoe UI", 9, "italic"), bg=BG, fg=TEXT_MUTED,
            wraplength=760, justify="left",
        )
        info.pack(fill="x", padx=30, pady=(25, 10), anchor="w")

    def refresh(self):
        """Rebuilds the dashboard (used after a new prediction is made)."""
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()
