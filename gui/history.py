"""
history.py
-----------
The "Prediction History" tab: displays every past prediction stored in
SQLite using a Treeview table, with Refresh and Clear buttons.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from utils.database import fetch_all_predictions, clear_history

BG = "#f4f6f9"
ACCENT = "#2c3e50"

COLUMNS = ("student_id", "cgpa", "projects", "internships", "prediction", "probability", "created_at")
HEADERS = {
    "student_id": "Student ID",
    "cgpa": "CGPA",
    "projects": "Projects",
    "internships": "Internships",
    "prediction": "Prediction",
    "probability": "Probability (%)",
    "created_at": "Date/Time",
}


class HistoryFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        title = tk.Label(
            self, text="Prediction History", font=("Segoe UI", 20, "bold"),
            bg=BG, fg=ACCENT, anchor="w",
        )
        title.pack(fill="x", padx=30, pady=(25, 10))

        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(fill="x", padx=30)

        tk.Button(
            btn_frame, text="Refresh", font=("Segoe UI", 9, "bold"),
            bg="#2980b9", fg="white", relief="flat", padx=15, pady=6,
            command=self.refresh,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            btn_frame, text="Clear History", font=("Segoe UI", 9, "bold"),
            bg="#c0392b", fg="white", relief="flat", padx=15, pady=6,
            command=self._on_clear_clicked,
        ).pack(side="left")

        table_frame = tk.Frame(self, bg=BG)
        table_frame.pack(fill="both", expand=True, padx=30, pady=15)

        self.tree = ttk.Treeview(table_frame, columns=COLUMNS, show="headings", height=18)
        for col in COLUMNS:
            self.tree.heading(col, text=HEADERS[col])
            self.tree.column(col, width=110, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        rows = fetch_all_predictions()
        for row in rows:
            student_id, cgpa, projects, internships, prediction, probability, created_at = row
            self.tree.insert(
                "", "end",
                values=(student_id, cgpa, projects, internships, prediction,
                        f"{probability:.1f}", created_at),
            )

    def _on_clear_clicked(self):
        confirmed = messagebox.askyesno(
            "Clear History", "This will permanently delete all prediction history. Continue?"
        )
        if confirmed:
            clear_history()
            self.refresh()
