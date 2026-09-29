"""
prediction_form.py
--------------------
The "Predict Placement" tab: a form with the 13 required input fields,
input validation, model prediction, probability display, saving to the
history database, and general improvement suggestions.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

from utils.validation import validate_form, ordered_feature_list
from utils.database import get_next_student_id, insert_prediction
from ml.data_generator import FEATURE_COLUMNS

BG = "#f4f6f9"
CARD_BG = "#ffffff"
ACCENT = "#2c3e50"
TEXT_MUTED = "#7f8c8d"

FORM_FIELDS = [
    ("cgpa", "CGPA (0-10)"),
    ("tenth", "10th Percentage (0-100)"),
    ("twelfth", "12th/Diploma Percentage (0-100)"),
    ("attendance", "Attendance Percentage (0-100)"),
    ("backlogs", "Number of Backlogs"),
    ("internships", "Number of Internships"),
    ("projects", "Number of Projects Completed"),
    ("certifications", "Number of Certifications"),
    ("coding", "Coding Score (0-100)"),
    ("communication", "Communication Score (0-100)"),
    ("aptitude", "Aptitude Score (0-100)"),
    ("technical", "Technical Interview Score (0-100)"),
    ("soft_skills", "Soft Skills Score (0-100)"),
]


class PredictionFormFrame(tk.Frame):
    def __init__(self, parent, model, scaler, on_prediction_saved=None):
        super().__init__(parent, bg=BG)
        self.model = model
        self.scaler = scaler
        self.on_prediction_saved = on_prediction_saved
        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self, text="Predict Placement", font=("Segoe UI", 20, "bold"),
            bg=BG, fg=ACCENT, anchor="w",
        )
        title.pack(fill="x", padx=30, pady=(25, 10))

        # Scrollable container in case the window is small
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=30)

        form_card = tk.Frame(container, bg=CARD_BG, highlightbackground="#dcdde1",
                              highlightthickness=1)
        form_card.pack(side="left", fill="both", expand=True, padx=(0, 15), pady=10)

        form_inner = tk.Frame(form_card, bg=CARD_BG)
        form_inner.pack(fill="both", expand=True, padx=20, pady=20)

        for i, (key, label) in enumerate(FORM_FIELDS):
            row, col = divmod(i, 2)
            cell = tk.Frame(form_inner, bg=CARD_BG)
            cell.grid(row=row, column=col, sticky="ew", padx=10, pady=8)
            form_inner.grid_columnconfigure(col, weight=1)

            tk.Label(cell, text=label, font=("Segoe UI", 9), bg=CARD_BG,
                     fg=ACCENT, anchor="w").pack(fill="x")
            entry = tk.Entry(cell, font=("Segoe UI", 10))
            entry.pack(fill="x")
            self.entries[key] = entry

        predict_btn = tk.Button(
            form_inner, text="PREDICT PLACEMENT", font=("Segoe UI", 11, "bold"),
            bg="#2980b9", fg="white", activebackground="#3498db",
            activeforeground="white", relief="flat", padx=20, pady=10,
            command=self._on_predict_clicked,
        )
        predict_btn.grid(row=(len(FORM_FIELDS) // 2) + 1, column=0, columnspan=2,
                          pady=(15, 0), sticky="ew", padx=10)

        clear_btn = tk.Button(
            form_inner, text="Clear Form", font=("Segoe UI", 9),
            bg="#bdc3c7", fg="#2c3e50", relief="flat", padx=10, pady=6,
            command=self._clear_form,
        )
        clear_btn.grid(row=(len(FORM_FIELDS) // 2) + 2, column=0, columnspan=2,
                        pady=(8, 0), sticky="ew", padx=10)

        # Result panel
        self.result_card = tk.Frame(container, bg=CARD_BG, highlightbackground="#dcdde1",
                                     highlightthickness=1, width=280)
        self.result_card.pack(side="left", fill="both", padx=(15, 0), pady=10)
        self.result_card.pack_propagate(False)

        self._render_empty_result()

    def _render_empty_result(self):
        for widget in self.result_card.winfo_children():
            widget.destroy()
        tk.Label(
            self.result_card, text="PLACEMENT RESULT", font=("Segoe UI", 12, "bold"),
            bg=CARD_BG, fg=ACCENT,
        ).pack(pady=(25, 10))
        tk.Label(
            self.result_card,
            text="Fill in the form and click\nPREDICT PLACEMENT to see\nthe result here.",
            font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_MUTED, justify="center",
        ).pack(pady=10)

    def _clear_form(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self._render_empty_result()

    def _on_predict_clicked(self):
        raw_values = {key: entry.get() for key, entry in self.entries.items()}
        is_valid, cleaned, error_message = validate_form(raw_values)

        if not is_valid:
            messagebox.showerror("Invalid Input", error_message)
            return

        features = ordered_feature_list(cleaned)
        try:
            features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            scaled_features = self.scaler.transform(features_df)
            prediction = self.model.predict(scaled_features)[0]
            probability_placed = self.model.predict_proba(scaled_features)[0][1] * 100
        except Exception as exc:
            messagebox.showerror(
                "Prediction Error",
                f"Something went wrong while making the prediction:\n{exc}",
            )
            return

        prediction_label = "PLACED" if prediction == 1 else "NOT PLACED"

        student_id = get_next_student_id()
        insert_prediction(
            student_id=student_id,
            cgpa=cleaned["cgpa"],
            projects=int(cleaned["projects"]),
            internships=int(cleaned["internships"]),
            prediction=prediction_label,
            probability=round(probability_placed, 1),
        )

        self._render_result(student_id, prediction_label, probability_placed, cleaned)

        if self.on_prediction_saved:
            self.on_prediction_saved()

    def _render_result(self, student_id, prediction_label, probability_placed, cleaned):
        for widget in self.result_card.winfo_children():
            widget.destroy()

        color = "#27ae60" if prediction_label == "PLACED" else "#e74c3c"

        tk.Label(
            self.result_card, text="PLACEMENT RESULT", font=("Segoe UI", 12, "bold"),
            bg=CARD_BG, fg=ACCENT,
        ).pack(pady=(20, 5))

        tk.Label(
            self.result_card, text=f"Student ID: {student_id}",
            font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_MUTED,
        ).pack(pady=(0, 10))

        tk.Label(
            self.result_card, text=prediction_label, font=("Segoe UI", 22, "bold"),
            bg=CARD_BG, fg=color,
        ).pack(pady=5)

        tk.Label(
            self.result_card, text=f"Probability of Placement: {probability_placed:.1f}%",
            font=("Segoe UI", 10, "bold"), bg=CARD_BG, fg=ACCENT,
        ).pack(pady=(0, 10))

        tk.Label(
            self.result_card,
            text="This is an academic ML prediction and does not\nguarantee actual employment or placement.",
            font=("Segoe UI", 8, "italic"), bg=CARD_BG, fg=TEXT_MUTED, justify="center",
        ).pack(pady=(0, 10))

        suggestions = self._build_suggestions(cleaned)
        if suggestions:
            tk.Label(
                self.result_card, text="Suggestions:", font=("Segoe UI", 9, "bold"),
                bg=CARD_BG, fg=ACCENT,
            ).pack(pady=(5, 2), anchor="center")
            for s in suggestions:
                tk.Label(
                    self.result_card, text=f"- {s}", font=("Segoe UI", 8),
                    bg=CARD_BG, fg=TEXT_MUTED, wraplength=240, justify="left",
                ).pack(pady=1, anchor="w", padx=15)

    def _build_suggestions(self, cleaned):
        """
        General, non-guaranteeing improvement suggestions based on simple
        thresholds. These are educational nudges, not promises.
        """
        suggestions = []

        if cleaned["coding"] < 60:
            suggestions.append("Consider improving your coding practice.")
        if cleaned["internships"] == 0:
            suggestions.append("Consider gaining practical internship experience.")
        if cleaned["projects"] < 2:
            suggestions.append("Consider completing additional practical projects.")
        if cleaned["communication"] < 55:
            suggestions.append("Consider practicing communication and interview skills.")
        if cleaned["aptitude"] < 55:
            suggestions.append("Consider practicing aptitude questions.")
        if cleaned["technical"] < 55:
            suggestions.append("Consider preparing more for technical interviews.")
        if cleaned["backlogs"] > 0:
            suggestions.append("Consider clearing pending academic backlogs.")
        if cleaned["attendance"] < 70:
            suggestions.append("Consider improving class attendance.")

        return suggestions
