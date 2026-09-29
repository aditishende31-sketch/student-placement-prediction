"""
main_window.py
----------------
The main Tkinter application window. Provides left-side navigation between
Dashboard, Predict Placement, Prediction History, Model Performance, and
Feature Importance / About.
"""

import tkinter as tk

from gui.dashboard import DashboardFrame
from gui.prediction_form import PredictionFormFrame
from gui.history import HistoryFrame
from gui.performance import PerformanceFrame
from gui.feature_importance import FeatureImportanceFrame

BG = "#f4f6f9"
NAV_BG = "#2c3e50"
NAV_FG = "#ecf0f1"
NAV_ACTIVE_BG = "#34495e"
ACCENT = "#2980b9"


class MainWindow:
    def __init__(self, root, model, scaler, metadata):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.metadata = metadata

        self.root.title("Student Placement Predictor")
        self.root.geometry("1050x680")
        self.root.minsize(950, 600)
        self.root.configure(bg=BG)

        self.nav_buttons = {}
        self.frames = {}

        self._build_layout()
        self._show_frame("Dashboard")

    def _build_layout(self):
        nav_frame = tk.Frame(self.root, bg=NAV_BG, width=210)
        nav_frame.pack(side="left", fill="y")
        nav_frame.pack_propagate(False)

        tk.Label(
            nav_frame, text="🎓 Placement\nPredictor", font=("Segoe UI", 14, "bold"),
            bg=NAV_BG, fg="white", justify="left", anchor="w",
        ).pack(fill="x", padx=20, pady=(25, 30))

        nav_items = [
            "Dashboard",
            "Predict Placement",
            "Prediction History",
            "Model Performance",
            "Feature Importance / About",
        ]

        for item in nav_items:
            btn = tk.Button(
                nav_frame, text=item, font=("Segoe UI", 10), anchor="w",
                bg=NAV_BG, fg=NAV_FG, activebackground=NAV_ACTIVE_BG,
                activeforeground="white", relief="flat", bd=0, padx=20, pady=12,
                command=lambda name=item: self._show_frame(name),
            )
            btn.pack(fill="x")
            self.nav_buttons[item] = btn

        self.content_area = tk.Frame(self.root, bg=BG)
        self.content_area.pack(side="left", fill="both", expand=True)

        # Build all frames up-front (data is already loaded/trained)
        self.frames["Dashboard"] = DashboardFrame(self.content_area, self.metadata)
        self.frames["Predict Placement"] = PredictionFormFrame(
            self.content_area, self.model, self.scaler,
            on_prediction_saved=self._on_prediction_saved,
        )
        self.frames["Prediction History"] = HistoryFrame(self.content_area)
        self.frames["Model Performance"] = PerformanceFrame(self.content_area, self.metadata)
        self.frames["Feature Importance / About"] = FeatureImportanceFrame(
            self.content_area, self.metadata
        )

        for frame in self.frames.values():
            frame.place(x=0, y=0, relwidth=1, relheight=1)

    def _show_frame(self, name):
        for item, btn in self.nav_buttons.items():
            btn.configure(bg=ACCENT if item == name else NAV_BG)
        self.frames[name].tkraise()

        if name in ("Prediction History", "Dashboard"):
            self.frames[name].refresh()

    def _on_prediction_saved(self):
        # Keep history and dashboard in sync with the newest prediction
        self.frames["Prediction History"].refresh()
        self.frames["Dashboard"].refresh()
