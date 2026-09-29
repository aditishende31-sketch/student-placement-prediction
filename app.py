import os
import sys
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.data_generator import DATA_PATH
from ml.train_model import train_and_select_best, load_metadata, load_model_and_scaler, artifacts_exist
from utils.database import init_db, DB_PATH


def run_first_time_setup():
   
    print("Checking application setup...")

    if not os.path.exists(DATA_PATH):
        print("  - No dataset found. It will be generated during training.")
    else:
        print("  - Dataset found.")

    if not artifacts_exist():
        print("  - No trained model found. Training models now (this may take a moment)...")
        metadata = train_and_select_best()
        print(f"  - Training complete. Best model: {metadata['best_model_name']}")
    else:
        print("  - Trained model found. Skipping retraining.")
        metadata = load_metadata()

    if not os.path.exists(DB_PATH):
        print("  - No database found. Creating a new one.")
    else:
        print("  - Database found.")
    init_db()

    print("Setup complete. Launching GUI...")
    return metadata


def main():
    try:
        metadata = run_first_time_setup()
        model, scaler = load_model_and_scaler()
    except Exception as exc:
        # Never let a setup failure crash silently - show it to the user.
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            f"The application could not complete first-time setup:\n\n{exc}\n\n"
            "Try deleting the 'data', 'models' and 'database' folders and "
            "running the app again.",
        )
        sys.exit(1)

    from gui.main_window import MainWindow

    root = tk.Tk()
    try:
        MainWindow(root, model, scaler, metadata)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("Application Error", f"An unexpected error occurred:\n\n{exc}")
        raise


if __name__ == "__main__":
    main()
