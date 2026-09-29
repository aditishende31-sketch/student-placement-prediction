"""
validation.py
--------------
Validates the 13 prediction form inputs so the application never crashes
on bad input and always shows a friendly, specific error message.
"""

# (field_key, display_name, min_value, max_value, must_be_integer)
FIELD_RULES = [
    ("cgpa", "CGPA", 0, 10, False),
    ("tenth", "10th Percentage", 0, 100, False),
    ("twelfth", "12th/Diploma Percentage", 0, 100, False),
    ("attendance", "Attendance Percentage", 0, 100, False),
    ("backlogs", "Number of Backlogs", 0, None, True),
    ("internships", "Number of Internships", 0, None, True),
    ("projects", "Number of Projects Completed", 0, None, True),
    ("certifications", "Number of Certifications", 0, None, True),
    ("coding", "Coding Score", 0, 100, False),
    ("communication", "Communication Score", 0, 100, False),
    ("aptitude", "Aptitude Score", 0, 100, False),
    ("technical", "Technical Interview Score", 0, 100, False),
    ("soft_skills", "Soft Skills Score", 0, 100, False),
]


def validate_form(raw_values):
    """
    raw_values: dict mapping field_key -> raw string from the GUI entry box.

    Returns (is_valid, cleaned_values_or_None, error_message_or_None).
    cleaned_values is a dict mapping field_key -> float, in the same order
    the ML model expects (caller is responsible for ordering into a list).
    """
    cleaned = {}

    for key, label, min_val, max_val, must_be_int in FIELD_RULES:
        raw = raw_values.get(key, "").strip()

        if raw == "":
            return False, None, f"Please enter a value for {label}."

        try:
            value = float(raw)
        except ValueError:
            return False, None, f"Please enter a valid number for {label}."

        if must_be_int and value != int(value):
            return False, None, f"{label} must be a whole number."

        if value < min_val:
            return False, None, f"{label} cannot be negative."

        if max_val is not None and value > max_val:
            return False, None, f"{label} must be between {min_val} and {max_val}."

        cleaned[key] = value

    return True, cleaned, None


def ordered_feature_list(cleaned_values):
    """
    Converts the cleaned_values dict into the exact ordered list of 13
    numbers the trained model expects (must match FEATURE_COLUMNS order
    in ml/data_generator.py).
    """
    order = [
        "cgpa", "tenth", "twelfth", "attendance", "backlogs",
        "internships", "projects", "certifications", "coding",
        "communication", "aptitude", "technical", "soft_skills",
    ]
    return [cleaned_values[key] for key in order]
