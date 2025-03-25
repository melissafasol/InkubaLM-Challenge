import csv
import os
from datetime import datetime

# Set your Google Drive output path here
OUTPUT_DIR = "/content/drive/MyDrive/InkubaLM/outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_log.csv")

def log_experiment(entry, log_file=LOG_FILE):
    """Appends an experiment run to the Google Drive CSV log."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)
