import csv
import os
from datetime import datetime

LOG_FILE = "/content/drive/MyDrive/InkubaLM/outputs/experiment_log.csv"

def next_run_id(log_file=LOG_FILE):
    if not os.path.exists(log_file):
        return "run_001"
    with open(log_file) as f:
        lines = f.readlines()
        return f"run_{len(lines)}"

def log_experiment_auto(
    trainer, 
    train_output, 
    balanced=True, 
    repetition_factor=11, 
    prompt_variant="default", 
    notes="", 
    task_metrics=None,
    lb_score=None
):
    model_name = trainer.model.config._name_or_path
    entry = {
    "run_id": next_run_id(),
    "date": datetime.now().strftime("%Y-%m-%d"),
    "model_name": model_name,
    "prompt_variant": prompt_variant,
    "balanced": balanced,
    "repetition_factor": repetition_factor,
    "epochs": trainer.args.num_train_epochs,
    "train_loss": round(train_output.training_loss, 3),
    "sentiment_f1": task_metrics.get("sentiment_f1") if task_metrics else None,
    "nli_f1": task_metrics.get("nli_f1") if task_metrics else None,
    "mt_chrf": task_metrics.get("mt_chrf") if task_metrics else None,
    "zindi_score": task_metrics.get("zindi_score") if task_metrics else lb_score,
    "notes": notes
}


    # Save to CSV
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)