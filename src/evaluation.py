import csv
from collections import Counter
import numpy as np

def get_char_ngrams(sentence, n):
    sentence = sentence.replace(" ", "")
    return [sentence[i: i + n] for i in range(len(sentence) - n + 1)]

def precision_recall(reference, hypothesis, n):
    ref_ngrams = Counter(get_char_ngrams(reference, n))
    hyp_ngrams = Counter(get_char_ngrams(hypothesis, n))

    common = ref_ngrams & hyp_ngrams
    tp = sum(common.values())

    precision = tp / max(len(hyp_ngrams), 1)
    recall = tp / max(len(ref_ngrams), 1)

    return precision, recall

def f_score(precision, recall, beta=2):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def chrF(reference, hypothesis, max_n=6, beta=2):
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        p, r = precision_recall(reference, hypothesis, n)
        precisions.append(p)
        recalls.append(r)
    avg_p = sum(precisions) / max_n
    avg_r = sum(recalls) / max_n
    return f_score(avg_p, avg_r, beta)


def evaluate_mt_hausa_only(csv_file_path):
    chrfs_scores = []

    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Only evaluate Hausa MT entries
            if "mt" in row["ID"] and "eng-hau" in row["Langs"]:
                ref = row["Targets"]
                pred = row["Response"]
                score = chrF(ref, pred)
                chrfs_scores.append(score)

    avg_chrfs = np.mean(chrfs_scores)
    return round(avg_chrfs, 4), len(chrfs_scores)