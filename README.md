
# **InkubaLM: Fine-Tuning a Lightweight African Language Model** 

Welcome! This repository documents my work for the Lelapa AI Buzuzu-Mavi challenge, where I fine-tuned the InkubaLM model using efficient low-resource techniques. My goal: keep the model small, but smart.

Notebooks:
- [Data Exploration](notebooks/01_implementation.ipynb)
- [Preprocessing](notebooks/02_model_evaluation.ipynb)
- [Fine-Tuning](notebooks/03_plots.ipynb)

## Baseline Performance
Before fine-tuning, I evaluated the pretrained model across all tasks and languages. Hausa performance was particularly weak on translation and inference.

📊 Baseline Zindi Score: 0.1824
(Zindi score = average of F1/ChrF scores across all tasks)

## Fine-Tuning Strategy: QLoRA = LoRA + 4-bit Quantization
To fine-tune efficiently on limited hardware, I combined:
1. LoRA Adapters: Freeze the base model, train only small injected matrices.
2. 4-bit Quantization: Reduce memory by compressing weights from 32-bit → 4-bit.

**Result**: Dramatically reduced memory requirements with minimal performance loss.

### Challenge: Token Imbalance

Multitask training uses token-level loss. Since translation outputs are longer than sentiment/inference labels, MT dominated training loss.

![Token Imbalance Visualization](https://github.com/user-attachments/assets/1999e7c4-bfb0-4054-aaf0-884715f6f900)

*Figure: Token length distribution before and after balancing.*

### Task Output Lengths
- Classification (sentiment/XNLI): ~1–3 tokens (e.g., "positive", "neutral")
- Translation: ~15–30 tokens
(e.g., "I hope you're having a good day so far" → "Natumai unakuwa na siku njema hadi sasa" in Swahili)

## Solution: Token Balancing
I padded short-output examples (sentiment/XNLI) to ~18 tokens to ensure each task contributed an equal number of tokens to the loss.

### New Zindi Score: 0.35 🎉
Biggest gain: translation & inference improved significantly.
Remaining issue: sentiment classification still underperformed.

## Model Evaluation with SHAP: Token-Level Insights
I used SHAP (SHapley Additive exPlanations) to understand which tokens influenced sentiment predictions. It helped highlight issues like poor tokenization (e.g., splitting Chanya into “ch” + “anya”) and cross-language artifacts (like English words in Swahili predictions). After QLoRA fine-tuning, the model showed more balanced token contributions—but still revealed areas for improvement in data quality and preprocessing.
![image](https://github.com/user-attachments/assets/0217d66b-394e-4588-9f34-461d95fa2dd5)


