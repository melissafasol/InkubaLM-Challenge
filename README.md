
## **Evaluating InkubaLM: Fine-Tuning a Lightweight African Language Model** 

Welcome! This repository documents my work for the Lelapa AI Buzuzu-Mavi challenge, where I fine-tuned the InkubaLM model on three multilingual NLP tasks using efficient low-resource techniques. My goal: keep the model small, but smart.

## 🚀 Overview
**InkubaLM** is an autoregressive language model trained on five African languages:IsiZulu, Yoruba, Hausa, Swahili, and IsiXhosa.  In this challenge, I fine-tuned it for:

1. Sentiment Analysis — classify text as positive, negative, or neutral.

2. African XNLI — determine whether a hypothesis follows from a premise.

3. Machine Translation (MT) — translate from English into Swahili or Hausa.


## 📊 Baseline Performance
Before fine-tuning, I evaluated the pretrained model across all tasks and languages. Hausa performance was particularly weak on translation and inference.

Baseline Zindi Score: 0.1824
(Zindi score = average of F1/ChrF scores across all tasks)

## 🛠️ Fine-Tuning Strategy
### ✅ QLoRA = LoRA + 4-bit Quantization
To fine-tune efficiently on limited hardware, I combined:
1. LoRA Adapters: Freeze the base model, train only small injected matrices.
2. 4-bit Quantization: Reduce memory by compressing weights from 32-bit → 4-bit.

### Result: Dramatically reduced memory requirements with minimal performance loss.

## ⚠️ Challenge: Token Imbalance
Multitask training uses token-level loss. Since translation outputs are longer than sentiment/inference labels, MT dominated training loss.

### Task Output Lengths
- Classification (sentiment/XNLI): ~1–3 tokens (e.g., "positive", "neutral")
- Translation: ~15–30 tokens
(e.g., "I hope you're having a good day so far" → "Natumai unakuwa na siku njema hadi sasa" in Swahili)

### 🧪 Solution: Token Balancing
I repeated or padded short-output examples (sentiment/XNLI) to ~18 tokens to ensure each task contributed an equal number of tokens to the loss.

## ✅ Post-Fine-Tuning Performance
After applying QLoRA and token balancing:
### New Zindi Score: 0.35 🎉
Biggest gain: translation & inference improved significantly.
Remaining issue: sentiment classification still underperformed.

## 🔍 SHAP Analysis: Token-Level Insights
To interpret model behavior, I used SHAP (SHapley Additive exPlanations):
SHAP assigns each token a value indicating how much it influenced the model’s prediction.
