# 📊 LLM vs SSL: Text Classification with Limited Labeled Data

This repository supports a research project that compares Large Language Models (LLMs) and Semi-Supervised Learning (SSL) techniques for binary sentiment classification on Amazon product reviews using limited labeled data and large unlabeled datasets.

## 🧠 Project Motivation

As businesses increasingly rely on customer reviews to guide decisions, the cost of data labeling becomes a major barrier—especially for SMEs. This project investigates how modern LLMs (like BLAIR) compare to traditional SSL methods (like Co-Training) in accuracy, training time, and scalability, using real-world e-commerce data.

---

## 📚 Methods Overview

### 🔹 Semi-Supervised Learning (SSL)
We implement **Co-Training** using:
- TF-IDF vectorization
- Random Forest + SVM classifiers (via `mvlearn`)

### 🔹 Large Language Models (LLMs)
We evaluate **BLAIR** embeddings (a domain-specific RoBERTa-based model), combined with:
- A shallow neural network classifier
- Few-shot label settings (50–1000 labels)

---

## 📊 Experiments

| Dataset Size | Labeled Samples | Model             | Best F1 Score |
|--------------|------------------|-------------------|---------------|
| 5K, 10K, 15K | 50 to 1000       | BLAIR + NN        | 0.8442        |
|              |                  | TF-IDF + CoTrain  | 0.8432        |

- Metrics: Accuracy, F1 Score, Training & Inference Time
- Tooling: HuggingFace Transformers, Scikit-learn, Keras, PyTorch

---

## 📁 Folder Descriptions

- `data/`: cleaned datasets used in training
- `Code/`: Jupyter notebooks for training and evaluation
- `Research/`: Including research papers and resources
- `Figures/`: All the results and diagrams figures

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/llm-vs-ssl-text-classification.git
cd llm-vs-ssl-text-classification
pip install -r requirements.txt
