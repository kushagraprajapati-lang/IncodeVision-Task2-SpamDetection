<div align="center">

# 📩 AI Spam Message Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Task-Completed-success?style=for-the-badge)](https://github.com/)

An intelligent Machine Learning model built using **Natural Language Processing (NLP)** and **Multinomial Naive Bayes** to accurately identify and classify text messages as **SPAM** or **HAM (Legitimate)**.

---

</div>

## 📌 Overview
This repository contains **Task 02** developed for the **IncodeVision Artificial Intelligence Internship**. The primary objective is to implement text vectorization techniques and train a probabilistic classifier capable of real-time spam detection with high precision and confidence metrics.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.13
* **Libraries Used:**
  * `scikit-learn` (Machine Learning Algorithms & Pipelines)
  * `CountVectorizer` (Text Feature Extraction / Bag-of-Words)
  * `MultinomialNB` (Multinomial Naive Bayes Classifier)

---

## ⚙️ Architecture & Implementation

```mermaid
graph LR
    A[Raw Input Text] --> B[CountVectorizer]
    B --> C[Tokenization & Vectorization]
    C --> D[Multinomial Naive Bayes]
    D --> E[Prediction: HAM / SPAM]