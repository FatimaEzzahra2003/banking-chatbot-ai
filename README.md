# Banking Intent Chatbot

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-FFD21E?style=flat&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-412991?style=flat&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

Banking assistant combining intent classification (Sentence-BERT + SVM) with contextual response generation (GPT-4o-mini), deployed through a Flask web interface.

## Overview

The chatbot detects the customer's intent among 77 banking-related categories (card issues, refunds, ATM support, payments, etc.), then generates a natural, context-aware response aligned with that intent using an LLM. The intent classification is handled by a dedicated ML model rather than the LLM alone, which keeps responses grounded and predictable in a banking context.

## Pipeline

**Raw text** → **Cleaning & lemmatization (spaCy)** → **Sentence embeddings (Sentence-BERT)** → **Intent classification (SVM)** → **Response generation (GPT-4o-mini)**

## Approach

1. **Preprocessing** — text cleaning, data augmentation, lemmatization and syntactic analysis (spaCy)
2. **Representation** — sentence embeddings generated with a Sentence-BERT model (MiniLM, 384-dim)
3. **Training** — SVM classifier (RBF kernel) trained on the embeddings to predict intent across 77 classes
4. **Evaluation** — performance measured on a held-out test set with a full classification report and confusion matrix
5. **Serving** — Flask app: user message → intent prediction → GPT-4o-mini generates the final response conditioned on that intent

## Results

- **Accuracy**: 86.8%
- **F1-score (macro)**: 86.9%
- **F1-score (weighted)**: 86.9%
- Dataset: ~10,000 labeled utterances across 77 banking intents

## Tech stack

- **NLP preprocessing**: spaCy
- **Embeddings**: Sentence-BERT (sentence-transformers)
- **Classification**: scikit-learn (SVM)
- **Response generation**: OpenAI GPT-4o-mini
- **Web app**: Flask

## Project structure

```
banking-chatbot-ai/
├── notebooks/
│   ├── 01_Preprocessing.ipynb      # Cleaning, augmentation, lemmatization
│   ├── 02_Representation.ipynb     # Sentence-BERT embeddings
│   └── 03_Training.ipynb           # SVM training & evaluation
├── templates/
│   └── index.html                  # Chat web interface
├── app.py                          # Flask app: intent prediction + response generation
└── data/                           # Raw, processed and augmented datasets
```

> Note: the trained model files (Sentence-BERT weights, SVM model) and embeddings are not included in this repository due to their size. See "How to run" below to regenerate them.

## How to run

1. Install dependencies: `pip install flask sentence-transformers scikit-learn spacy openai joblib`
2. Run the notebooks in order (`01` → `02` → `03`) to preprocess the data, generate embeddings, and train the intent classifier
3. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your-key-here`
4. Run the app: `python app.py`

## Author

**Fatima Ezzahra Bououdi** — Data Analyst | Data Scientist
[LinkedIn](https://www.linkedin.com/in/fatima-ezzahra-bououdi-5b9615240) · [Portfolio](https://fatima-ezzahra-bououdi.vercel.app/) · fatimaezzahrabououdi@gmail.com
