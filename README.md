# 🎫 Customer Support Ticket Classification System

An end-to-end Machine Learning system that automatically **classifies** customer support tickets into categories and assigns a **priority level** using NLP and classical ML.

---

## What It Does

Support teams receive hundreds of tickets daily. Manually triaging them is slow, inconsistent, and delays resolution. This system reads the raw ticket text and instantly predicts:

- **Category** — Billing · Technical · Account · General
- **Priority** — High · Medium · Low

This allows tickets to be routed to the right team automatically and urgent issues to surface immediately.

---

## Project Structure

```
CUSTOMER_SUPPORT_TICKET_CLASSIFICATION_SYSTEM/
│
├── src/
│   ├── app.py                          # Streamlit web application (2-page UI)
│   ├── train.py                        # Data loading, model training, saving
│   ├── evaluate.py                     # Classification reports, confusion matrices, CV
│   ├── predict.py                      # Load models and classify new tickets
│   └── preprocess.py                   # Text cleaning and preprocessing utilities
│
├── model/
│   ├── category_model.pkl              # Trained category classifier pipeline
│   └── priority_model.pkl              # Trained priority classifier pipeline
│
├── output/
│   ├── training_summary.png            # Model comparison chart (from train.py)
│   ├── confusion_matrix_category.png   # Category confusion matrix heatmap
│   ├── confusion_matrix_priority.png   # Priority confusion matrix heatmap
│   └── evaluation_summary.png         # Combined evaluation chart
│
├── customer_support_tickets.csv        # Dataset (Kaggle)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/masihatasneem99-png/FUTURE_ML_01.git
cd CUSTOMER_SUPPORT_TICKET_CLASSIFICATION_SYSTEM
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset from Kaggle:
[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

Place `customer_support_tickets.csv` in the project root.

### 5. Train the models

```bash
python src/train.py
```

This produces:
- `model/category_model.pkl`
- `model/priority_model.pkl`
- `output/training_summary.png`

### 6. Evaluate the models (optional)

```bash
python src/evaluate.py
```

This produces:
- `output/confusion_matrix_category.png`
- `output/confusion_matrix_priority.png`
- `output/evaluation_summary.png`

### 7. Run predictions from terminal

```bash
python src/predict.py
```

### 8. Launch the web app

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501`

---

## ML Pipeline

```
Raw ticket text
      │
      ▼
 Preprocessing
  • Lowercase
  • Normalise prices  ($149 → PRICE)
  • Normalise numbers (500  → NUM)
  • Remove punctuation
  • Remove stopwords
  • Filter short tokens (len ≤ 2)
      │
      ▼
 TF-IDF Vectorisation
  • ngram_range  : (1, 2)   — unigrams + bigrams
  • max_features : 5,000
  • sublinear_tf : True
  • min_df       : 2
      │
      ▼
 Two independent classifiers
  ┌─────────────────┐    ┌─────────────────┐
  │ Category Model  │    │ Priority Model  │
  │ Logistic Reg.   │    │ Logistic Reg.   │
  └────────┬────────┘    └────────┬────────┘
           │                      │
           ▼                      ▼
      Category              Priority
  (Billing / Technical   (High / Medium / Low)
   Account / General)
```

---

## Models Compared

| Model               | Category F1 | Priority F1 | Selected |
|---------------------|:-----------:|:-----------:|:--------:|
| Logistic Regression | Best        | Best        | ✓        |
| Linear SVM          | ~1% lower   | ~1% lower   |          |
| Naive Bayes (MNB)   | ~3% lower   | ~2% lower   |          |
| Random Forest       | ~4% lower   | ~3% lower   |          |

Logistic Regression is selected as the best model — fastest to train, most interpretable, and highest weighted F1 across both tasks.

---

## Evaluation

Models are evaluated using:

- **Accuracy** — overall correct predictions
- **Precision** — of predicted positives, how many were correct
- **Recall** — of actual positives, how many were found
- **F1 Score (weighted)** — harmonic mean of precision and recall, weighted by class size
- **Confusion Matrix** — per-class breakdown of correct and incorrect predictions
- **5-Fold Cross-Validation** — mean ± std F1 across 5 stratified folds

---

## Web Application

The Streamlit app has two pages:

### 🎫 Classify
- Text area to type or paste any support ticket
- 8 quick-example buttons to pre-fill common ticket types
- Instant result card showing predicted category and priority with colour-coded badges
- Live history table of the last 12 tickets classified

### 📊 Analytics
- Session stats — total classified, urgent count, top category, top priority
- Priority distribution bar chart
- Category distribution pie chart
- Priority × Category heatmap
- Full session history table with export to CSV

---

## File Descriptions

| File | Purpose |
|------|---------|
| `src/preprocess.py` | `clean_text()` and `preprocess_dataframe()` — shared by all other scripts |
| `src/train.py` | Loads CSV, relabels data, trains 4 models, saves best two as `.pkl` files, generates training chart |
| `src/evaluate.py` | Loads saved models, prints classification reports, saves confusion matrix PNGs, runs cross-validation |
| `src/predict.py` | Loads saved models, classifies 10 example tickets, prints results table, provides interactive prompt |
| `src/app.py` | Streamlit web UI — Classify page and Analytics dashboard |

---

## Dependencies

```
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
streamlit
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Dataset

**Source:** [Customer Support Ticket Dataset — Kaggle (suraj520)](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

| Property | Value |
|----------|-------|
| Total rows | ~8,469 |
| Train split | 80% |
| Test split | 20% |
| Split strategy | Stratified by category |

Key columns used:

| Dataset column | Used as |
|----------------|---------|
| `Ticket Subject` | Combined with description as model input text |
| `Ticket Description` | Primary ticket body text |
| `Ticket Type` | Category label (relabelled by keyword rules) |
| `Ticket Priority` | Priority label (relabelled by keyword rules) |

> **Note:** The original Kaggle dataset assigns categories and priorities somewhat randomly with respect to ticket text. This project applies keyword-based relabelling in `train.py` so that labels reflect the actual ticket content, producing meaningful and accurate predictions.

---

## .gitignore

```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
```

## Author

**Masiha Tasneem**
- GitHub: https://github.com/masihatasneem99-png
- LinkedIn: www.linkedin.com/in/masihatasneem

---

## License

This project is for educational and portfolio purposes.