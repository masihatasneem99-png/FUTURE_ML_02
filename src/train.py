"""
Loads the dataset, trains four classifiers for both category and priority,
prints a comparison table, and saves the two best-performing models to disk.
Run this first before evaluate.py or predict.py.

Outputs
    category_model.pkl   — best category classifier 
    priority_model.pkl   — best priority classifier 
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocess import preprocess_dataframe

# CONFIGURATION

DATASET_PATH = 'customer_support_tickets.csv'
CATEGORY_MODEL_PATH = 'model\category_model.pkl'
PRIORITY_MODEL_PATH = 'model\priority_model.pkl'
os.makedirs('model', exist_ok=True) # ensure model directory exists

TEST_SIZE = 0.20  # 80% train, 20% test
RANDOM_STATE = 42 # fixed seed — reproducible splits every run

TFIDF_PARAMS = dict(
    ngram_range = (1, 2),# unigrams + bigrams
    max_features = 5000, # vocabulary cap — keeps memory manageable
    sublinear_tf = True, # log(1+tf) dampens very frequent terms
    min_df = 2, # ignore terms that appear in only 1 document
)

#LOADING DATA

def load_data(path: str) -> pd.DataFrame:

    #Load CSV and run the full preprocessing pipeline.

    print("\n")
    print("LOADING DATA")
    print()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  If Dataset not found at: {path}"
            f"\n  Download it from:"
            f"\n  https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset"
            f"\n  and place the CSV in the same folder as train.py"
        )

    raw_df = pd.read_csv(path)
    print(f"\nRaw rows loaded: {len(raw_df):,}")
    print(f"Columns found:{raw_df.columns.tolist()}")

    df = preprocess_dataframe(raw_df)
    print(f"Rows after cleaning : {len(df):,}")

    print(f"\nCategory distribution:")
    for label, count in df['category'].value_counts().items():
        pct = count / len(df) * 100
        print(f"{label:<25} {count:>5}    {pct:.1f}%")

    print(f"\nPriority distribution:")
    for label, count in df['priority'].value_counts().items():
        pct = count / len(df) * 100
        print(f"{label:<10} {count:>5}    {pct:.1f}%")
    return df

#SPLITTING DATA

def split_data(df: pd.DataFrame):
    """
    Create a stratified 80/20 train/test split.
    Returns X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test
    """

    print("\n")
    print("SPLITTING DATA")
    print()

    X = df['clean_text']
    y_cat = df['category']
    y_pri = df['priority']

    X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = train_test_split(
        X, y_cat, y_pri,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y_cat    # preserves category proportions in both splits
    )

    print(f"\nTraining samples: {len(X_train):,} ({100 - TEST_SIZE*100:.0f}%)")
    print(f"Test samples: {len(X_test):,} ({TEST_SIZE*100:.0f}%)")
    print(f"Random seed: {RANDOM_STATE} (fixed for reproducibility)")

    return X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test

#DEFINING MODELS

def build_pipeline(classifier) -> Pipeline:
    #Wrapping a classifier inside a TF-IDF → classifier pipeline.
    return Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   classifier)
    ])

def get_classifiers() -> dict:
    #Return a dictionary of name → classifier instance.
    return {
        'Logistic Regression': LogisticRegression(
            max_iter     = 500,
            C            = 1.0,    # regularisation strength
            random_state = RANDOM_STATE
        ),
        'Naive Bayes (MNB)': MultinomialNB(
            alpha = 0.1            # Laplace smoothing — handles unseen words
        ),
        'Linear SVM': LinearSVC(
            C            = 1.0,
            max_iter     = 2000,
            random_state = RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators = 100,
            random_state = RANDOM_STATE,
            n_jobs       = -1      # use all CPU cores
        ),
    }

# TRAINING AND COMPARING ALL MODELS

def train_and_compare(X_train, X_test, y_train, y_test, task_name: str) -> tuple:
    """
    Trains all four classifiers on the given labels, prints a comparison table,
    and returns the best-performing pipeline and its name.
    """

    print(f"\n{task_name}")
    print(f"{'Model'}                      {'Accuracy'}   {'F1(weighted)'}")
    print()

    classifiers = get_classifiers()
    results = {}

    for name, clf in classifiers.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        preds= pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        results[name] = {
            'pipeline': pipeline,
            'preds': preds,
            'accuracy': accuracy,
            'f1': f1,
        }

        marker = ''
        print(f"{name:<25} {accuracy*100:>8.2f}% {f1*100:>12.2f}% {marker}")

    # Picking the model with the highest weighted F1
    best_name     = max(results, key=lambda k: results[k]['f1'])
    best_pipeline = results[best_name]['pipeline']
    print(f"\nBest: {best_name} (F1 = {results[best_name]['f1']*100:.2f}%)")

    return best_pipeline, best_name, results

#SAVING MODELS

def save_models(cat_pipeline, pri_pipeline) -> None:

    print("\n")
    print("SAVING MODELS")

    joblib.dump(cat_pipeline, CATEGORY_MODEL_PATH)
    joblib.dump(pri_pipeline, PRIORITY_MODEL_PATH)

    cat_size = os.path.getsize(CATEGORY_MODEL_PATH) / 1024
    pri_size = os.path.getsize(PRIORITY_MODEL_PATH) / 1024

    print(f"\n{CATEGORY_MODEL_PATH} ({cat_size:.1f} KB)")
    print(f"{PRIORITY_MODEL_PATH} ({pri_size:.1f} KB)")
    print(f"\nThese files are loaded by evaluate.py and predict.py")

#VISUALIZATION

MODEL_COLORS = {
    'Logistic Regression' : '#4C72B0',
    'Naive Bayes (MNB)'   : '#55A868',
    'Linear SVM'          : '#C44E52',
    'Random Forest'       : '#DD8452',
}

def plot_training_summary(df, cat_results, pri_results, best_cat_name, best_pri_name):

    model_names = list(cat_results.keys())
    short_names = ['LR', 'NB', 'SVM', 'RF']
    x           = np.arange(len(model_names))
    bar_width   = 0.35
    colors      = [MODEL_COLORS[m] for m in model_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Support Ticket Classifier — Training Summary',
                 fontsize=15, fontweight='bold', y=0.98)

    # ── Top-left: Dataset Distribution ───────────────────────────────────────
    ax1 = axes[0, 0]
    cat_counts = df['category'].value_counts()
    pri_counts = df['priority'].value_counts()
    cat_pos    = np.arange(len(cat_counts))
    pri_pos    = np.arange(len(pri_counts)) + len(cat_counts) + 0.8

    cat_bars = ax1.bar(cat_pos, cat_counts.values, width=0.6, edgecolor='white',
                       color=['#4C72B0','#55A868','#C44E52','#DD8452'][:len(cat_counts)])
    pri_bars = ax1.bar(pri_pos, pri_counts.values, width=0.6, edgecolor='white',
                       color=['#E74C3C','#F39C12','#27AE60'][:len(pri_counts)])

    for bar in list(cat_bars) + list(pri_bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(list(cat_pos) + list(pri_pos))
    ax1.set_xticklabels(list(cat_counts.index) + list(pri_counts.index),
                        rotation=20, ha='right', fontsize=9)
    ax1.set_title('Dataset Distribution', fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel('Number of Tickets')
    ax1.set_ylim(0, max(cat_counts.max(), pri_counts.max()) * 1.18)
    ax1.axvline(x=len(cat_counts)+0.3, color='grey', linestyle='--',
                linewidth=0.8, alpha=0.5)
    ax1.spines[['top','right']].set_visible(False)

    # ── Top-right: Accuracy Comparison ───────────────────────────────────────
    ax2 = axes[0, 1]
    cat_acc = [cat_results[m]['accuracy']*100 for m in model_names]
    pri_acc = [pri_results[m]['accuracy']*100 for m in model_names]

    b1 = ax2.bar(x - bar_width/2, cat_acc, width=bar_width, label='Category',
                 color=colors, alpha=0.9, edgecolor='white')
    b2 = ax2.bar(x + bar_width/2, pri_acc, width=bar_width, label='Priority',
                 color=colors, alpha=0.5, edgecolor='white', hatch='//')

    for bar in list(b1) + list(b2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.set_title('Accuracy by Model', fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.spines[['top','right']].set_visible(False)

    # ── Bottom-left: F1 Score Comparison ─────────────────────────────────────
    ax3 = axes[1, 0]
    cat_f1 = [cat_results[m]['f1']*100 for m in model_names]
    pri_f1 = [pri_results[m]['f1']*100 for m in model_names]

    b3 = ax3.bar(x - bar_width/2, cat_f1, width=bar_width, label='Category',
                 color=colors, alpha=0.9, edgecolor='white')
    b4 = ax3.bar(x + bar_width/2, pri_f1, width=bar_width, label='Priority',
                 color=colors, alpha=0.5, edgecolor='white', hatch='//')

    for bar in list(b3) + list(b4):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, fontsize=10)
    ax3.set_title('F1 Score (Weighted) by Model', fontsize=11, fontweight='bold', pad=10)
    ax3.set_ylabel('F1 Score (%)')
    ax3.set_ylim(0, 115)
    ax3.legend(fontsize=9)
    ax3.spines[['top','right']].set_visible(False)

    # ── Bottom-right: Best Model Highlight ───────────────────────────────────────
    ax4 = axes[1, 1]
    metrics = ['Accuracy', 'F1 Score']
    cat_scores = [cat_results[best_cat_name]['accuracy']*100,
                  cat_results[best_cat_name]['f1']*100]
    pri_scores = [pri_results[best_pri_name]['accuracy']*100,
                  pri_results[best_pri_name]['f1']*100]
    x4 = np.arange(len(metrics))

    b5 = ax4.bar(x4 - bar_width/2, cat_scores, width=bar_width,
             label=f'Category ({best_cat_name})',
             color='#4C72B0', alpha=0.85, edgecolor='white')
    b6 = ax4.bar(x4 + bar_width/2, pri_scores, width=bar_width,
                 label=f'Priority ({best_pri_name})',
                 color='#C44E52', alpha=0.85, edgecolor='white')

    for bar in list(b5) + list(b6):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{bar.get_height():.2f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bars in [b5, b6]:
        tallest = max(bars, key=lambda b: b.get_height())
        ax4.text(tallest.get_x() + tallest.get_width()/2,
                tallest.get_height() + 5.5, '',
                ha='center', fontsize=12, color='gold')

    ax4.set_xticks(x4)
    ax4.set_xticklabels(metrics, fontsize=11)
    ax4.set_title('Best Model Performance', fontsize=11, fontweight='bold', pad=10)
    ax4.set_ylabel('Score (%)')
    ax4.set_ylim(0, 130)                      # taller y-axis — bars stay at ~25%
    ax4.legend(
               fontsize=8,
               loc='upper center',                   # centred above the bars
               bbox_to_anchor=(0.5, 1.0),            # anchored just inside top edge
               bbox_transform=ax4.transAxes,
               frameon=True,
               framealpha=0.9,
               edgecolor='lightgrey',
            ncol=1
            )
    ax4.spines[['top','right']].set_visible(False)

    # ── Shared legend: short name → full model name ───────────────────────────
    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS[m], label=f'{s} = {m}')
        for m, s in zip(model_names, short_names)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/training_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nChart saved → output/training_summary.png")

# MAIN

def main():

    print("\n")
    print("SUPPORT TICKET CLASSIFIER — TRAINING PIPELINE")
    print(0)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Test size: {TEST_SIZE*100:.0f}%  |  Seed: {RANDOM_STATE}")

    #Load and preprocess data
    df = load_data(DATASET_PATH)

    #Split into train / test
    X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = split_data(df)

    #Train and compare — Category
    print("\n")
    print("TRAINING MODELS")
    print()

    best_cat_pipeline, best_cat_name, cat_results = train_and_compare(
        X_train, X_test, ycat_train, ycat_test,
        task_name="CATEGORY CLASSIFICATION"
    )
 
    best_pri_pipeline, best_pri_name, pri_results = train_and_compare(
        X_train, X_test, ypri_train, ypri_test,
        task_name="PRIORITY CLASSIFICATION"
    )
    #Save best models
    save_models(best_cat_pipeline, best_pri_pipeline)

    plot_training_summary(df, cat_results, pri_results, best_cat_name, best_pri_name)

    # Summary
    print("\n")
    print("TRAINING COMPLETE")
    print(f"\nCategory model : {best_cat_name}")
    print(f"Priority model : {best_pri_name}")
    print()

if __name__ == '__main__':
    main()