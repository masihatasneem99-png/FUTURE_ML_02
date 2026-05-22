"""
evaluate.py
-----------
Loads saved models and the dataset, runs evaluation on the held-out test set,
prints classification reports, runs cross-validation, and saves clean charts.

Run after train.py has produced category_model.pkl and priority_model.pkl.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from preprocess import preprocess_dataframe

warnings.filterwarnings('ignore')

# CONFIGURATION  — must match the values used in train.py exactly
DATASET_PATH        = 'customer_support_tickets.csv'
CATEGORY_MODEL_PATH = 'model/category_model.pkl'
PRIORITY_MODEL_PATH = 'model/priority_model.pkl'
OUTPUT_DIR          = 'output'

TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Design tokens ─────────────────────────────────────────────────────────────
FONT_FAMILY   = 'DejaVu Sans'
COLOR_BG      = '#FFFFFF'
COLOR_TEXT    = '#111111'
COLOR_SUBTEXT = '#555555'
COLOR_GRID    = '#E8E8E8'
COLOR_CAT     = '#1B4F9B'   # deep royal blue
COLOR_PRI     = '#B22020'   # deep crimson red
COLOR_ACCENT  = '#F3F3F3'

# High-contrast colourmap — near white → deep saturated colour
CMAP_CAT = LinearSegmentedColormap.from_list(
    'cat_blue', ['#F0F4FF', '#6494E8', '#1B4F9B', '#0A2860']
)
CMAP_PRI = LinearSegmentedColormap.from_list(
    'pri_red', ['#FFF0F0', '#E87878', '#B22020', '#660A0A']
)

plt.rcParams.update({
    'font.family'      : FONT_FAMILY,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : False,
    'figure.facecolor' : COLOR_BG,
    'axes.facecolor'   : COLOR_BG,
})

#LOADING DATA AND RECREATING THE SAME SPLIT AS TRAIN.PY

def load_and_split():
    """
    Load the CSV, preprocess it, and recreate the identical 80/20 split
    that train.py used. Same random_state + same stratify column = same
    test rows every time, so evaluation is always on unseen data.
    """

    print("LOADING DATA")
    print()

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\n  Dataset not found: {DATASET_PATH}"
            f"\n  Place the CSV in the same folder as evaluate.py"
        )

    df = preprocess_dataframe(pd.read_csv(DATASET_PATH))
    print(f"Rows loaded : {len(df):,}")

    X = df['clean_text']
    y_cat = df['category']
    y_pri = df['priority']

    _, X_test, _, ycat_test, _, ypri_test = train_test_split(
        X, y_cat, y_pri,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y_cat
    )

    print(f"  Test samples: {len(X_test):,}  (identical split to train.py)")
    return df, X_test, ycat_test, ypri_test, X, y_cat, y_pri

#LOADING MODELS

def load_models():
    """Load the two pipelines saved by train.py."""

    print("\n")
    print("LOADING MODELS")

    for path in [CATEGORY_MODEL_PATH, PRIORITY_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  Model not found: {path}"
                f"\n  Run train.py first to generate the model files."
            )

    cat_model = joblib.load(CATEGORY_MODEL_PATH)
    pri_model = joblib.load(PRIORITY_MODEL_PATH)

    print(f"\n{CATEGORY_MODEL_PATH}")
    print(f"{PRIORITY_MODEL_PATH}")

    return cat_model, pri_model

#CLASSIFICATION REPORTS

def print_classification_reports(cat_model, pri_model, X_test, ycat_test, ypri_test):
    """
    Predict on the test set and print full per-class reports for both tasks.
    Returns predictions and report dicts for use in charting.
    """

    print("\n")
    print("CLASSIFICATION REPORTS")

    cat_preds = cat_model.predict(X_test)
    pri_preds = pri_model.predict(X_test)

    cat_report = classification_report(ycat_test, cat_preds, output_dict=True, zero_division=0)
    pri_report = classification_report(ypri_test, pri_preds, output_dict=True, zero_division=0)

    def _print_report(title, true, preds, report_dict):
        print(f"\n{title}")
        classes = [k for k in report_dict if k not in
                   ('accuracy', 'macro avg', 'weighted avg')]
        col_w = max(len(c) for c in classes) + 2
        header = f"  {'Class':<{col_w}} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}"
        print(header)
        print("  " + "─" * (col_w + 40))
        for cls in classes:
            v = report_dict[cls]
            print(f"  {cls:<{col_w}} {v['precision']:>10.3f} "
                  f"{v['recall']:>8.3f} {v['f1-score']:>8.3f} "
                  f"{int(v['support']):>9}")
        print("  " + "─" * (col_w + 40))
        wa = report_dict['weighted avg']
        print(f"  {'Weighted Avg':<{col_w}} {wa['precision']:>10.3f} "
              f"{wa['recall']:>8.3f} {wa['f1-score']:>8.3f} "
              f"{int(wa['support']):>9}")
        overall_acc = (np.array(preds) == np.array(true)).mean()
        print(f"\n  Overall Accuracy : {overall_acc*100:.2f}%")

    _print_report("CATEGORY", ycat_test, cat_preds, cat_report)
    _print_report("PRIORITY", ypri_test, pri_preds, pri_report)

    return cat_preds, pri_preds, cat_report, pri_report

#CROSS-VALIDATION

def run_cross_validation(cat_model, pri_model, X_full, y_cat_full, y_pri_full):
    """
    Run 5-fold stratified cross-validation on the full dataset for both models.
    Prints each fold score plus mean ± std.
    """

    print("\n")
    print(f"{CV_FOLDS}-FOLD CROSS-VALIDATION")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}

    for label, model, y in [
        ('Category', cat_model, y_cat_full),
        ('Priority', pri_model, y_pri_full),
    ]:
        scores = cross_val_score(model, X_full, y,cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_results[label] = scores

        print(f"\n  {label}")
        fold_line = "  Folds  : " + "  |  ".join(f"Fold {i+1}: {s*100:.2f}%" for i, s in enumerate(scores))
        print(fold_line)
        print(f"Mean: {scores.mean()*100:.2f}%")
        print(f"Std: ± {scores.std()*100:.2f}%")
        print(f"Range: {scores.min()*100:.2f}% – {scores.max()*100:.2f}%")

    return cv_results

#CONFUSION MATRIX CHARTS

def _draw_confusion_matrix(ax, cm, labels, cmap, title, task_color):

    n = len(labels)

    # Row-normalise for percentage display (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)

    # ── Cell backgrounds ──────────────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal — deep solid task colour, min 60% opacity
                intensity  = 0.55 + cm_norm[i, j] * 0.45
                rgba       = list(plt.matplotlib.colors.to_rgba(task_color))
                rgba[3]    = intensity
                cell_color = tuple(rgba)
            else:
                # Off-diagonal — white to orange-red scale so errors pop
                err = cm_norm[i, j]
                if err < 0.01:
                    cell_color = '#FAFAFA'          # near-zero — almost white
                elif err < 0.10:
                    cell_color = '#FFE8CC'          # low error — pale amber
                elif err < 0.20:
                    cell_color = '#FFBA70'          # medium error — amber
                else:
                    cell_color = '#F07030'          # high error — deep orange

            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                color=cell_color, zorder=1
            ))

    # ── White grid lines between cells ───────────────────────────────────────
    for k in range(1, n):
        ax.axhline(k - 0.5, color='white', linewidth=2.0, zorder=2)
        ax.axvline(k - 0.5, color='white', linewidth=2.0, zorder=2)

    # ── Cell text — count (bold large) + percentage (small) ──────────────────
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100

            if i == j:
                txt_color = 'white'                # always white on dark diagonal
            elif cm_norm[i, j] >= 0.15:
                txt_color = '#1A1A1A'              # dark text on amber/orange
            else:
                txt_color = '#444444'              # medium grey on pale cells

            ax.text(j, i - 0.1, f'{count}',
                    ha='center', va='center',
                    fontsize=12, fontweight='800',
                    color=txt_color, zorder=3)
            ax.text(j, i + 0.22, f'{pct:.0f}%',
                    ha='center', va='center',
                    fontsize=8.5, color=txt_color,
                    alpha=0.9, zorder=3)

    # ── Bold border around diagonal cells ─────────────────────────────────────
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False,
            edgecolor='white',
            linewidth=3.0,
            zorder=4
        ))

    # ── Outer border ──────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.add_patch(plt.Rectangle(
        (-0.5, -0.5), n, n,
        fill=False,
        edgecolor='#CCCCCC',
        linewidth=1.0,
        zorder=5
    ))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9.5, rotation=30,
                       ha='right', color=COLOR_TEXT)
    ax.set_yticklabels(labels, fontsize=9.5, color=COLOR_TEXT)
    ax.set_xlabel('Predicted', fontsize=10,
                  color=COLOR_SUBTEXT, labelpad=10)
    ax.set_ylabel('Actual', fontsize=10,
                  color=COLOR_SUBTEXT, labelpad=10)
    ax.set_title(title, fontsize=12, fontweight='700',
                 color=COLOR_TEXT, pad=16)
    ax.tick_params(length=0)

    # ── Legend strip below matrix explaining colours ──────────────────────────
    legend_x    = -0.5
    legend_y    = n - 0.28
    legend_data = [
        ('#1B4F9B' if 'blue' in str(cmap.name) else '#B22020', 'Correct'),
        ('#FAFAFA',  'No errors'),
        ('#FFE8CC',  'Few errors'),
        ('#FFBA70',  'Some errors'),
        ('#F07030',  'Many errors'),
    ]
    for idx, (col, lbl) in enumerate(legend_data):
        ax.add_patch(plt.Rectangle(
            (legend_x + idx * 1.05, legend_y + 0.62),
            0.22, 0.22,
            color=col, zorder=5,
            clip_on=False,
            transform=ax.transData
        ))
        ax.text(legend_x + idx * 1.05 + 0.27, legend_y + 0.73,
                lbl, fontsize=6.5, va='center',
                color=COLOR_SUBTEXT, clip_on=False)

def save_confusion_matrices(cat_preds, pri_preds, ycat_test, ypri_test):
    """Save individual confusion matrix PNGs — wider figure to prevent clipping."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cat_labels = sorted(ycat_test.unique())
    pri_labels = sorted(ypri_test.unique())

    cm_cat = confusion_matrix(ycat_test, cat_preds, labels=cat_labels)
    cm_pri = confusion_matrix(ypri_test, pri_preds, labels=pri_labels)

    # Dynamically size figure by number of classes
    for cm, labels, cmap, color, title, fname in [
        (cm_cat, cat_labels, CMAP_CAT, COLOR_CAT,
         'Confusion Matrix — Category', 'confusion_matrix_category.png'),
        (cm_pri, pri_labels, CMAP_PRI, COLOR_PRI,
         'Confusion Matrix — Priority', 'confusion_matrix_priority.png'),
    ]:
        n        = len(labels)
        fig_size = max(7, n * 1.4)    # grows with number of classes
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        fig.patch.set_facecolor(COLOR_BG)
        _draw_confusion_matrix(ax, cm, labels, cmap, title, color)
        plt.tight_layout(pad=2.5)
        path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
        plt.close()
        print(f"Saved → {path}")

    return cm_cat, cm_pri, cat_labels, pri_labels

#EVALUATION SUMMARY CHART

def save_evaluation_summary(
    cat_report, pri_report,
    cm_cat, cm_pri,
    cat_labels, pri_labels,
    cv_results,
):
    """
    Redesigned summary — two rows:
        Row 1: Category F1 bars  |  Category confusion matrix
        Row 2: Priority F1 bars  |  Priority confusion matrix
    Each task gets its own row so nothing is cramped or mixed.
    """

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLOR_BG)

    gs = fig.add_gridspec(
        2, 2,
        width_ratios  = [1, 1.6],
        height_ratios = [1, 1],
        wspace=0.38, hspace=0.55,
        left=0.08, right=0.96,
        top=0.91,   bottom=0.07
    )

    ax_cat_bar = fig.add_subplot(gs[0, 0])
    ax_cat_cm  = fig.add_subplot(gs[0, 1])
    ax_pri_bar = fig.add_subplot(gs[1, 0])
    ax_pri_cm  = fig.add_subplot(gs[1, 1])

    def draw_f1_bars(ax, report, color, task_label):
        classes = sorted([k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')])
        f1_vals = [report[c]['f1-score'] * 100 for c in classes]
        y_pos   = np.arange(len(classes))

    # Background rows — strong alternating contrast
        for i in range(len(classes)):
            ax.axhspan(i - 0.48, i + 0.48,
                   color='#F0F0F0' if i % 2 == 0 else '#FAFAFA',
                   zorder=0)

        bars = ax.barh(y_pos, f1_vals,height=0.58,
                 color=color,
                 alpha=1.0,                       # full opacity — no more muted bars
                 edgecolor='white',
                 linewidth=0.8,
                 zorder=3
                )

    # Value labels with strong contrast
        for bar, val in zip(bars, f1_vals):
            ax.text(val + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%',
                va='center', ha='left',
                fontsize=8.5, fontweight='600',
                color=COLOR_TEXT)

    # Weighted average reference line — thick and labelled
        wa_f1 = report['weighted avg']['f1-score'] * 100
        ax.axvline(wa_f1, color=color, linewidth=1.8,
               linestyle='--', alpha=0.7, zorder=4)
        ax.text(wa_f1 + 0.4, len(classes) - 0.2,
            f'avg\n{wa_f1:.1f}%',
            fontsize=7.5, color=color,
            fontweight='600', va='top')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=9.5,
                       fontweight='500', color=COLOR_TEXT)
        ax.set_xlabel('F1 Score (%)', fontsize=9,
                  color=COLOR_SUBTEXT, labelpad=6)
        ax.set_title(f'{task_label} — Class F1',
                 fontsize=11, fontweight='700',
                 color=COLOR_TEXT, pad=10)
        ax.set_xlim(0, 115)
        ax.tick_params(axis='x', labelsize=8.5,
                   colors=COLOR_SUBTEXT, length=0)
        ax.tick_params(axis='y', length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(x=0, color='#BBBBBB', linewidth=1.0)

    # Draw F1 bar charts
    draw_f1_bars(ax_cat_bar, cat_report, COLOR_CAT, 'Category')
    draw_f1_bars(ax_pri_bar, pri_report, COLOR_PRI, 'Priority')

    # Draw confusion matrices
    _draw_confusion_matrix(
        ax_cat_cm, cm_cat, cat_labels,
        CMAP_CAT, 'Confusion Matrix — Category', COLOR_CAT
    )
    _draw_confusion_matrix(
        ax_pri_cm, cm_pri, pri_labels,
        CMAP_PRI, 'Confusion Matrix — Priority', COLOR_PRI
    )

    # CV scores as figure-level subtitle
    cat_cv_mean = cv_results['Category'].mean() * 100
    cat_cv_std  = cv_results['Category'].std()  * 100
    pri_cv_mean = cv_results['Priority'].mean()  * 100
    pri_cv_std  = cv_results['Priority'].std()   * 100

    fig.text(
        0.5, 0.955,
        'Evaluation Summary',
        ha='center', fontsize=14,
        fontweight='700', color=COLOR_TEXT
    )
    fig.text(
        0.5, 0.928,
        f'5-fold CV  →  Category: {cat_cv_mean:.1f}% ± {cat_cv_std:.1f}%'
        f'     Priority: {pri_cv_mean:.1f}% ± {pri_cv_std:.1f}%',
        ha='center', fontsize=9,
        color=COLOR_SUBTEXT, style='italic'
    )

    path = os.path.join(OUTPUT_DIR, 'evaluation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close()
    print(f"Saved → {path}")

# MAIN

def main():

    print("\n")
    print("SUPPORT TICKET CLASSIFIER — EVALUATION")

    # 1. Load data and recreate test split
    df, X_test, ycat_test, ypri_test, X_full, y_cat_full, y_pri_full = (load_and_split())

    # 2. Load saved models
    cat_model, pri_model = load_models()

    # 3. Classification reports
    cat_preds, pri_preds, cat_report, pri_report = print_classification_reports(cat_model, pri_model, X_test, ycat_test, ypri_test)

    # 4. Cross-validation
    cv_results = run_cross_validation(cat_model, pri_model, X_full, y_cat_full, y_pri_full)

    # 5. Individual confusion matrix PNGs
    print("\n")
    print("SAVING CONFUSION MATRICES")
    cm_cat, cm_pri, cat_labels, pri_labels = save_confusion_matrices(
        cat_preds, pri_preds, ycat_test, ypri_test
    )

    # 6. Combined evaluation summary chart
    print("\n")
    print("SAVING EVALUATION SUMMARY CHART")
    save_evaluation_summary(
        cat_report, pri_report,
        cm_cat, cm_pri,
        cat_labels, pri_labels,
        cv_results,
    )

    # Done
    print("\n")
    print("EVALUATION COMPLETE — FILES PRODUCED")
    print(f"\noutput/confusion_matrix_category.png")
    print(f"output/confusion_matrix_priority.png")
    print(f"output/evaluation_summary.png")
    print()


if __name__ == '__main__':
    main()