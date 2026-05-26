"""
app.py
Two-page Streamlit application for the Support Ticket Classifier.

Pages
    Classify   — single ticket classification with live result
    Analytics  — session dashboard with charts and breakdowns
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

import streamlit as st
from preprocess import clean_text

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS (Updated for Dynamic Light/Dark Mode Visibility)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main .block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── Page title ─────────────────────────────────────────── */
.page-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-color);
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.92rem;
    color: var(--text-color);
    opacity: 0.8;
    margin-bottom: 1.75rem;
    font-weight: 400;
}

/* ── Section label ──────────────────────────────────────── */
.sec-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.7;
    margin-bottom: 0.6rem;
    margin-top: 1.5rem;
}

/* ── Stat cards ─────────────────────────────────────────── */
.stat-card {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color, #DEDEDE);
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.stat-card-accent {
    background: #0A0A0A;
    border: 1px solid #0A0A0A;
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.stat-lbl {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.7; margin-bottom: 3px;
}
.stat-lbl-inv {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: #AAAAAA; margin-bottom: 3px;
}
.stat-val     { font-size:1.5rem; font-weight:700; color: var(--text-color); line-height:1.1; }
.stat-val-inv { font-size:1.5rem; font-weight:700; color:#FFFFFF; line-height:1.1; }
.stat-hint    { font-size:0.72rem; color: var(--text-color); opacity: 0.6; margin-top:2px; font-weight:500; }
.stat-hint-inv{ font-size:0.72rem; color:#888888; margin-top:2px; font-weight:500; }

/* ── Result card ────────────────────────────────────────── */
.result-card {
    background: var(--background-color);
    border: 1.5px solid var(--border-color, #CCCCCC);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.result-row {
    display: flex; align-items: center;
    gap: 0.75rem; margin-bottom: 0.9rem;
}
.result-row:last-child { margin-bottom: 0; }
.result-key {
    font-size: 0.72rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.8;
    width: 76px; flex-shrink: 0;
}
.result-val { font-size: 1rem; font-weight: 600; color: var(--text-color); }

/* ── Badges ─────────────────────────────────────────────── */
.badge {
    display: inline-block; font-size: 0.78rem; font-weight: 700;
    padding: 4px 13px; border-radius: 20px; letter-spacing: .02em;
}
.b-critical { background:#FFDDDD; color:#7A0000; }
.b-high     { background:#FFE8CC; color:#6B2D00; }
.b-medium   { background:#FFF5CC; color:#5A3D00; }
.b-low      { background:#D4F5E5; color:#004D2E; }
.b-cat      { background:#DDE5FF; color:#1A2980; }

/* ── Mono text ──────────────────────────────────────────── */
.mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; color: var(--text-color); line-height: 1.55;
}

/* ── History table ──────────────────────────────────────── */
.hist-tbl { width:100%; border-collapse:collapse; font-size:0.86rem; }
.hist-tbl th {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.8;
    padding: 0 10px 9px 0; border-bottom: 2px solid var(--border-color, #CCCCCC); text-align: left;
}
.hist-tbl td {
    padding: 9px 10px 9px 0; border-bottom: 1px solid var(--border-color, #E8E8E8);
    vertical-align: top; color: var(--text-color);
}
.hist-tbl tr:last-child td { border-bottom: none; }

/* ── Divider ────────────────────────────────────────────── */
.div { border:none; border-top:1.5px solid var(--border-color, #E0E0E0); margin:1.5rem 0; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--secondary-background-color); }
[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }

/* Hide Streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

HIGH_KEYWORDS = [
    'urgent', 'asap', 'immediately', 'critical', 'emergency',
    'blocked', 'cannot access', 'data loss', 'hacked', 'breach',
    'down', 'outage', 'production', 'demo', 'deadline', 'lawsuit',
    'legal', 'not working', 'broken', 'crashed', 'locked out',
]
CRITICAL_KEYWORDS = [
    'production down', 'security breach', 'data breach',
    'complete outage', 'all users affected', 'system down',
]
LOW_KEYWORDS = [
    'suggestion', 'feature request', 'wondering', 'curious',
    'nice to have', 'minor', 'when possible', 'feedback',
    'question', 'quick question', 'how do i', 'how to',
]

def smart_priority(text: str, model_priority: str) -> str:
    """Override model priority using keyword signals for clearer results."""
    t = text.lower()
    if any(kw in t for kw in CRITICAL_KEYWORDS):
        return 'Critical'
    if any(kw in t for kw in HIGH_KEYWORDS):
        return 'High'
    if any(kw in t for kw in LOW_KEYWORDS):
        return 'Low'
    return model_priority


def pri_badge(p: str) -> str:
    cls  = {'Critical':'b-critical','High':'b-high',
            'Medium':'b-medium','Low':'b-low'}.get(p,'b-medium')
    dot  = {'Critical':'🔴','High':'🟠',
            'Medium':'🟡','Low':'🟢'}.get(p,'')
    return f'<span class="badge {cls}">{dot}&nbsp;{p}</span>'


def cat_badge(c: str) -> str:
    return f'<span class="badge b-cat">{c}</span>'


PRI_COLOR = {
    'Critical': '#9B1C1C',
    'High':     '#C2570A',
    'Medium':   '#92620A',
    'Low':      '#065F46',
}
CAT_COLOR = ['#4C72B0','#55A868','#C44E52','#DD8452','#8172B2']


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    root = r'D:\Customer_Support_Ticket_Classification_System'

    candidates = [
        (os.path.join(root, 'category_model.pkl'),
         os.path.join(root, 'priority_model.pkl')),
        (os.path.join(root, 'model', 'category_model.pkl'),
         os.path.join(root, 'model', 'priority_model.pkl')),
        ('category_model.pkl', 'priority_model.pkl'),
        (os.path.join('model', 'category_model.pkl'),
         os.path.join('model', 'priority_model.pkl')),
    ]

    for cat_path, pri_path in candidates:
        if os.path.exists(cat_path) and os.path.exists(pri_path):
            return joblib.load(cat_path), joblib.load(pri_path), []

    return None, None, ['category_model.pkl', 'priority_model.pkl']


def classify_one(text: str, cat_model, pri_model) -> tuple:
    cleaned          = clean_text(text)
    category         = cat_model.predict([cleaned])[0]
    model_priority   = pri_model.predict([cleaned])[0]
    priority         = smart_priority(text, model_priority)
    return category, priority


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

for k, v in [('history', []), ('input_text', ''), ('n_classified', 0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

cat_model, pri_model, missing = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color: var(--text-color);"
        "margin-bottom:0.25rem'>🎫 Ticket Classifier</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:0.78rem;color: var(--text-color); opacity: 0.7; margin-bottom:1.25rem'>"
        "TF-IDF · Logistic Regression</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        ["🎫   Classify", "📊   Analytics"],
        label_visibility="collapsed",
    )

    st.markdown(
        "<hr style='border-color: var(--border-color, #CCCCCC); margin:1rem 0'>",
        unsafe_allow_html=True
    )

    # Model status
    if missing:
        st.error("Models not found.\nRun `python train.py` first.")
    else:
        st.markdown(
            "<div style='font-size:0.82rem;color:#004D2E;"
            "font-weight:600'>✓ Models loaded</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<hr style='border-color: var(--border-color, #CCCCCC); margin:1rem 0'>",
        unsafe_allow_html=True
    )

    n  = st.session_state.n_classified
    nh = sum(
        1 for h in st.session_state.history
        if h['priority'] in ('Critical', 'High')
    )
    st.markdown(f"""
    <div style='font-size:0.72rem;color: var(--text-color); opacity: 0.8; margin-bottom:6px;
         font-weight:700;letter-spacing:.07em;text-transform:uppercase'>
         Session</div>
    <div style='font-size:0.88rem;color: var(--text-color);line-height:2.1'>
        Classified &nbsp;<strong>{n}</strong><br>
        Urgent &nbsp;<strong style='color:#7A0000'>{nh}</strong>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────

if missing:
    st.error(
        f"**Model files not found:** `{'`, `'.join(missing)}`\n\n"
        "Run `python train.py` from your project folder, "
        "then refresh this page."
    )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLASSIFY
# ═════════════════════════════════════════════════════════════════════════════

EXAMPLES = {
    "Billing dispute"   : "I was charged $299 twice this month and need an immediate refund.",
    "Platform down"     : "The entire platform is down — our team cannot log in. Client demo in 30 minutes!",
    "Account hacked"    : "My account was compromised and the password was changed without my consent.",
    "Upload failing"    : "File uploads fail intermittently for files larger than 10MB.",
    "2FA broken"        : "Two-factor authentication is broken — the code never arrives and I'm locked out.",
    "Dark mode request" : "Quick suggestion — a dark mode toggle in the settings would be great.",
    "Billing question"  : "When does my billing cycle reset each month?",
    "Slack setup"       : "Can you walk me through setting up the Slack integration step by step?",
}

if page == "🎫   Classify":

    st.markdown('<div class="page-title">Classify a ticket</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Enter any support ticket and get an instant '
        'category and priority prediction.</div>',
        unsafe_allow_html=True
    )

    # ── Stats row ────────────────────────────────────────────────────────────
    total  = st.session_state.n_classified
    urgent = sum(
        1 for h in st.session_state.history
        if h['priority'] in ('Critical', 'High')
    )
    cats_n = len(set(h['category'] for h in st.session_state.history)) \
             if st.session_state.history else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, hint, accent in [
        (c1, "Classified",    total,  "this session",       False),
        (c2, "Urgent",        urgent, "Critical or High",   True),
        (c3, "Categories",    cats_n, "seen this session",  False),
        (c4, "Models active", 2,      "category + priority",False),
    ]:
        with col:
            if accent:
                st.markdown(f"""
                <div class="stat-card-accent">
                    <div class="stat-lbl-inv">{lbl}</div>
                    <div class="stat-val-inv">{val}</div>
                    <div class="stat-hint-inv">{hint}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-lbl">{lbl}</div>
                    <div class="stat-val">{val}</div>
                    <div class="stat-hint">{hint}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="sec-label">Ticket text</div>',
                    unsafe_allow_html=True)

        ticket_input = st.text_area(
            "ticket",
            label_visibility="collapsed",
            placeholder="Paste or type a support ticket here…",
            height=145,
            value=st.session_state.input_text,
            key="ta_classify",
        )

        st.markdown(
            '<div class="sec-label" style="margin-top:0.75rem">Quick examples</div>',
            unsafe_allow_html=True
        )

        ex_keys = list(EXAMPLES.keys())
        row1 = st.columns(4)
        row2 = st.columns(4)
        for i, label in enumerate(ex_keys):
            col = row1[i] if i < 4 else row2[i - 4]
            with col:
                if st.button(label, key=f"ex_{i}",
                             use_container_width=True):
                    st.session_state.input_text = EXAMPLES[label]
                    st.rerun()

        st.markdown("<div style='height:0.4rem'></div>",
                    unsafe_allow_html=True)

        go = st.button("Classify →", type="primary",
                       use_container_width=True)

        # ── Result display ───────────────────────────────────────────────────
        def show_result(ticket_text, cat, pri):
            st.markdown(f"""
            <div class="result-card">
                <div class="result-row">
                    <div class="result-key">Category</div>
                    <div class="result-val">{cat_badge(cat)}</div>
                </div>
                <div class="result-row">
                    <div class="result-key">Priority</div>
                    <div class="result-val">{pri_badge(pri)}</div>
                </div>
                <div class="result-row" style="padding-top:0.75rem;
                     margin-top:0.1rem;border-top:1.5px solid var(--border-color, #EEEEEE)">
                    <div class="result-key">Input</div>
                    <div class="result-val mono">
                        {ticket_text[:180]}{'…' if len(ticket_text)>180 else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if go:
            raw = ticket_input.strip()
            if not raw:
                st.warning("Please enter a ticket before classifying.")
            else:
                with st.spinner("Classifying…"):
                    cat, pri = classify_one(raw, cat_model, pri_model)
                st.session_state.history.insert(0, {
                    'ticket'  : raw,
                    'category': cat,
                    'priority': pri,
                    'time'    : datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state.n_classified += 1
                st.session_state.input_text = raw
                show_result(raw, cat, pri)
                st.rerun()

        elif st.session_state.history:
            last = st.session_state.history[0]
            show_result(
                last['ticket'],
                last['category'],
                last['priority']
            )

    with right:
        st.markdown('<div class="sec-label">Recent history</div>',
                    unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("""
            <div style="color: var(--text-color); opacity: 0.6; font-size:0.88rem;
                 padding:2.5rem 0;text-align:center;font-weight:500">
                No tickets classified yet.
            </div>""", unsafe_allow_html=True)
        else:
            rows = ""
            for h in st.session_state.history[:12]:
                short = (h['ticket'][:55] +
                         ('…' if len(h['ticket']) > 55 else ''))
                rows += f"""<tr>
                    <td class="mono" style="font-size:0.76rem; color: var(--text-color);">{short}</td>
                    <td>{cat_badge(h['category'])}</td>
                    <td>{pri_badge(h['priority'])}</td>
                </tr>"""
            st.markdown(f"""
            <table class="hist-tbl">
                <thead><tr>
                    <th>Ticket</th>
                    <th>Category</th>
                    <th>Priority</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

            if len(st.session_state.history) > 1:
                st.markdown(
                    "<div style='height:0.75rem'></div>",
                    unsafe_allow_html=True
                )
                if st.button("Clear history", key="clr"):
                    st.session_state.history      = []
                    st.session_state.n_classified = 0
                    st.session_state.input_text   = ''
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊   Analytics":

    st.markdown('<div class="page-title">Session analytics</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Live breakdown of every ticket '
        'classified in this session.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:
        st.info("No tickets classified yet. Go to **Classify** first.")
        st.stop()

    hist       = st.session_state.history
    total      = len(hist)
    pri_counts = Counter(h['priority'] for h in hist)
    cat_counts = Counter(h['category'] for h in hist)
    urgent_n   = (pri_counts.get('Critical', 0) +
                  pri_counts.get('High', 0))

    # ── Stats row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    top_cat  = cat_counts.most_common(1)[0][0] if cat_counts else "—"
    top_cat_n= cat_counts.most_common(1)[0][1] if cat_counts else 0
    top_pri  = pri_counts.most_common(1)[0][0] if pri_counts else "—"
    top_pri_n= pri_counts.most_common(1)[0][1] if pri_counts else 0

    for col, lbl, val, hint, accent in [
        (c1, "Total classified", total,    "this session",          False),
        (c2, "Urgent tickets",   urgent_n, "Critical + High",       True),
        (c3, "Top category",     top_cat,  f"{top_cat_n} tickets",  False),
        (c4, "Top priority",     top_pri,  f"{top_pri_n} tickets",  False),
    ]:
        with col:
            if accent:
                st.markdown(f"""
                <div class="stat-card-accent">
                    <div class="stat-lbl-inv">{lbl}</div>
                    <div class="stat-val-inv">{val}</div>
                    <div class="stat-hint-inv">{hint}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-lbl">{lbl}</div>
                    <div class="stat-val">{val}</div>
                    <div class="stat-hint">{hint}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>",
                unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    chart_l, chart_r = st.columns(2, gap="large")

    with chart_l:
        st.markdown('<div class="sec-label">Priority distribution</div>',
                    unsafe_allow_html=True)

        pri_order  = [p for p in ['Critical','High','Medium','Low']
                      if p in pri_counts]
        pri_vals   = [pri_counts[p] for p in pri_order]
        pri_colors = [PRI_COLOR[p] for p in pri_order]

        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        bars = ax.barh(
            pri_order[::-1], pri_vals[::-1],
            color=pri_colors[::-1], height=0.52, edgecolor='white'
        )
        for bar, val in zip(bars, pri_vals[::-1]):
            ax.text(
                bar.get_width() + 0.08,
                bar.get_y() + bar.get_height() / 2,
                str(val), va='center',
                fontsize=9.5, fontweight='600', color='#1A1A1A'
            )
        ax.set_xlim(0, max(pri_vals) * 1.3)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=10,
                       colors='#1A1A1A', length=0)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_r:
        st.markdown('<div class="sec-label">Category distribution</div>',
                    unsafe_allow_html=True)

        cat_labels = list(cat_counts.keys())
        cat_vals   = list(cat_counts.values())

        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        fig2.patch.set_facecolor('#FFFFFF')
        ax2.set_facecolor('#FFFFFF')
        wedges, texts, autotexts = ax2.pie(
            cat_vals,
            labels     = cat_labels,
            colors     = CAT_COLOR[:len(cat_labels)],
            autopct    = '%1.0f%%',
            startangle = 140,
            wedgeprops = {'edgecolor':'white','linewidth':2.5},
            textprops  = {'fontsize':9, 'color':'#1A1A1A',
                          'fontweight':'500'},
        )
        for at in autotexts:
            at.set_fontsize(8.5)
            at.set_color('white')
            at.set_fontweight('700')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Priority × Category heatmap ──────────────────────────────────────────
    st.markdown(
        '<div class="sec-label" style="margin-top:0.5rem">'
        'Priority × Category heatmap</div>',
        unsafe_allow_html=True
    )

    all_cats = sorted(set(h['category'] for h in hist))
    all_pris = [p for p in ['Critical','High','Medium','Low']
                if p in set(h['priority'] for h in hist)]

    matrix = np.zeros((len(all_pris), len(all_cats)), dtype=int)
    for h in hist:
        if h['priority'] in all_pris and h['category'] in all_cats:
            r = all_pris.index(h['priority'])
            c = all_cats.index(h['category'])
            matrix[r, c] += 1

    fig3, ax3 = plt.subplots(
        figsize=(9, max(2.5, len(all_pris) * 0.95))
    )
    fig3.patch.set_facecolor('#FFFFFF')
    ax3.set_facecolor('#FFFFFF')
    ax3.imshow(matrix, cmap='Blues', aspect='auto')

    for i in range(len(all_pris)):
        for j in range(len(all_cats)):
            val = matrix[i, j]
            ax3.text(
                j, i, str(val),
                ha='center', va='center',
                fontsize=11, fontweight='700',
                color='white' if val > matrix.max() / 2 else '#1A1A1A'
            )

    ax3.set_xticks(range(len(all_cats)))
    ax3.set_yticks(range(len(all_pris)))
    ax3.set_xticklabels(all_cats, fontsize=10, rotation=20,
                        ha='right', color='#1A1A1A', fontweight='500')
    ax3.set_yticklabels(all_pris, fontsize=10,
                        color='#1A1A1A', fontweight='500')
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(length=0)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    # ── Full history table ────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">Full session history</div>',
        unsafe_allow_html=True
    )

    rows = ""
    for h in hist:
        short = h['ticket'][:70] + ('…' if len(h['ticket']) > 70 else '')
        t     = h.get('time', '—')
        rows += f"""<tr>
            <td style="color: var(--text-color); opacity: 0.7; font-size:0.76rem; white-space:nowrap; font-weight:500">{t}</td>
            <td class="mono" style="font-size:0.76rem; color: var(--text-color);">{short}</td>
            <td>{cat_badge(h['category'])}</td>
            <td>{pri_badge(h['priority'])}</td>
        </tr>"""

    st.markdown(f"""
    <table class="hist-tbl">
        <thead><tr>
            <th>Time</th>
            <th>Ticket</th>
            <th>Category</th>
            <th>Priority</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.75rem'></div>",
                unsafe_allow_html=True)
    df_hist = pd.DataFrame(st.session_state.history)
    csv     = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇   Export session history as CSV",
        data      = csv,
        file_name = "session_history.csv",
        mime      = "text/csv",
    )