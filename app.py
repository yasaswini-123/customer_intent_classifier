import html
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Intent Classification",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# Folders that may contain the three CSV files (checked in this order).
DATASET_FOLDERS = ["datasets", "archive (1)", "archive"]

# Predictions with a lower confidence than this are shown as inconclusive.
CONFIDENCE_THRESHOLD = 0.25

EXAMPLES = {
    "Reset password": "I want to reset my password",
    "Track order": "Where is my order? I want to track it",
    "Cancel order": "I want to cancel my order",
    "Get a refund": "I would like to get a refund for my purchase",
    "Change address": "How can I change my shipping address?",
}


# ==================================================
# 2. HELPER: RENDER HTML WITHOUT MARKDOWN CODE BLOCKS
# ==================================================
# st.markdown treats indented lines and anything after a blank line as a
# code block. This helper removes indentation and blank lines first, so the
# HTML is rendered as HTML.

def render_html(markup):
    cleaned = "\n".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# ==================================================
# 3. WHITE INTERFACE DESIGN
# ==================================================

st.markdown("""
<style>
/* ---------- White page ---------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: #ffffff !important;
    color: #172033 !important;
}

[data-testid="stHeader"] * {
    color: #475569 !important;
}

.block-container {
    max-width: 860px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

/* ---------- Text ---------- */
h1, h2, h3, h4 {
    color: #173b70 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
label, .stCaption {
    color: #334155 !important;
}

/* ---------- Header card ---------- */
.hero {
    background: #f1f6ff;
    border: 1px solid #dbe7f7;
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 22px;
}

.hero h1 {
    margin: 0 0 8px 0;
    font-size: 34px;
    color: #173b70 !important;
}

.hero p {
    margin: 0;
    color: #52647d !important;
    line-height: 1.7;
}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {
    background: #f8fbff;
    border: 1px solid #dbe7f7;
    border-radius: 14px;
    padding: 16px 18px;
}

[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
}

/* ---------- Text box ---------- */
div[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: #ffffff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    padding: 4px 10px !important;
}

.stButton > button:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

/* ---------- Result cards ---------- */
.result-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 24px 26px;
    margin: 18px 0 8px 0;
}

.result-card.warn {
    background: #fffbeb;
    border-color: #fde68a;
}

.result-label {
    color: #52647d !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
}

.result-value {
    color: #1d4ed8 !important;
    font-size: 32px;
    font-weight: 700;
    margin: 6px 0 4px 0;
}

.result-card.warn .result-value {
    color: #b45309 !important;
}

.result-meta {
    color: #475569 !important;
    font-size: 15px;
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e7f0;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #173b70 !important;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #8490a3;
    border-top: 1px solid #e5e7eb;
    padding-top: 18px;
    margin-top: 34px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. LOAD DATA AND TRAIN THE MODEL (runs once)
# ==================================================

def find_dataset_dir():
    for name in DATASET_FOLDERS:
        folder = BASE_DIR / name
        if folder.is_dir() and list(folder.glob("*.csv")):
            return folder

    searched = ", ".join(str(BASE_DIR / n) for n in DATASET_FOLDERS)
    raise FileNotFoundError(
        "Dataset folder not found. Put the three CSV files inside a "
        f"'datasets' folder next to app.py. Searched: {searched}"
    )


def find_csv(files, keyword):
    match = next(
        (f for f in files if keyword in f.name.lower()),
        None,
    )

    if match is None:
        raise FileNotFoundError(
            f"No CSV file with '{keyword}' in its name was found."
        )

    return match


@st.cache_resource(show_spinner=False)
def load_model():
    dataset_dir = find_dataset_dir()
    files = sorted(dataset_dir.glob("*.csv"))

    train_df = pd.read_csv(find_csv(files, "training"))
    validation_df = pd.read_csv(find_csv(files, "validation"))
    test_df = pd.read_csv(find_csv(files, "testing"))

    df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    # Clean data
    df = df.dropna(subset=["utterance", "intent"]).copy()
    df["utterance"] = df["utterance"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["utterance"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(subset=["utterance", "intent"])

    X = df["utterance"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

    # intent -> category (for example cancel_order -> ORDER)
    if "category" in df.columns:
        category_of = (
            df.groupby("intent")["category"]
            .agg(lambda s: s.mode().iloc[0])
            .to_dict()
        )
    else:
        category_of = {}

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy,
        "records": len(df),
        "intents": df["intent"].nunique(),
        "category_of": category_of,
    }


def pretty(label):
    return str(label).replace("_", " ").strip().title()


# ==================================================
# 5. HEADER
# ==================================================

render_html("""
<div class="hero">
    <h1>💬 Customer Intent Classification</h1>

    <p>
        Type a customer message and the model predicts what the customer
        wants, for example cancelling an order, tracking a refund or
        recovering a password.
    </p>
</div>
""")


# ==================================================
# 6. TRAIN / LOAD
# ==================================================

try:
    with st.spinner("Training the model... this takes a few seconds."):
        bundle = load_model()

except Exception as error:
    st.error(f"Could not load the dataset or train the model: {error}")
    st.stop()

vectorizer = bundle["vectorizer"]
model = bundle["model"]
category_of = bundle["category_of"]

col1, col2, col3 = st.columns(3)
col1.metric("Model accuracy", f"{bundle['accuracy'] * 100:.2f}%")
col2.metric("Dataset records", f"{bundle['records']:,}")
col3.metric("Intents", bundle["intents"])


# ==================================================
# 7. MESSAGE INPUT
# ==================================================

st.write("")
st.subheader("Customer message")

st.session_state.setdefault("message", "")


def use_example(text):
    st.session_state["message"] = text


st.caption("Try an example:")

example_cols = st.columns(len(EXAMPLES))

for index, (label, text) in enumerate(EXAMPLES.items()):
    example_cols[index].button(
        label,
        key=f"example_{index}",
        on_click=use_example,
        args=(text,),
        use_container_width=True,
    )

message = st.text_area(
    "Customer message",
    key="message",
    height=130,
    placeholder="Example: I want to reset my password",
    label_visibility="collapsed",
)

predict_clicked = st.button("Predict intent", type="primary")


# ==================================================
# 8. PREDICTION
# ==================================================

if predict_clicked:

    if not message.strip():
        st.warning("Please enter a customer message first.")

    else:
        features = vectorizer.transform([message])
        probabilities = model.predict_proba(features)[0]

        order = probabilities.argsort()[::-1][:3]
        top_labels = [model.classes_[i] for i in order]
        top_scores = [float(probabilities[i]) for i in order]

        best_label = top_labels[0]
        best_score = top_scores[0]
        category = category_of.get(best_label, "")

        if best_score < CONFIDENCE_THRESHOLD:
            guesses = ", ".join(pretty(label) for label in top_labels[:2])

            render_html(f"""
            <div class="result-card warn">
                <div class="result-label">INCONCLUSIVE PREDICTION</div>
                <div class="result-value">Not sure</div>
                <div class="result-meta">
                    The model is not confident about this message
                    ({best_score * 100:.1f}% for the best guess).
                    Its closest guesses are: {html.escape(guesses)}.
                    Try rewording the message with more detail.
                </div>
            </div>
            """)

        else:
            category_text = (
                f" &nbsp;|&nbsp; Category: {html.escape(pretty(category))}"
                if category else ""
            )

            render_html(f"""
            <div class="result-card">
                <div class="result-label">PREDICTED INTENT</div>
                <div class="result-value">{html.escape(pretty(best_label))}</div>
                <div class="result-meta">
                    Confidence: {best_score * 100:.1f}%{category_text}
                </div>
            </div>
            """)

        st.markdown("**Top predictions**")

        for label, score in zip(top_labels, top_scores):
            st.write(f"{pretty(label)}: {score * 100:.1f}%")
            st.progress(min(max(score, 0.0), 1.0))

        st.caption(
            "Confidence is the model's estimated probability, not a "
            "guarantee of correctness. Predictions below "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% are marked inconclusive."
        )


# ==================================================
# 9. ABOUT SECTION
# ==================================================

st.write("")

with st.expander("About this model"):
    st.write(
        "The message is converted to numbers with TF-IDF (single words and "
        "two-word phrases) and classified with Logistic Regression. The "
        "model is trained on the Bitext sample customer service dataset "
        "(training, validation and testing files combined, duplicates "
        "removed, then split 80/20). The accuracy shown above is measured "
        "on the 20% that the model did not see during training."
    )

with st.expander("Intents the model can recognise"):
    by_category = {}

    for intent, category in category_of.items():
        by_category.setdefault(category, []).append(intent)

    if by_category:
        for category in sorted(by_category):
            names = ", ".join(pretty(i) for i in sorted(by_category[category]))
            st.markdown(f"**{pretty(category)}**: {names}")
    else:
        st.write(", ".join(pretty(c) for c in model.classes_))

render_html("""
<div class="footer">
    Customer Service Intent Classification | TF-IDF + Logistic Regression
</div>
""")