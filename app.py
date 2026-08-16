import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #1e293b 100%);
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(59,130,246,0.25),
            rgba(139,92,246,0.25)
        );
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 25px;
        text-align: center;
    }

    .hero h1 {
        font-size: 48px;
        margin-bottom: 10px;
        font-weight: 800;
    }

    .hero p {
        font-size: 18px;
        color: #cbd5e1;
    }

    /* Cards */
    .card {
        padding: 25px;
        border-radius: 20px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }

    /* Generated text */
    .generated-box {
        padding: 30px;
        border-radius: 20px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(96,165,250,0.35);
        margin-top: 20px;
        font-size: 23px;
        line-height: 1.7;
        color: #f8fafc;
    }

    /* Section title */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Metric cards */
    .metric-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #94a3b8;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    lstm_model = load_model("lstm_model.h5")
    rnn_model = load_model("rnn_model.h5")

    return lstm_model, rnn_model


@st.cache_resource
def load_tokenizer():

    with open("tokienizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


@st.cache_resource
def load_max_len():

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return max_len


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    lstm_model, rnn_model = load_models()
    tokenizer = load_tokenizer()
    max_len = load_max_len()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "Unable to load the model files. "
        "Make sure all .h5 and .pkl files are in the same folder as app.py."
    )

    st.code(str(e))


# ============================================================
# WORD DICTIONARY
# ============================================================

if models_loaded:

    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
    }


# ============================================================
# PREDICT NEXT WORD
# ============================================================

def predict_next_word(model, text):

    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return ""

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_word_index = np.argmax(prediction)

    predicted_word = index_to_word.get(
        predicted_word_index,
        ""
    )

    return predicted_word


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(model, seed_text, next_words):

    generated_text = seed_text.strip()

    for _ in range(next_words):

        next_word = predict_next_word(
            model,
            generated_text
        )

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <h1>🧠 AI Quote Generator</h1>

    <p>
        Generate meaningful text using Deep Learning
        with LSTM and RNN models
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Model Settings")

    model_choice = st.selectbox(
        "Choose Deep Learning Model",
        [
            "LSTM",
            "RNN"
        ]
    )

    next_words = st.slider(
        "Number of words to generate",
        min_value=1,
        max_value=30,
        value=10
    )

    st.divider()

    st.markdown("### 🧠 Model Information")

    if model_choice == "LSTM":

        st.info(
            """
            **LSTM Model**

            Embedding → LSTM → Dense

            • Embedding Dimension: 50  
            • LSTM Units: 128  
            • Vocabulary: 10,000
            """
        )

    else:

        st.info(
            """
            **RNN Model**

            Embedding → SimpleRNN → Dense

            • Embedding Dimension: 50  
            • RNN Units: 128  
            • Vocabulary: 10,000
            """
        )

    st.divider()

    st.markdown("### 📊 Dataset / Tokenizer")

    st.write(
        f"Vocabulary: **{len(tokenizer.word_index):,}**"
    )

    st.write(
        f"Maximum sequence length: **{max_len}**"
    )

    st.divider()

    st.caption(
        "Deep Learning Text Generation Project"
    )


# ============================================================
# MAIN INPUT
# ============================================================

st.markdown(
    '<div class="section-title">✍️ Start Your Quote</div>',
    unsafe_allow_html=True
)

seed_text = st.text_area(
    "Enter a starting phrase",
    placeholder="Example: life is",
    height=140,
    label_visibility="collapsed"
)


# ============================================================
# EXAMPLE PROMPTS
# ============================================================

st.markdown("### 💡 Try an example")

col1, col2, col3, col4 = st.columns(4)

examples = [
    "life is",
    "success is",
    "never give",
    "the future"
]

if col1.button("✨ Life", use_container_width=True):
    seed_text = examples[0]

if col2.button("🚀 Success", use_container_width=True):
    seed_text = examples[1]

if col3.button("🔥 Motivation", use_container_width=True):
    seed_text = examples[2]

if col4.button("🌟 Future", use_container_width=True):
    seed_text = examples[3]


# ============================================================
# GENERATE BUTTON
# ============================================================

st.markdown("")

generate_button = st.button(
    "🚀 Generate Quote",
    type="primary",
    use_container_width=True
)


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    if not seed_text.strip():

        st.warning(
            "Please enter some starting text first."
        )

    else:

        if model_choice == "LSTM":
            selected_model = lstm_model
        else:
            selected_model = rnn_model

        with st.spinner(
            f"🤖 {model_choice} is generating your text..."
        ):

            generated_text = generate_text(
                selected_model,
                seed_text,
                next_words
            )

        st.success("Generation completed!")


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">✨ Generated Quote</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="generated-box">
                “{generated_text}”
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        st.markdown("### 📊 Generation Details")

        words = generated_text.split()

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {len(words)}
                    </div>
                    <div class="metric-label">
                        Total Words
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {next_words}
                    </div>
                    <div class="metric-label">
                        Requested Words
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with metric3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {model_choice}
                    </div>
                    <div class="metric-label">
                        Model Used
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # COPYABLE TEXT
        # ====================================================

        st.markdown("### 📋 Copy Your Quote")

        st.code(
            generated_text,
            language="text"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔬 Model Comparison</div>',
    unsafe_allow_html=True
)

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:

    st.markdown(
        """
        <div class="card">

        ### 🧠 LSTM

        **Architecture**

        Embedding  
        ↓  
        LSTM (128 units)  
        ↓  
        Dense (10,000)

        **Best for:**  
        Capturing longer-term dependencies
        in sequential text.

        </div>
        """,
        unsafe_allow_html=True
    )


with comparison_col2:

    st.markdown(
        """
        <div class="card">

        ### 🔄 Simple RNN

        **Architecture**

        Embedding  
        ↓  
        SimpleRNN (128 units)  
        ↓  
        Dense (10,000)

        **Best for:**  
        Simple sequential text-generation
        tasks.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🧠 AI Quote Generator |
        Built with TensorFlow + Streamlit

        <br><br>

        LSTM • RNN • NLP • Deep Learning

    </div>
    """,
    unsafe_allow_html=True
)