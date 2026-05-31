import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI News Classification System",
    page_icon="📰",
    layout="wide",
)

MAX_LEN = 100

LABELS = {
    0: "🌍 World",
    1: "🏆 Sports",
    2: "💼 Business",
    3: "🔬 Science & Technology"
}

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Container */
.block-container{
    padding-top:1rem;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg,#2563EB,#06B6D4);
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    animation: fadeIn 1s ease-in-out;
}

.hero h1{
    font-size:42px;
    font-weight:700;
    margin-bottom:10px;
}

.hero p{
    font-size:18px;
    opacity:0.95;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0F172A;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Buttons */
.stButton button{
    background: linear-gradient(90deg,#2563EB,#06B6D4);
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:600;
    height:50px;
    transition:0.3s;
}

.stButton button:hover{
    transform:scale(1.02);
}

/* Metric Cards */
[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

/* Cards */
.custom-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-top:10px;
}

/* Animation */
@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(15px);
    }
    to{
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():

    model = load_model("news_classifier.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

try:
    model, tokenizer = load_artifacts()

except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📰 News AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📜 Prediction History",
        "ℹ️ About Model"
    ]
)

st.sidebar.markdown("---")

example = st.sidebar.selectbox(
    "📌 Quick Examples",
    [
        "Select Example",
        "World",
        "Sports",
        "Business",
        "Technology"
    ]
)

examples = {

    "World":
    "World leaders gathered at the climate summit to discuss reducing carbon emissions.",

    "Sports":
    "India won the cricket series after defeating Australia in the final match.",

    "Business":
    "Stock markets rose sharply after stronger-than-expected earnings reports.",

    "Technology":
    "Scientists developed a new AI system capable of early disease detection."
}

default_text = examples.get(example, "")

# =====================================================
# HISTORY PAGE
# =====================================================

if menu == "📜 Prediction History":

    st.title("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            csv = history_df.to_csv(index=False)

            st.download_button(
                "📥 Download History",
                csv,
                "prediction_history.csv",
                "text/csv"
            )

        with col2:
            if st.button("🗑 Clear History"):
                st.session_state.history = []
                st.rerun()

    else:
        st.info("No prediction history available.")

    st.stop()

# =====================================================
# ABOUT PAGE
# =====================================================

if menu == "ℹ️ About Model":

    st.title("ℹ️ About This Model")

    st.markdown("""
    ### AI News Classification System

    This application uses a Deep Learning LSTM model to classify news articles into:

    - 🌍 World
    - 🏆 Sports
    - 💼 Business
    - 🔬 Science & Technology

    ### Technology Stack

    - TensorFlow / Keras
    - LSTM Neural Network
    - Streamlit
    - Plotly

    ### Features

    ✅ Real-time prediction  
    ✅ Confidence score  
    ✅ Interactive visualization  
    ✅ Prediction history  
    ✅ Downloadable history report  
    """)

    st.stop()

# =====================================================
# HOME PAGE
# =====================================================

st.markdown("""
<div class="hero">
    <h1>📰 AI News Classification System</h1>
    <p>Deep Learning Powered News Categorization using TensorFlow LSTM</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================

st.markdown("### ✍️ Enter News Article / Headline")

news_text = st.text_area(
    "",
    value=default_text,
    height=220,
    placeholder="Paste news article or headline here..."
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_news(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probs = model.predict(
        padded,
        verbose=0
    )[0]

    pred_class = np.argmax(probs)

    return pred_class, probs

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("🚀 Predict Category", use_container_width=True):

    if not news_text.strip():

        st.warning("⚠ Please enter a news article.")

    else:

        with st.spinner("🔍 Analyzing article..."):

            pred_class, probs = predict_news(news_text)

            confidence = probs[pred_class] * 100

        # =================================================
        # METRICS
        # =================================================

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Category",
                LABELS[pred_class]
            )

        with col2:
            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        # =================================================
        # SAVE HISTORY
        # =================================================

        st.session_state.history.append({

            "News":
            news_text[:120],

            "Prediction":
            LABELS[pred_class],

            "Confidence":
            f"{confidence:.2f}%"
        })

        # =================================================
        # CHART
        # =================================================

        st.markdown("### 📊 Category Probability Analysis")

        df = pd.DataFrame({
            "Category": list(LABELS.values()),
            "Probability": probs * 100
        })

        fig = px.bar(
            df,
            x="Category",
            y="Probability",
            text="Probability",
            title="Prediction Probability Distribution"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=500,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # DETAILED SCORES
        # =================================================

        st.markdown("### 📈 Detailed Confidence Scores")

        for i, prob in enumerate(probs):

            st.write(
                f"**{LABELS[i]}** — {prob*100:.2f}%"
            )

            st.progress(float(prob))

# =====================================================
# SAMPLE NEWS
# =====================================================

st.markdown("### 📋 Sample News Articles")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌍 World",
        "🏆 Sports",
        "💼 Business",
        "🔬 Technology"
    ]
)

with tab1:
    st.info("World leaders gathered at the climate summit to discuss reducing carbon emissions.")
    st.info("The two countries signed a peace agreement after months of negotiations.")
    st.info("The United Nations announced additional humanitarian aid for refugees.")

with tab2:
    st.info("India won the cricket series after defeating Australia in the final match.")
    st.info("The football team secured a dramatic victory in the championship final.")
    st.info("The athlete set a new world record in the 100-meter sprint.")

with tab3:
    st.info("Stock markets rose sharply after stronger-than-expected earnings reports.")
    st.info("The company announced plans to expand operations across Asia.")
    st.info("The startup secured $50 million in funding from venture capital firms.")

with tab4:
    st.info("Scientists developed a new AI system capable of early disease detection.")
    st.info("Researchers discovered a potentially habitable planet outside our solar system.")
    st.info("A breakthrough in quantum computing could accelerate scientific research.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<hr>

<div style='text-align:center;color:gray;'>

© 2026 AI News Classification System <br>
Powered by TensorFlow • Keras • Streamlit

</div>
""", unsafe_allow_html=True)