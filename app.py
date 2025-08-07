import streamlit as st
import joblib
import pandas as p
from text_utils import optimized_text_processing

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
testing = p.read_csv("testingtexts.csv")

# Get 20 random messages as sample
sample_texts = testing['Messages'].sample(20, random_state=42).tolist()

# Configure Streamlit page
st.set_page_config(page_title="Spam Classifier", layout="centered")

# --- Initialize Session State ---
if "message" not in st.session_state:
    st.session_state.message = ""

if "clear_clicked" not in st.session_state:
    st.session_state.clear_clicked = False

# --- Sidebar: Sample text dropdown ---
with st.sidebar:
    st.markdown("### 🧪 Try Sample Messages")
    selected_sample = st.selectbox("Choose a sample text to test:", [""] + sample_texts)

    # If user selects a sample message, update session_state and rerun
    if selected_sample and selected_sample != st.session_state.message:
        st.session_state.message = selected_sample
        st.rerun()

# --- Handle Clear Button ---
if st.session_state.clear_clicked:
    st.session_state.message = ""
    st.session_state.clear_clicked = False
    st.rerun()

# --- Title and Instruction ---
st.markdown("<h1 style='text-align: center; font-size: 42px;'>📩 Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size: 30px;'>✍️ Enter your message below:</h3>", unsafe_allow_html=True)

# --- Text Area (MUST come after setting message) ---
st.text_area(
    label="",
    height=200,
    placeholder="Type or paste your message here...",
    key="message"
)

# --- Text Area Styling ---
st.markdown("""
    <style>
        textarea {
            font-family: 'Segoe UI', sans-serif;
            font-size: 18px !important;
            font-weight: 500;
            line-height: 1.5;
        }
        .stTextArea > div > textarea {
            border: 1.5px solid #999 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Buttons: Predict & Clear ---
col1, col_spacer, col2 = st.columns([1, 0.5, 1])

with col1:
    if st.button("Predict"):
        if st.session_state.message.strip() == "":
            st.warning("Please enter a message.")
        else:
            processed_msg = optimized_text_processing(st.session_state.message)
            vect_msg = vectorizer.transform([processed_msg])
            prediction = model.predict(vect_msg)[0]

            if prediction == 1:
                st.error("⚠️ It's a SPAM message!")
            else:
                st.success("✅ It's a NOT SPAM message.")

with col2:
    if st.button("Clear"):
        st.session_state.clear_clicked = True

# --- Button Styling ---
st.markdown("""
    <style>
        div.stButton > button {
            font-size: 20px;
            padding: 0.75em 3em;
        }
    </style>
""", unsafe_allow_html=True)
# --- Sample text data (generated manually, not from dataset)
sample_spam_texts = [
    "Win a FREE iPhone now! Click here!",
    "Congratulations, you've won $1000. Claim now!",
    "Get rich quick with this one simple trick.",
    "Exclusive deal just for you. Limited time!",
    "You've been selected for a special reward!",
    "URGENT: Update your account to avoid suspension.",
    "Claim your free vacation today!",
    "This is not a scam. Click to receive your prize.",
    "Earn money from home easily. No experience needed!",
    "Act fast! Your reward is waiting."
]

sample_nonspam_texts = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Please find the attached report for your review.",
    "Let me know your thoughts on the proposal.",
    "Happy birthday! Wishing you a great year ahead.",
    "Can you help me with the code review?",
    "Reminder: Your appointment is scheduled for 3 PM.",
    "Thanks for your help yesterday!",
    "Let’s catch up over coffee soon.",
    "Don’t forget to submit the assignment by Friday.",
    "Here are the meeting notes from today."
]

st.markdown("---")
st.markdown("### 💬 Try These Sample Messages")

col1, spacer, col2 = st.columns([1.2, 0.1, 1.2])

with col1:
    st.markdown("#### 🔴 Spam Samples")
    for i, msg in enumerate(sample_spam_texts, 1):
        st.markdown(f"""
        <div style='border:1px solid #BF3B37;
                    border-radius:6px;
                    padding:8px;
                    margin-bottom:6px;
                    background-color:#BF3B37;
                    font-size:15px;'>
            <b>{i}.</b> {msg}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("#### 🟢 Not Spam Samples")
    for i, msg in enumerate(sample_nonspam_texts, 1):
        st.markdown(f"""
        <div style='border:1px solid #510FFF;
                    border-radius:6px;
                    padding:8px;
                    margin-bottom:6px;
                    background-color:#510FFF
;
                    font-size:15px;'>
            <b>{i}.</b> {msg}
        </div>
        """, unsafe_allow_html=True)
