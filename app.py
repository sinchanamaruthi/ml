import streamlit as st
import pandas as pd
from datasets_search import load_openml_dataset, load_huggingface_dataset
from preprocessing import preprocess_data
from models import train_and_evaluate
from utils import plot_correlation_matrix

st.set_page_config(page_title="ML Playground", layout="wide")

st.title("🚀 Machine Learning Playground")

# Dataset selection
st.sidebar.header("📂 Dataset Options")
dataset_source = st.sidebar.radio("Choose dataset source:", ["Upload CSV", "OpenML", "Hugging Face"])

df = None

if dataset_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif dataset_source == "OpenML":
    openml_id = st.sidebar.text_input("Enter OpenML dataset ID", "61")
    st.sidebar.caption("👉 Example: 61 = Iris dataset")
    if st.sidebar.button("Load from OpenML"):
        df = load_openml_dataset(openml_id)

elif dataset_source == "Hugging Face":
    hf_name = st.sidebar.text_input("Enter Hugging Face dataset name", "imdb")
    st.sidebar.caption("👉 Example: imdb = sentiment analysis dataset")
    if st.sidebar.button("Load from Hugging Face"):
        df = load_huggingface_dataset(hf_name)

if df is not None:
    st.write("### 📊 Dataset Preview", df.head())
    st.write("Shape:", df.shape)

    target_col = st.selectbox("🎯 Select target column", df.columns)
    if target_col:
        X_train, X_test, y_train, y_test = preprocess_data(df, target_col)
        results = train_and_evaluate(X_train, X_test, y_train, y_test)

        st.write("### 📈 Model Results")
        for model, metrics in results.items():
            st.write(f"**{model}** → {metrics}")

        st.write("### 🔎 Correlation Matrix")
        st.pyplot(plot_correlation_matrix(df))
