import streamlit as st
from datasets_search import search_datasets
from utils import plot_corr, plot_roc_auc

st.title("ML Streamlit Demo (Fixed Runtime)")

query = st.text_input("Search Kaggle datasets", placeholder="e.g. Titanic, Iris, MNIST")
if query:
    results = search_datasets(query)
    st.write("Top Results:", results)

st.write("Demo visualizations:")
plot_corr()
plot_roc_auc()
